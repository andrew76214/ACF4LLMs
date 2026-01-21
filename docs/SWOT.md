# Green_AI 專案研究論文框架分析

## 一、嘗試解決的問題是什麼？

### 問題陳述
Green_AI (Agentic Compression Framework) 試圖解決的核心問題是：

**大型語言模型 (LLM) 壓縮的自動化決策問題**

具體而言：
1. **壓縮方法選擇困難**：現有的 LLM 壓縮方法眾多（量化、剪枝、蒸餾、LoRA 等），每種方法都有不同的參數空間，人工選擇需要大量專業知識
2. **多目標權衡複雜**：壓縮需要同時考慮準確度、延遲、記憶體、模型大小、能源消耗等多個相互衝突的目標
3. **缺乏自主優化**：現有工具大多需要人工反覆嘗試，無法自主探索策略空間
4. **硬體適配挑戰**：不同硬體環境對壓縮策略的要求不同，需要針對性調整

### 研究目標
設計一個 **LLM 驅動的多代理系統**，只需最少輸入（模型名稱 + 資料集），即可：
- 自動推斷模型規格與硬體需求
- 由 GPT-4o 自主決定壓縮策略
- 執行多種壓縮方法（量化、剪枝、LoRA/QLoRA）
- 追蹤 Pareto 最優解，平衡多個優化目標

---

## 二、如何證明這個問題的存在？

### 2.1 問題的衡量方法

Green_AI 使用以下標準化 benchmark 來衡量壓縮效果：

| Benchmark | 用途 | 評估指標 |
|-----------|------|----------|
| **GSM8K** | 數學推理能力 | Exact Match Accuracy |
| **CommonsenseQA** | 常識推理能力 | Accuracy (%) |
| **TruthfulQA** | 真實性/避免幻覺 | MC1/MC2 Score |
| **HumanEval** | 程式碼生成 | pass@1 |
| **BIG-Bench Hard** | 複雜推理任務 | Accuracy |
| **MMLU, HellaSwag, ARC, WinoGrande** | 通用能力 | Accuracy (via lm-eval) |

### 2.2 多目標優化指標

系統追蹤的關鍵指標（定義於 `src/common/schemas.py:90-116`）：

```python
class EvaluationResult:
    accuracy: float         # 準確度（越高越好）
    latency_ms: float       # 推理延遲（越低越好）
    memory_gb: float        # 記憶體使用（越低越好）
    model_size_gb: float    # 模型大小（越低越好）
    energy_joules: float    # 能源消耗（越低越好）
    compression_ratio: float # 壓縮比（越高越好）
```

### 2.3 Pareto 前緣追蹤

系統使用 Pareto 優化來處理多目標衝突（`src/coordinator/pareto.py:29-77`）：
- 解決方案 A 被 B **支配** 當且僅當 B 在所有目標上至少一樣好，且在至少一個目標上更好
- 非支配解構成 Pareto 前緣，代表不同權衡下的最優解

### 2.4 參考文獻建議

建議引用以下文獻來證明問題的存在：
- **量化精度損失**: Dettmers et al., "LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale" (NeurIPS 2022)
- **多目標優化必要性**: Cai et al., "Once-for-All: Train One Network and Specialize it for Efficient Deployment" (ICLR 2020)
- **自動化壓縮需求**: Wang et al., "HAQ: Hardware-Aware Automated Quantization with Mixed Precision" (CVPR 2019)

---

## 三、別人是怎麼解決這個問題的？

### 3.1 現有方法分類

| 方法類別 | 代表工具/論文 | 優點 | 缺點 |
|----------|--------------|------|------|
| **手動量化工具** | GPTQ, AWQ, AutoRound | 壓縮效果好 | 需要專業知識選擇參數 |
| **自動化 NAS** | Once-for-All, HAQ | 自動搜尋架構 | 計算成本高，不適用於 LLM |
| **單一目標優化** | 各種 PTQ 工具 | 簡單直接 | 無法處理多目標權衡 |
| **啟發式搜尋** | AutoML-based | 可自動化 | 缺乏對 LLM 特性的理解 |

### 3.2 現有方法的不足

1. **缺乏智能決策**：現有工具大多是「工具」而非「系統」，不具備自主決策能力
2. **單一方法侷限**：大多專注於單一壓縮技術，無法組合多種方法（如先剪枝再量化再 LoRA 恢復）
3. **無法學習歷史**：不會從過去的實驗中學習，每次都從頭開始
4. **硬體無感知**：不考慮目標硬體的特性

### 3.3 為什麼需要新方法？

Green_AI 的創新點：
1. **LLM 驅動決策**：使用 GPT-4o 作為「壓縮專家」，具備推理和規劃能力
2. **多代理協作**：分工明確的 Agent 系統（Coordinator + Method Agents）
3. **Pipeline 組合**：支援多步驟壓縮管線（Pruning → Quantization → LoRA）
4. **技能學習**：從歷史實驗中學習，提供更好的策略建議

---

## 四、解決問題的方法與實現

### 4.1 系統架構

```
                    ┌─────────────────┐
                    │     START       │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │  Coordinator    │ ◄─────────────────┐
                    │   (GPT-4o)      │                   │
                    │  分析 Pareto    │                   │
                    │  決定下一步     │                   │
                    └────────┬────────┘                   │
                             │                            │
              ┌──────────────┼──────────────┐             │
              ▼              ▼              ▼             │
     ┌─────────────┐ ┌─────────────┐ ┌─────────────┐     │
     │ Quantization│ │  Pruning    │ │  Pipeline   │     │
     │   Node      │ │   Node      │ │   Node      │     │
     └──────┬──────┘ └──────┬──────┘ └──────┬──────┘     │
            │               │               │             │
            └───────────────┼───────────────┘             │
                            ▼                             │
                   ┌─────────────────┐                    │
                   │   Evaluation    │                    │
                   │     Node        │                    │
                   └────────┬────────┘                    │
                            ▼                             │
                   ┌─────────────────┐                    │
                   │  Update State   │────────────────────┘
                   │  (Pareto 更新)  │
                   └────────┬────────┘
                            │
                            ▼ (收斂或預算耗盡)
                   ┌─────────────────┐
                   │      END        │
                   └─────────────────┘
```

### 4.2 核心組件

| 組件 | 位置 | 功能 |
|------|------|------|
| **LangGraph Coordinator** | `src/coordinator/langgraph_coordinator.py` | GPT-4o 驅動的主協調器 |
| **State Schema** | `src/coordinator/state.py` | LangGraph 狀態定義 |
| **Spec Inference** | `src/coordinator/spec_inference.py` | 從模型名稱推斷規格 |
| **Pareto Frontier** | `src/coordinator/pareto.py` | 多目標優化追蹤 |
| **Quantization Agent** | `src/agents/quantization_agent.py` | 量化工具（AutoRound, GPTQ, AWQ, INT8） |
| **Pruning Agent** | `src/agents/pruning_agent.py` | 剪枝工具（Magnitude, Structured） |
| **Evaluation Agent** | `src/agents/evaluation_agent.py` | Benchmark 評估 |
| **Search Agent** | `src/agents/search_agent.py` | 策略搜尋（Bayesian, Evolutionary, Bandit） |
| **Skill System** | `src/skills/` | 技能發現與學習 |

### 4.3 工作流程（Episode-Based）

每個優化 episode 的流程（定義於 `langgraph_coordinator.py:326-477`）：

1. **Coordinator 決策**：GPT-4o 分析當前 Pareto 前緣、歷史記錄、技能建議
2. **選擇動作**：quantization / lora / qlora / pruning / pipeline / search / end
3. **執行壓縮**：對應的 Agent 執行壓縮操作
4. **評估結果**：在 benchmark 上評估壓縮後的模型
5. **更新狀態**：更新 Pareto 前緣、記錄到技能記憶
6. **迴圈判斷**：直到收斂、預算耗盡或達到最大 episode

### 4.4 支援的壓縮方法

```python
class CompressionMethod(str, Enum):
    AUTOROUND = "autoround"   # Intel AutoRound (2/3/4/8-bit)
    GPTQ = "gptq"             # GPTQ 量化 (2/3/4/8-bit)
    INT8 = "int8"             # BitsAndBytes 8-bit
    AWQ = "awq"               # Activation-aware 量化 (4-bit)
    PRUNING = "pruning"       # 權重剪枝
    DISTILLATION = "distillation"  # 知識蒸餾（計劃中）
    LORA = "lora"             # LoRA 微調
    QLORA = "qlora"           # QLoRA (4-bit + LoRA)
```

### 4.5 使用方式

```bash
# 最簡單的用法
python scripts/run_pipeline.py --model gpt2 --dataset gsm8k

# 完整選項
python scripts/run_pipeline.py \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --dataset gsm8k \
    --episodes 10 \
    --budget 4.0 \
    --interactive
```

### 4.6 輸出結構

```
data/experiments/{experiment_name}/
├── model_spec.json           # 推斷的模型規格
├── pareto_frontier.json      # Pareto 最優解
├── final_results.json        # 優化摘要
├── pareto_visualization.html # 互動式視覺化
└── episode_XXX/
    ├── strategy.json         # 壓縮策略
    └── results.json          # 評估結果
```

---

## 五、驗證結果

### 5.1 評估方法

系統使用 EleutherAI 的 `lm-eval` harness 進行標準化評估（`src/evaluation/evaluators/lm_eval_evaluator.py`）：

```python
# 支援的 lm-eval benchmarks
SUPPORTED_LMEVAL_TASKS = {
    "mmlu": "mmlu",
    "hellaswag": "hellaswag",
    "arc_easy": "arc_easy",
    "arc_challenge": "arc_challenge",
    "winogrande": "winogrande",
    "gsm8k": "gsm8k_cot",
    "truthfulqa": "truthfulqa_mc2",
}
```

### 5.2 評估指標一致性

為了前後一致，驗證使用與問題定義相同的指標：
- **準確度**：各 benchmark 的 accuracy/pass@1
- **延遲**：ms/token
- **記憶體**：GB
- **模型大小**：GB
- **壓縮比**：原始大小 / 壓縮後大小

### 5.3 Pareto 前緣視覺化

系統自動生成互動式 Pareto 前緣視覺化（使用 Plotly）：
- Accuracy vs Latency
- Accuracy vs Memory
- Accuracy vs Model Size
- Latency vs Memory

---

## 六、討論

### 6.1 優勢

1. **自主決策**：GPT-4o 能夠根據上下文做出合理的壓縮決策，無需人工干預
2. **多方法組合**：Pipeline 支援多步驟壓縮，可以先剪枝、再量化、再用 LoRA 恢復精度
3. **技能學習**：系統會從歷史實驗中學習，提供更好的策略建議
4. **標準化評估**：整合 lm-eval，確保評估結果可比較

### 6.2 潛在改進空間

1. **Spec Inference**：目前使用靜態資料庫，可改為從 HuggingFace Hub API 動態取得
2. **Search Agent**：可整合更先進的搜尋演算法
3. **蒸餾支援**：Knowledge Distillation 尚未完整實現
4. **能源追蹤**：codecarbon 整合可更精確

---

## 七、結論、限制與未來方向

### 7.1 結論

Green_AI (Agentic Compression Framework) 提出了一個 LLM 驅動的多代理壓縮系統，能夠：
- 自動化 LLM 壓縮的策略選擇
- 透過多目標優化找到 Pareto 最優解
- 支援多種壓縮方法的組合與學習

### 7.2 限制 (Limitations)

1. **依賴 GPT-4o API**：需要 OpenAI API 金鑰，有成本考量
2. **計算資源需求**：完整評估需要較大的 GPU 記憶體
3. **Benchmark 覆蓋**：部分 benchmark（如 BIG-Bench Hard）的子集有限
4. **蒸餾未實現**：Distillation Agent 尚在計劃中
5. **硬體適配有限**：目前主要針對 NVIDIA GPU 優化

### 7.3 未來研究方向

根據 `TODO.md` 和系統架構分析，可能的研究方向包括：

1. **動態 Spec Inference**：從 HuggingFace Hub API 即時獲取模型資訊
2. **進階搜尋演算法**：Population-based Training (PBT)、更好的 Bayesian Optimization
3. **知識蒸餾整合**：實現完整的 teacher-student 訓練流程
4. **MLflow 整合**：更好的實驗追蹤與模型 registry
5. **Streamlit Dashboard**：即時監控壓縮進度
6. **Docker 部署**：簡化部署流程
7. **邊緣裝置優化**：針對特定硬體（如 Apple Silicon、Intel NPU）的優化

---

## 附錄：關鍵檔案索引

| 檔案 | 行數 | 說明 |
|------|------|------|
| `src/coordinator/langgraph_coordinator.py` | 1-1159 | 主協調器 |
| `src/coordinator/pareto.py` | 1-480 | Pareto 前緣追蹤 |
| `src/coordinator/state.py` | - | 狀態定義 |
| `src/common/schemas.py` | 1-179 | 資料模型定義 |
| `src/agents/quantization_agent.py` | - | 量化工具 |
| `src/agents/pruning_agent.py` | - | 剪枝工具 |
| `src/evaluation/evaluators/lm_eval_evaluator.py` | - | lm-eval 整合 |
| `README.md` | 1-322 | 專案說明（中文） |
| `SPEC.md` | 1-583 | 架構規格（中文） |
| `TODO.md` | 1-90 | 待辦事項 |
