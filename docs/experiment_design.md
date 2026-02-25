# 實驗設計

## 0. 實驗前置作業

### 0.1 系統缺口與必要修改

以下列出執行實驗前必須完成的程式碼修改。標記 ✅ 表示已實作、⚠️ 表示部分實作、❌ 表示尚未實作。

| # | 項目 | 狀態 | 檔案 | 影響的 RQ | 說明 |
|---|------|------|------|-----------|------|
| P1 | 新增 `--seed` CLI 參數 | ❌ | `scripts/run_pipeline.py` | RQ1, RQ2, RQ3 | 目前 CLI 不支援設定隨機種子，所有 RQ 均需要此參數以確保可重現性。需將 seed 傳遞至 LangGraphCoordinator 與 GPT-4o temperature 設定。 |
| P2 | 新增 `energy_focused` 預設 | ❌ | `src/common/config.py:373` | RQ2 | 在 `from_preset()` 中新增 energy_focused 配置（見 RQ2 第 7 節）。 |
| P3 | 修正 `compute_bandit_reward()` | ❌ | `src/agents/search_agent.py:331-387` | RQ1, RQ2, RQ3 | 目前硬編碼權重 `(acc=0.7, lat=0.2, comp=0.1)`，完全忽略 `RewardConfig`。需改為從組態讀取權重，並加入 `energy_score`。 |
| P4 | 修正 `get_balanced_solution()` | ❌ | `src/coordinator/pareto.py:186-245` | RQ1, RQ2, RQ3 | 目前硬編碼權重 `(acc=0.35, lat=0.20, mem=0.15, size=0.15, co2=0.15)`，不受 `RewardConfig` 控制。需接受外部權重參數。 |
| P5 | 實作超體積 (HV) 計算 | ❌ | `src/coordinator/pareto.py` | RQ1, RQ2, RQ3 | Pareto 前沿品質核心指標，目前未實作。建議使用 `pymoo` 套件的 `Hypervolume` 或自行實作。 |
| P6 | 新建 `scripts/run_static_baseline.py` | ❌ | 新檔案 | RQ1, RQ2 | 以固定參數執行單一壓縮方法（不使用 LLM 協調器），直接呼叫 `quantization_wrapper` / `pruning_wrapper`，並進行標準化評估。 |
| P7 | 新建 `scripts/run_rq3_experiment.py` | ❌ | 新檔案 | RQ3 | Sweep runner，遍歷 12 組權重配置 × 模型 × 資料集（見 RQ3 第 4 節）。 |
| P8 | 新建 `config/rq3/` YAML 配置檔 | ❌ | 新目錄 | RQ3 | 12 個 YAML 檔案（W01–W12），見 RQ3 第 4 節。 |
| P9 | 新建 `scripts/analyze_rq3.py` | ❌ | 新檔案 | RQ3 | 分析腳本：計算指標、生成圖表與 LaTeX 表格（見 RQ3 第 10 節）。 |
| P10 | 協調器支援多 LLM 後端 | ❌ | `src/coordinator/langgraph_coordinator.py`、`src/common/config.py` | RQ4 | 新增 `llm_provider` 與 `llm_base_url` 欄位至 `CoordinatorConfig`；建立 LLM 工廠函式，支援 OpenAI API、OpenAI-compatible（vLLM/Ollama）、與 LangChain 本地後端。 |
| P11 | 協調器輸出強健化 | ❌ | `src/coordinator/langgraph_coordinator.py` | RQ4 | 小型模型可能產生無效 JSON。需新增：結構化輸出解析重試（至少 3 次）、fallback 至正則表達式抽取、無效決策計數器。 |
| P12 | 新建 `scripts/run_rq4_experiment.py` | ❌ | 新檔案 | RQ4 | Sweep runner，遍歷協調器 LLM 條件 × 目標模型（見 RQ4 第 5 節）。 |

### 0.2 現有實作狀態確認

| 元件 | 狀態 | 說明 |
|------|------|------|
| GPT-4o 協調器 | ✅ | LangGraph + OpenAI，`langgraph_coordinator.py` |
| Pareto 前沿（5D 支配檢查） | ✅ | 追蹤 accuracy, latency, memory, size, energy/CO₂ |
| GPTQ / AWQ / INT8 / AutoRound 量化 | ✅ | 真實實作，含 fallback |
| ASVD 壓縮 | ✅ | `ASVDCompressor` 於 `quantization_wrapper.py:1599` |
| 幅度剪枝 / 結構化剪枝 | ✅ | `pruning_wrapper.py` + `pruning_agent.py` |
| LoRA / QLoRA 微調 | ✅ | PEFT 函式庫，含 fallback |
| lm-eval 整合 | ✅ | 支援 GSM8K, HellaSwag, CommonsenseQA, TruthfulQA 等 10 個 benchmark |
| HumanEval | ✅ | AST 安全驗證 + 沙箱執行 |
| 延遲量測 | ✅ | `LatencyEvaluator`，CUDA 同步計時 |
| 能耗/CO₂ 量測 | ⚠️ | CodeCarbon → PyNVML → mock 階層式回退；結果須確認 `is_carbon_mock: false` |
| Bandit 搜索 | ⚠️ | 已實作但權重硬編碼（見 P3） |
| HV 計算 | ❌ | 需新增（見 P5） |

### 0.3 共用基線設計

RQ1 的靜態基線（S1–S10）與 RQ2 的靜態基線（B1–B6）存在大量重疊：

| RQ2 條件 | 對應 RQ1 條件 | 壓縮方法 |
|----------|--------------|----------|
| B1 (INT8) | S4 | INT8 量化 |
| B2 (GPTQ-4bit) | S1 | GPTQ 4-bit |
| B3 (AWQ-4bit) | S3 | AWQ 4-bit |
| B4 (剪枝 30%) | S6 | 幅度剪枝 30% |
| B6 (GPTQ+剪枝) | S10 | 管線壓縮 |

**建議**：`run_static_baseline.py` 設計為通用腳本，同時服務 RQ1 與 RQ2。對重疊模型（gpt2、Llama-3-8B-Instruct），直接複用結果，避免重複計算。

### 0.4 建議執行順序

```
Phase 0: 完成 P1–P12 程式碼修改（約 3–4 天）
    │
    ├─→ Phase 1: RQ3 先行（最輕量，48 次執行）
    │       └─ 同時驗證組態系統是否正確傳遞權重
    │
    ├─→ Phase 2: RQ1 靜態基線 (B0 + S1–S10)
    │       └─ 結果可同時用於 RQ2 的 B1–B6
    │
    ├─→ Phase 3: RQ1 動態執行 (D1–D3) + 消融 (A1–A7)
    │       └─ 產出壓縮後的 Llama-3-8B，供 RQ4 使用
    │
    ├─→ Phase 4: RQ2 補充基線（phi-2、Mistral-7B 的 B1–B6）
    │       └─ 與 RQ1 Phase 3 可並行
    │
    ├─→ Phase 5: RQ2 代理執行 (C1–C2) + 事後評估
    │
    └─→ Phase 6: RQ4 協調器 LLM 比較（依賴 Phase 3 產出的壓縮模型）
            └─ 雲端 API 條件可與 Phase 4/5 並行
```

### 0.5 必要依賴套件

```bash
# 核心
pip install pymoo          # HV 計算（P5）
pip install codecarbon     # 真實能耗量測
pip install pynvml         # GPU 功率讀取

# RQ4 本地 LLM 後端
pip install vllm           # 本地 LLM 推論伺服器（P10）
# 或
# pip install ollama       # 替代方案：Ollama 本地推論

# 已包含於 requirements.txt
# transformers, torch, auto-gptq, autoawq, bitsandbytes, peft, lm-eval
```

---

## RQ 1：多代理動態壓縮 vs. 靜態一次性壓縮

### 研究問題

**RQ 1:** How does a multi-agent dynamic compression framework compare to static one-shot compression methods in optimizing the Pareto-optimal trade-offs among post-compression accuracy, inference latency, and inference-stage CO₂ emissions for large language models?

多代理動態壓縮框架相較於靜態一次性壓縮方法，在大型語言模型壓縮後的準確率、推論延遲與推論階段 CO₂ 排放之間的 Pareto 最適權衡上，表現有何差異？

### 假說

由 GPT-4o 驅動的多代理框架，透過 Pareto 前沿分析動態選擇與排序壓縮策略，能產生優於固定超參數靜態一次性壓縮的準確度-延遲-碳排放權衡前沿。動態方法能發現任何單一靜態方法無法觸及的 Pareto 支配解。

---

### 1. 實驗變數

#### 1.1 自變數 — 壓縮方法（14 個條件 + 7 個消融條件）

**FP16 基線**

| 編號 | 條件 | 說明 |
|------|------|------|
| B0 | FP16 基線 | 未壓縮 FP16 參考模型 |

**靜態一次性壓縮 (S1–S10)**

| 編號 | 條件 | 說明 | 實作狀態 |
|------|------|------|----------|
| S1 | GPTQ-4bit | `method=gptq`, `bit_width=4`, `group_size=128`, 512 校準樣本 | ✅ `GPTQQuantizer` |
| S2 | GPTQ-8bit | `method=gptq`, `bit_width=8`, `group_size=128`, 512 校準樣本 | ✅ `GPTQQuantizer` |
| S3 | AWQ-4bit | `method=awq`, `bit_width=4`, `group_size=128`, 512 校準樣本 | ✅ `AWQQuantizer` |
| S4 | INT8 | `method=int8`, bitsandbytes int8 量化 | ✅ `INT8Quantizer` |
| S5 | AutoRound-4bit | `method=autoround`, `bit_width=4`, `group_size=128`, 512 校準樣本 | ✅ `AutoRoundQuantizer` |
| S6 | Pruning-30% | `method=pruning`, 幅度剪枝，30% 稀疏度 | ✅ `MagnitudePruner` |
| S7 | Pruning-50% | `method=pruning`, 幅度剪枝，50% 稀疏度 | ✅ `MagnitudePruner` |
| S8 | ASVD-0.5 | `method=asvd`, `rank_ratio=0.5` | ✅ `ASVDCompressor` |
| S9 | GPTQ-4bit + LoRA | GPTQ-4bit → LoRA `rank=16`, `num_train_steps=100` | ✅ `GPTQQuantizer` + `LoRATrainer` |
| S10 | Pruning-30% + GPTQ-4bit | 30% 幅度剪枝 → GPTQ-4bit | ✅ 組合管線 |

**多代理動態壓縮 (D1–D3)**

| 編號 | 條件 | 說明 |
|------|------|------|
| D1 | 多代理（平衡） | `preset=balanced`, 15 episodes, `search.method=bandit` |
| D2 | 多代理（準確度優先） | `preset=accuracy_focused`, 15 episodes |
| D3 | 多代理（延遲優先） | `preset=latency_focused`, 15 episodes |

**消融條件 (A1–A7) — 僅在 Llama-3-8B-Instruct 上執行**

| 編號 | 消融項目 | 驗證假說 |
|------|----------|----------|
| A1 | 隨機協調器 | 以均勻隨機取樣取代 GPT-4o 決策 → 驗證 LLM 智慧是否重要 |
| A2 | 無 Pareto 回饋 | 移除協調器 prompt 中的 Pareto 前沿摘要 → 驗證 Pareto 引導探索是否重要 |
| A3 | 無 Bandit 搜索 | 移除搜索節點，協調器直接決定所有策略 → 驗證多臂賭博機是否改善探索 |
| A4 | 無技能記憶 | 停用跨 episode 學習（`use_skill_memory=False`） → 驗證歷史學習是否重要 |
| A5 | 單方法動態 | 限制僅使用 GPTQ，允許調整 `bit_width` (2,3,4,8) 與 `group_size` → 驗證方法多樣性是否重要 |
| A6 | 5 episodes | 與 D1 相同設定但僅 5 episodes → 預算敏感度（低） |
| A7 | 30 episodes | 與 D1 相同設定但 30 episodes → 遞減報酬分析（高） |

#### 1.2 控制變數

| 變數 | 設定值 | 對應組態 |
|------|--------|----------|
| LLM 模型 | `gpt-4o` | `coordinator.llm_model` |
| LLM 溫度 | `0.7` | `coordinator.llm_temperature` |
| 校準資料集 | C4 | — |
| 校準樣本數 | 512 | `quantization.calibration_samples` |
| group_size | 128 | `quantization.group_size` |
| 碳排放量測推論次數 | 500 | `evaluation.carbon_inference_count` |
| 硬體 | 同一 GPU（所有條件） | — |
| 隨機種子 | 42, 123, 456 | `--seed`（需實作 P1） |

#### 1.3 應變數

| 變數 | 定義 | 量測方式 |
|------|------|----------|
| 準確度 | 3 個 benchmark 的未加權平均值 | lm-eval harness |
| 推論延遲 | 平均延遲（ms） | `LatencyEvaluator.evaluate_performance()` |
| 每 500 次推論 CO₂ 排放量 | 克 CO₂ | `estimate_energy_consumption()` |

---

### 2. 模型選擇

| 模型 | 參數量 | 選擇理由 |
|------|--------|----------|
| `gpt2` | 124M | 快速迭代驗證方法論，所有壓縮方法皆可運行 |
| `meta-llama/Meta-Llama-3-8B-Instruct` | 8B | 主流研究目標，能測試真實 GPU 記憶體壓力 |

---

### 3. 基準測試選擇

| 基準測試 | Benchmark enum | lm-eval 任務 | 類型 | 對壓縮的敏感度 |
|----------|---------------|-------------|------|---------------|
| GSM8K | `gsm8k` | `gsm8k` | 數學推理 | 高 — 思維鏈對量化敏感 |
| HellaSwag | `hellaswag` | `hellaswag` | 常識自然語言推理 | 中 — 校準良好，穩定 |
| TruthfulQA | `truthfulqa` | `truthfulqa_mc2` | 事實性 | 高 — 測試壓縮是否引入幻覺 |

> **實作備註**：HellaSwag 已在系統中註冊（`schemas.py:HELLASWAG`、`lm_eval_evaluator.py:40`），使用 `acc_norm` 指標。

**聚合準確度** = 三個 benchmark 分數的未加權平均值。

---

### 4. 評估協議

對**每一個**壓縮後的 checkpoint（靜態或動態），量測以下指標：

| 指標 | 方法 | 參數 |
|------|------|------|
| 準確度 | lm-eval harness | 動態迴圈中：proxy 模式（`use_proxy=True`, `proxy_samples=200`）；最終報告：完整評估 |
| 延遲 | `LatencyEvaluator.evaluate_performance()` | `num_iterations=20`, `batch_size=1`, `sequence_length=512`, 含 CUDA 同步 |
| 碳排放 | `estimate_energy_consumption()` | `num_inferences=500`, CodeCarbon → pynvml → mock 階層式量測 |
| 附加指標 | — | `model_size_gb`、`compression_ratio`、`throughput_tokens_per_sec`、`peak_memory_gb` |

> **注意**：Pareto 系統內部追蹤 5 維（accuracy, latency, memory, size, energy/CO₂），但 RQ1 的分析聚焦於 3 維（accuracy, latency, CO₂）以匹配研究問題範圍。

---

### 5. 實驗流程

每個條件 × 3 個種子 = 3 次執行。

#### 第一階段 — FP16 基線 (B0)

透過 `scripts/run_manual_eval.py` 評估未壓縮模型。

```bash
python scripts/run_manual_eval.py --model gpt2 --device cuda --proxy
python scripts/run_manual_eval.py --model meta-llama/Meta-Llama-3-8B-Instruct --device cuda --proxy
```

#### 第二階段 — 靜態基線 (S1–S10)

**需先完成 P6**：新腳本 `scripts/run_static_baseline.py`：
- 以固定參數套用單一壓縮方法（不使用 LLM 協調器）
- 呼叫 `src/tools/quantization_wrapper.py`（S1–S5, S8）、`src/tools/pruning_wrapper.py`（S6–S7）
- 在 3 個 benchmark + 延遲 + 碳排放上進行評估
- 管線基線（S9, S10）：依序串接壓縮方法

```bash
# 範例用法
python scripts/run_static_baseline.py --model gpt2 --condition S1 --seed 42
python scripts/run_static_baseline.py --model gpt2 --condition all --seed 42,123,456
```

#### 第三階段 — 多代理執行 (D1–D3)

透過 `scripts/run_pipeline.py` 執行，每個 preset 15 episodes。**需先完成 P1（`--seed` 參數）**。

```bash
# D1: 平衡
python scripts/run_pipeline.py -m MODEL -d gsm8k -e 15 --preset balanced --seed SEED

# D2: 準確度優先
python scripts/run_pipeline.py -m MODEL -d gsm8k -e 15 --preset accuracy_focused --seed SEED

# D3: 延遲優先
python scripts/run_pipeline.py -m MODEL -d gsm8k -e 15 --preset latency_focused --seed SEED
```

#### 第四階段 — 消融實驗 (A1–A7)

僅在 Llama-3-8B-Instruct 上執行。各消融條件需要的程式碼修改：

| 消融 | 修改位置 | 實作方式 |
|------|----------|----------|
| A1 | `langgraph_coordinator.py` | 替換協調器決策邏輯為均勻隨機取樣 |
| A2 | `langgraph_coordinator.py` | 從協調器 prompt 中移除 `pareto_summary` 欄位 |
| A3 | `langgraph_coordinator.py` | 在 LangGraph 狀態機中移除搜索節點 |
| A4 | `langgraph_coordinator.py` | 設定 `use_skill_memory=False` |
| A5 | `langgraph_coordinator.py` | 限制 `CompressionMethod` 僅為 `gptq`，允許調整 `bit_width` 與 `group_size` |
| A6 | CLI | `--episodes 5`（其餘同 D1） |
| A7 | CLI | `--episodes 30`（其餘同 D1） |

> **實作建議**：A1–A5 可透過在 `LangGraphCoordinator` 中新增 `ablation_mode` 參數實現，避免修改生產程式碼。A6、A7 僅需調整 CLI 參數。

#### 第五階段 — 事後標準化評估

以相同設定重新評估所有 checkpoint：
- 完整 benchmark（非 proxy）
- 20 次延遲量測
- 500 次碳排放推論

#### 第六階段 — 分析

建構 Pareto 前沿、計算超體積、統計檢定、視覺化。**需先完成 P5（HV 計算）**。

---

### 6. Pareto 前沿品質指標

| 指標 | 說明 | 越好的方向 |
|------|------|-----------|
| 超體積指標 (HV) | 前沿在 3D 目標空間中支配的體積 | 越高越好 |
| 反向世代距離 (IGD) | 前沿與理想參考集的距離 | 越低越好 |
| 前沿大小 | 非支配解的數量 | 越多越好 |
| 前沿展幅 | 前沿上各目標的範圍 | 越寬越好 |
| 覆蓋率 | 各方法對合併前沿的貢獻比例 | 越高越好 |
| 支配比率 | 兩兩比較：A 的多少點支配 B 的點 | 越高越好 |

**HV 參考點**：`(accuracy=0, latency=max_observed×1.1, CO₂=max_observed×1.1)`

> 注意：準確度為越高越好，HV 計算時需取反（1 − accuracy）或以 max − accuracy 轉換，使三個目標方向一致（均為越低越好）。

---

### 7. 統計分析計畫

1. **條件內變異**：3 個種子 → 報告平均值 ± 95% 信賴區間
2. **超體積比較**：Kruskal-Wallis 檢定比較各方法的 HV；Cliff's delta 衡量效果量
3. **兩兩支配分析**：計算靜態與動態前沿之間的相互支配數量
4. **單目標分析**：Wilcoxon 符號秩檢定比較各方法跨種子的最佳可達值
5. **收斂分析**：HV 隨 episode 的進展曲線；達到最終 HV 90% 所需的 episode 數
6. **消融分析**：Wilcoxon 符號秩檢定比較各消融條件的 HV 與 D1（平衡）

**顯著水準**：α = 0.05，多重比較使用 Bonferroni 校正。

---

### 8. 消融實驗設計

每個消融透過移除/替換一個組件來隔離其貢獻：

| 消融 | 移除/替換的組件 | 預期效果 |
|------|----------------|----------|
| A1（隨機） | GPT-4o 決策 → 均勻隨機 | HV 顯著下降 → LLM 智慧有價值 |
| A2（無 Pareto） | Pareto 前沿摘要 | HV 下降 → Pareto 引導探索有效 |
| A3（無搜索） | 搜索節點（bandit） | HV 下降 → 多臂賭博機改善探索 |
| A4（無記憶） | 跨 episode 技能記憶 | HV 下降 → 歷史學習有價值 |
| A5（單方法） | 方法多樣性 → 僅 GPTQ | HV 下降 → 方法多樣性重要 |
| A6（5 ep） | 預算 15→5 | 量化短預算下的效能損失 |
| A7（30 ep） | 預算 15→30 | 量化額外預算的邊際收益 |

**預算公平性說明**：A6 使用 5 episodes 探索 5 個配置，而靜態基線 S1–S10 提供 10 個配置。此比較旨在測試動態方法在低預算下是否仍能找到優質解，而非等量比較。等量比較為 D1（15 episodes）vs. S1–S10 聯集前沿（10 個配置）。

---

### 9. 預期圖表

**表格**
1. 所有條件 × {準確度, 延遲, CO₂, HV}，含平均值 ± 95% CI
2. 消融結果含統計顯著性（p 值、Cliff's delta 效果量）
3. 各壓縮方法的逐 benchmark 準確度保留率

**圖**
1. 3D Pareto 散佈圖（準確度 vs 延遲 vs CO₂），靜態=藍色，動態=紅色
2. 2D 投影：準確度-延遲、準確度-CO₂、延遲-CO₂
3. D1–D3 的 HV 隨 episode 進展曲線（含信賴帶）
4. 消融長條圖 — 各消融條件的 HV vs D1
5. 損益兩平分析 — 累積 CO₂ 節省量 vs 推論次數

---

### 10. 計算資源估計

| 項目 | GPT-2（×3 種子） | Llama-3-8B（×3 種子） |
|------|-----------------|---------------------|
| B0 + S1–S10 基線 | ~3 小時 | ~24 小時 |
| D1–D3 動態執行 | ~9 小時 | ~54 小時 |
| A1–A7 消融實驗 | — | ~84 小時 |
| 事後評估 | ~3 小時 | ~12 小時 |
| **合計** | **~15 小時** | **~174 小時（約 7.3 天）** |

**OpenAI API 費用**：約 $31 USD（39 次動態執行 × 16 次 GPT-4o 呼叫 × ~$0.05/次）

---

### 11. 參考的程式碼檔案

| 檔案 | 在 RQ1 中的角色 | 需修改？ |
|------|-----------------|----------|
| `scripts/run_pipeline.py` | 執行 D1–D3 動態條件 | 是（P1: `--seed`） |
| `scripts/run_manual_eval.py` | 執行 B0 基線 | 否 |
| `scripts/run_static_baseline.py` | 執行 S1–S10 靜態基線 | 新建（P6） |
| `src/tools/quantization_wrapper.py` | S1–S5, S8 靜態壓縮 | 否 |
| `src/tools/pruning_wrapper.py` | S6–S7 靜態剪枝 | 否 |
| `src/coordinator/langgraph_coordinator.py` | 核心協調器，消融目標 | A1–A5 修改 |
| `src/coordinator/pareto.py` | Pareto 前沿邏輯 | 是（P4, P5: HV + 權重） |
| `src/agents/evaluation_agent.py` | `evaluate_model`、`estimate_energy_consumption` | 否 |
| `src/agents/search_agent.py` | Bandit 搜索 | 是（P3: 能耗權重） |
| `src/evaluation/evaluators/latency_evaluator.py` | `evaluate_performance()` 延遲量測 | 否 |
| `src/common/config.py` | 組態預設值與欄位定義 | 否（RQ1 使用現有 preset） |
| `config/default.yaml` | 預設組態範本 | 否 |

---

## RQ 2：自適應代理驅動壓縮對能耗/CO₂ 減少之效果

### 研究問題

**RQ 2:** To what extent can adaptive, agent-driven pruning and quantization reduce inference-stage energy consumption and CO₂ emissions while maintaining acceptable task-level accuracy across diverse benchmarks, compared to fixed human-specified compression strategies?

自適應的代理驅動剪枝與量化，相較於固定的人為指定壓縮策略，能在多大程度上降低推論階段的能源消耗與 CO₂ 排放，同時維持跨多元基準測試的可接受任務層級準確率？

### 假說

由 GPT-4o 驅動的代理框架，透過 Pareto 前沿分析動態選擇與排序壓縮策略，能產生在最小化能耗/CO₂ 的同時維持準確度的壓縮策略，並優於固定（人工指定）壓縮策略。系統已透過 CodeCarbon/PyNVML 進行 5 維 Pareto 最佳化追蹤能耗（準確度、延遲、記憶體、模型大小、CO₂）。

---

### 1. 實驗條件（自變數）

#### 條件 A：基線（未壓縮 FP16）
原始模型，未施加任何壓縮。建立所有指標的基準真值。

#### 條件 B：靜態壓縮（6 種固定策略，無代理）
以預定壓縮方式直接套用，不經過 LLM 協調器或迭代最佳化。

| 編號 | 方法 | 參數 | 原因 | 實作狀態 |
|------|------|------|------|----------|
| B1 | INT8 量化 | 8-bit, bitsandbytes | 保守量化，準確度損失最小 | ✅ `INT8Quantizer` |
| B2 | GPTQ 4-bit | 4-bit, group_size=128 | 標準激進量化 | ✅ `GPTQQuantizer` |
| B3 | AWQ 4-bit | 4-bit, group_size=128 | 激活感知量化 | ✅ `AWQQuantizer` |
| B4 | 幅度剪枝 30% | sparsity=0.3, 非結構化 | 中度剪枝 | ✅ `MagnitudePruner` |
| B5 | 結構化剪枝 30% | sparsity=0.3, 通道層級 | 結構化壓縮 | ✅ `prune_model_structured()` |
| B6 | GPTQ 4-bit + 剪枝 30% | 管線：先剪枝 → 再量化 | 組合策略 | ✅ 組合管線 |

#### 條件 C：自適應代理驅動壓縮（完整系統）
GPT-4o 協調器自主選擇並迭代壓縮策略。

| 編號 | 預設 | energy_weight | 說明 | 需修改？ |
|------|------|--------------|------|----------|
| C1 | `energy_focused`（新增） | 1.5 | 大幅優先降低能耗 | 是（P2） |
| C2 | `balanced`（現有） | 0.1 | 預設多目標平衡 | 否 |

---

### 2. 模型選擇

| 模型 | 家族 | 參數量 | FP16 大小 | 用途 |
|------|------|--------|-----------|------|
| `gpt2` | GPT-2 | 124M | 0.5 GB | 小型參考，快速迭代 |
| `microsoft/phi-2` | Phi | 2.7B | 5.4 GB | 中小型，高效架構 |
| `mistralai/Mistral-7B-v0.1` | Mistral | 7B | 13.5 GB | 中型，廣泛部署 |
| `meta-llama/Meta-Llama-3-8B-Instruct` | Llama 3 | 8B | 16.0 GB | 中大型，指令微調 |

> **實作備註**：4 個模型均已在 `MODEL_SIZE_DATABASE`（`src/common/model_utils.py`）中註冊，規格推斷可正確運作。phi-2 與 Mistral-7B 為 RQ2 新增模型，不在 RQ1 範圍內。

---

### 3. 基準測試選擇

| 基準測試 | 領域 | 指標 | 程式碼 ID |
|----------|------|------|-----------|
| GSM8K | 數學推理 | 精確匹配 | `gsm8k` |
| CommonsenseQA | 常識推理 | 準確度 | `commonsenseqa` |
| TruthfulQA | 真實性 | MC2 準確度 | `truthfulqa` |
| HumanEval | 程式碼生成 | pass@1 | `humaneval` |
| BIG-Bench Hard | 多任務推理 | 準確度 | `bigbench_hard` |

> **RQ1 與 RQ2 差異說明**：RQ1 使用 3 個 benchmark（GSM8K、HellaSwag、TruthfulQA）聚焦壓縮對推理能力的影響；RQ2 使用 5 個 benchmark 以測試能耗降低後的跨任務泛化能力，且包含 HumanEval（程式碼生成）作為高複雜度任務的代表。

---

### 4. 評估指標

#### 主要指標（能耗/CO₂）

| 指標 | 來源 | 單位 |
|------|------|------|
| 每次推論能耗 | `estimate_energy_consumption()`（`evaluation_agent.py:835`） | 焦耳 (J) |
| 總 CO₂ 排放 | 換算：energy_kwh × 500 | 克 (g) |
| 能耗降低比率 | `(E_baseline − E_compressed) / E_baseline` | % |

#### 次要指標（準確度）

| 指標 | 定義 |
|------|------|
| 各基準測試準確度 | 5 個基準測試的絕對分數 |
| 準確度保留率 | `accuracy_compressed / accuracy_baseline`（逐基準測試） |
| 平均保留率 | 5 個基準測試的平均 |
| 最差情況保留率 | 5 個基準測試的最小值 |

#### 第三指標（效率）

| 指標 | 來源 |
|------|------|
| 延遲 (ms) | `LatencyEvaluator` |
| 峰值記憶體 (GB) | 推論時 VRAM |
| 模型大小 (GB) | 壓縮後檔案大小 |
| 壓縮比 | `原始大小 / 壓縮後大小` |

#### Pareto 品質指標

| 指標 | 定義 |
|------|------|
| 超體積指標 (Hypervolume) | Pareto 前沿所支配的目標空間體積（需完成 P5） |
| Pareto 解數量 | 來自 `ParetoFrontier.get_summary()` |
| Pareto 分散度 | 各目標維度的範圍 |

---

### 5.「不可接受降級」之定義

一個解被視為**可接受**，當且僅當：

```
∀ 基準測試 b: retention(b) = score_compressed(b) / score_baseline(b) ≥ 0.90
且
mean(retention(b), ∀ b) ≥ 0.95
```

- **逐基準測試底線（90%）**：確保無單一任務發生災難性失敗
- **平均門檻（95%）**：維持整體品質
- 與 `RewardConfig` 中 `min_accuracy: 0.9` 預設值一致（`src/common/config.py:180`）

---

### 6. 實驗流程

#### 階段 1：基線評估（條件 A）

對 4 個模型各自執行：

```bash
python scripts/run_manual_eval.py \
    --model MODEL_NAME \
    --device cuda \
    --benchmarks gsm8k commonsenseqa truthfulqa humaneval bigbench_hard \
    --batch-size 8 \
    --energy-inferences 1000 \
    --output-dir data/experiments/rq2/baseline/MODEL_SHORT
```

- 每個模型的能耗量測重複 **3 次**，確保統計可靠性
- 記錄：GPU 型號、驅動版本、CUDA 版本、環境溫度

#### 階段 2：靜態壓縮（條件 B）

對每個模型 × 每種靜態策略（B1–B6），使用 `run_static_baseline.py`（P6）：
1. 套用壓縮（單次，無迭代）
2. 執行完整基準測試評估（相同 5 個基準測試）
3. 量測能耗（1000 次推論，重複 3 次）

**不使用協調器、不使用搜尋、不進行迭代** — 此為「人工專家」基線。

> **共用基線**：gpt2 與 Llama-3-8B-Instruct 的 B1–B4, B6 結果可直接複用 RQ1 的 S4, S1, S3, S6, S10 結果（見 0.3 節），僅需補充 5 個 benchmark 的完整評估與 1000 次能耗量測。

#### 階段 3：代理驅動壓縮（條件 C）

對每個模型 × 每種預設（C1, C2）× 3 個種子。**需先完成 P1（`--seed`）與 P2（`energy_focused` 預設）**。

```bash
python scripts/run_pipeline.py \
    -m MODEL_NAME \
    -d gsm8k \
    -e 15 \
    --preset energy_focused \  # C2 使用 balanced
    --seed SEED \
    -n "rq2_C1_MODEL_seed42" \
    -o data/experiments/rq2/agent/
```

- 每次執行 **15 輪迭代**（收斂耐心 = 5，可能在第 10–12 輪提前終止）
- 最佳化期間使用**代理評估**（200 樣本）以提升速度
- **3 個種子**（42, 123, 456）以處理 LLM 協調器的隨機性
- 收斂後：對最佳 Pareto 解以**完整評估**重新評估所有 5 個基準測試

#### 階段 4：跨基準測試泛化驗證

代理以 GSM8K 作為主要最佳化資料集。找到最佳解後：
- 對所有 Pareto 最佳解在全部 5 個基準測試上進行完整評估
- 測試節能解是否能泛化至不同任務
- 另外針對 Phi-2 和 Mistral-7B 以 TruthfulQA 為最佳化目標執行（2 組額外實驗），驗證對最佳化目標選擇的穩健性

---

### 7. 新增 `energy_focused` 預設

**需完成 P2**。新增至 `AdvancedConfig.from_preset()` (`src/common/config.py:373`)：

```python
"energy_focused": {
    "quantization": {
        "default_bit_width": 4,
        "default_method": "gptq",
    },
    "evaluation": {
        "measure_carbon": True,
        "carbon_inference_count": 1000,
    },
    "reward": {
        "accuracy_weight": 1.0,
        "latency_weight": 0.2,
        "memory_weight": 0.2,
        "energy_weight": 1.5,     # 大幅提升能耗權重
        "min_accuracy": 0.90,     # 準確度硬性底線
    },
    "search": {
        "method": "bandit",
        "exploration_ratio": 0.3,
    },
    "termination": {
        "max_episodes": 15,
        "convergence_patience": 5,
    },
}
```

#### 7.1 已知的權重傳播缺口

以下位置使用硬編碼權重，不受 `RewardConfig` 控制。需同步修正以使 `energy_focused` 預設真正生效：

| 位置 | 目前行為 | 修正方式 |
|------|----------|----------|
| `search_agent.py:331-387` `compute_bandit_reward()` | 硬編碼 `acc=0.7, lat=0.2, comp=0.1`，忽略能耗 | **P3**：從 config 讀取權重，加入 `energy_score` |
| `pareto.py:186-245` `get_balanced_solution()` | 硬編碼 `acc=0.35, lat=0.20, mem=0.15, size=0.15, co2=0.15` | **P4**：接受 RewardConfig 權重參數 |
| `langgraph_coordinator.py` 協調器 prompt | 提及「balancing accuracy, latency, memory, and model size」，未提及能耗 | 需在 system prompt 中加入能耗/CO₂ 最佳化目標描述 |

**修正 `compute_bandit_reward()` 範例**（P3）：

```python
# 在 compute_bandit_reward 中新增：
energy_j = eval_result.get("energy_joules", None)
if energy_j is not None:
    energy_score = 1.0 / (1.0 + energy_j)
    weights.setdefault("energy", config.reward.energy_weight)
    reward += weights["energy"] * energy_score
```

---

### 8. 混淆變數控制

| 混淆變數 | 控制方式 |
|----------|----------|
| 硬體差異 | 所有實驗使用相同 GPU；記錄硬體規格 |
| GPU 熱狀態 | 各次執行間以 `torch.cuda.empty_cache()` 重置 + 冷卻間隔 |
| 評估協定 | 統一 `carbon_inference_count=1000`、`batch_size=8`、相同提示語進行能耗量測 |
| LLM 隨機性 | 每種代理條件使用 3 個種子；靜態條件為確定性 |
| 比較公平性 | 代理獲得 15 輪迭代；靜態有 6 種策略。報告逐策略及組合比較 |
| 代理評估 vs 完整評估 | 最佳化期間使用代理評估；最終報告數字使用完整評估 |
| Mock 能耗回退 | 實驗前驗證 `is_carbon_mock: false`，排除使用 mock 的執行 |

---

### 9. 統計分析方案

#### 主要分析：在準確度約束下的能耗降低
- 對每個模型，提取滿足接受門檻（第 5 節）之解的能耗 (J)
- **Wilcoxon 符號秩次檢定**（配對）：B_best vs C1_best、B_best vs C2_best、C1 vs C2
- 報告 **Cohen's d** 效應量及來自 3 次重複的 **95% 信賴區間**
- 效應門檻：≥20% 降低 = 有意義、≥50% = 顯著

#### 次要分析：Pareto 前沿品質
- 在（準確度、能耗、延遲）上計算**超體積指標**（需完成 P5）
- 以 **Mann-Whitney U 檢定** 跨 3 次試驗比較
- 比較可接受的 Pareto 解數量

#### 第三分析：收斂行為
- 各模型/預設的收斂所需輪數
- 當 energy_weight 較高（C1）vs 較低（C2）時策略分佈如何轉移

---

### 10. 論文表格與圖表

#### 表 1：基線模型特性

| 模型 | 參數量 | FP16 大小 | 能耗/推論 (J) | CO₂/千次推論 (g) | GSM8K | CSQA | TQA | HumanEval | BBH |
|------|--------|-----------|---------------|------------------|-------|------|-----|-----------|-----|

#### 表 2：主要結果 — 能耗降低與準確度保留

| 模型 | 條件 | 策略 | 能耗 (J) | 降低 (%) | CO₂ (g/千次) | 平均保留率 | 最差保留率 | 可接受？ |
|------|------|------|----------|----------|--------------|------------|------------|----------|

（列：對每個模型列出 A、B1–B6、B_best、C1、C2）

#### 表 3：逐基準測試準確度保留率

| 模型 | 條件 | GSM8K | CSQA | TQA | HumanEval | BBH | 平均 |
|------|------|-------|------|-----|-----------|-----|------|

#### 表 4：Pareto 前沿品質

| 模型 | 條件 | 解數量 | 超體積 | 能耗範圍 | 準確度範圍 |
|------|------|--------|--------|----------|------------|

#### 圖 1：能耗 vs 準確度 Pareto 前沿（2x2 格，每模型一子圖）
- 靜態策略為散點、代理 C1 前沿為線+點、代理 C2 前沿為線+點
- 90% 保留率以下的區域標為「不可接受」

#### 圖 2：CO₂ 排放長條圖
- 分組長條：每模型的基線、B_best、C1、C2
- 來自 3 次能耗量測重複的誤差條

#### 圖 3：代理收斂曲線
- X 軸：輪次、Y 軸：截至目前最佳能耗
- 來自 3 個種子的誤差帶；C1 vs C2 分開面板

#### 圖 4：代理策略選擇分佈
- 堆疊面積圖：各輪次選擇的壓縮方法
- C1 vs C2 比較，展示能耗權重如何影響策略選擇

#### 圖 5：雷達圖（多目標比較）
- 5 個軸：準確度、延遲、記憶體、大小、CO₂
- 疊加：基線、B_best、C1_best、C2_best（旗艦模型）

---

### 11. 實驗總矩陣

| 維度 | 數值 | 數量 |
|------|------|------|
| 模型 | gpt2, phi-2, Mistral-7B, Llama-3-8B | 4 |
| 靜態策略 | B1–B6 | 6 |
| 代理預設 | C1, C2 | 2 |
| 種子（僅代理） | 42, 123, 456 | 3 |
| 能耗量測重複 | 每次量測 3 次 | 3 |

**總執行次數：**
- 基線：4 模型 × 5 基準測試 × 3 能耗重複 = 12 次評估+能耗量測
- 靜態：4 模型 × 6 策略 × 3 能耗重複 = 72 次壓縮+評估+能耗量測
- 代理：4 模型 × 2 預設 × 3 種子 × 15 輪 = 360 輪迭代（+ Pareto 解重新評估）
- 跨基準測試：對最佳解在其餘 4 個基準測試上重新評估 + 2 組額外最佳化目標

---

### 12. 預期結果

1. **C1（能耗導向）達成 30–50% 能耗降低**，同時維持所有基準測試 ≥90% 保留率
2. **C1 > B_best 在能耗降低方面** — 代理找到更佳組合（如剪枝 + 量化 + LoRA 恢復）
3. **C1 > C2 在能耗降低方面**，但 C2 可能具有略高的準確度（不同 Pareto 重點）
4. **較小模型（gpt2）的相對效益較小** — 絕對能耗已經很低
5. **組合策略（管線）出現在代理的 Pareto 前沿中** — 展示自適應多步驟最佳化的價值

---

### 13. 計算資源估計

| 項目 | GPT-2 | Phi-2 | Mistral-7B | Llama-3-8B |
|------|-------|-------|------------|------------|
| 基線 A（×3 重複） | ~0.5 hr | ~2 hr | ~4 hr | ~5 hr |
| 靜態 B1–B6（×3 重複） | ~3 hr | ~12 hr | ~24 hr | ~30 hr |
| 代理 C1+C2（×3 種子） | ~3 hr | ~12 hr | ~36 hr | ~42 hr |
| 跨基準測試驗證 | ~1 hr | ~2 hr | ~4 hr | ~5 hr |
| **小計** | **~7.5 hr** | **~28 hr** | **~68 hr** | **~82 hr** |

**總計**：約 185.5 小時（約 7.7 天）

**OpenAI API 費用**：約 $19 USD（4 模型 × 2 預設 × 3 種子 × 16 次 GPT-4o 呼叫 × ~$0.05/次）

---

### 14. 所需程式碼變更

| 檔案 | 變更 | 對應前置作業 |
|------|------|-------------|
| `src/common/config.py` | 新增 `energy_focused` 預設 | P2 |
| `src/agents/search_agent.py` | 在 `compute_bandit_reward()` 中加入能耗 | P3 |
| `src/coordinator/pareto.py` | `get_balanced_solution()` 接受外部權重 | P4 |
| `src/coordinator/langgraph_coordinator.py` | 協調器 prompt 加入能耗/CO₂ 目標 | — |
| `scripts/run_pipeline.py` | 在 CLI `--preset` 選項中新增 `energy_focused` | P2 附帶 |

---

### 15. 參考的程式碼檔案

| 檔案 | 在 RQ2 中的角色 |
|------|-----------------|
| `scripts/run_pipeline.py` | 執行 C1–C2 代理條件 |
| `scripts/run_manual_eval.py` | 執行 A 基線評估 |
| `scripts/run_static_baseline.py` | 執行 B 靜態條件（P6 新建） |
| `src/tools/quantization_wrapper.py` | B1–B3 靜態量化 |
| `src/tools/pruning_wrapper.py` | B4–B5 靜態剪枝 |
| `src/agents/pruning_agent.py` | `prune_model()`、`prune_model_structured()` 工具 |
| `src/coordinator/langgraph_coordinator.py` | 核心協調器 |
| `src/coordinator/pareto.py` | Pareto 前沿與超體積計算 |
| `src/agents/evaluation_agent.py` | `evaluate_model`、`estimate_energy_consumption` |
| `src/agents/search_agent.py` | Bandit 搜索與 `compute_bandit_reward` |
| `src/common/config.py` | `AdvancedConfig`、`RewardConfig`、預設定義 |
| `config/default.yaml` | 預設組態範本 |

---

## RQ 3：權重方案對 Pareto 前沿之影響

### 研究問題

**RQ 3:** How do variations in weighting schemes for multi-objective optimization (accuracy, latency, memory, energy, CO₂ emissions) influence the shape of the Pareto frontier and the selection of optimal compression strategies?

多目標優化中（準確率、延遲、記憶體、能耗、CO₂ 排放）不同的權重配置，如何影響 Pareto 前緣的形狀與最適壓縮策略的選擇？

### 假說

偏向單一目標的極端權重設定會產出在該目標上強但多樣性低的前沿；均衡權重則會產出較廣泛的前沿。同時，與既有 preset 對應的權重組態應可復現 preset 行為，確認組態系統正確傳遞權重至各子系統。

### 已知限制

`RewardConfig` 中的權重目前未完全傳播至所有子系統。具體而言：

| 子系統 | 現狀 | 影響 |
|--------|------|------|
| `compute_bandit_reward()` | 硬編碼 `acc=0.7, lat=0.2, comp=0.1` | 搜索節點不受權重配置影響 |
| `get_balanced_solution()` | 硬編碼 `acc=0.35, lat=0.20, ...` | 平衡解選擇不受權重配置影響 |
| 協調器 prompt | 不含具體權重數值 | GPT-4o 無法感知權重偏好 |

**本 RQ 可作為偵測此問題實際影響程度的手段**。若 W01–W12 的結果無顯著差異，即證實權重傳播缺口的嚴重性。**建議先完成 P3、P4 修正後再執行 RQ3**。

---

### 1. 權重配置表（W01–W12）

| ID | 名稱 | acc_w | lat_w | mem_w | energy_w | min_acc | quant_method | bit_width | 設計理由 |
|----|------|-------|-------|-------|----------|---------|--------------|-----------|----------|
| W01 | `accuracy_extreme` | 2.0 | 0.1 | 0.1 | 0.05 | 0.95 | int8 | 8 | 單目標：準確度 |
| W02 | `latency_extreme` | 0.5 | 1.5 | 0.5 | 0.1 | 0.85 | gptq | 4 | 單目標：延遲 |
| W03 | `memory_extreme` | 0.5 | 0.3 | 1.5 | 0.1 | 0.85 | awq | 4 | 單目標：記憶體 |
| W04 | `energy_extreme` | 0.3 | 0.3 | 0.3 | 2.0 | 0.80 | gptq | 2 | 單目標：能耗/CO₂ |
| W05 | `size_extreme` | 0.3 | 0.3 | 0.3 | 0.5 | 0.80 | gptq | 2 | 單目標：模型大小 |
| W06 | `equal_weights` | 1.0 | 1.0 | 1.0 | 1.0 | 0.85 | gptq | 4 | 均衡基線 |
| W07 | `green_focused` | 0.5 | 0.3 | 0.3 | 1.5 | 0.80 | gptq | 3 | 永續優先 |
| W08 | `perf_focused` | 1.5 | 1.5 | 0.3 | 0.1 | 0.90 | gptq | 4 | 準確度 + 速度 |
| W09 | `preset_accuracy` | 2.0 | 0.1 | 0.1 | 0.1 | 0.95 | int8 | 8 | 對應既有 accuracy_focused preset |
| W10 | `preset_latency` | 0.5 | 1.5 | 0.5 | 0.1 | 0.85 | gptq | 4 | 對應既有 latency_focused preset |
| W11 | `preset_balanced` | 1.0 | 0.3 | 0.3 | 0.1 | 0.90 | gptq | 4 | 對應既有 balanced preset |
| W12 | `preset_memory` | 0.5 | 0.3 | 1.5 | 0.1 | 0.85 | awq | 4 | 對應既有 memory_constrained preset |

**配置分組說明：**

- **W01–W05**（單目標極端）：各自將一個目標權重拉高至 1.5–2.0，其餘壓低，測試單目標偏好對前沿的塑形效果。
- **W06**（均衡基線）：所有權重相等，作為參照。
- **W07–W08**（混合偏好）：同時強調兩個以上目標的組合。
- **W09–W12**（Preset 對照）：權重設定對應系統中既有 preset（`src/common/config.py:373-434`），驗證組態一致性。

> **比對現有 preset**：W09 對應 `accuracy_focused`（acc=2.0, lat=0.1, mem=0.1, min_acc=0.95）；W10 對應 `latency_focused`（acc=0.5, lat=1.5, mem=0.5, min_acc=0.85）；W11 對應 `balanced`（acc=1.0, lat=0.3, mem=0.3, min_acc=0.9）；W12 對應 `memory_constrained`（acc=0.5, lat=0.3, mem=1.5, min_acc=0.85）。注意現有 preset 均未設定 `energy_weight`（使用預設值 0.1）。

---

### 2. 模型-資料集矩陣

| 模型 | 資料集 | Episodes |
|------|--------|----------|
| `gpt2`（124M） | gsm8k, commonsenseqa | 12 |
| `meta-llama/Meta-Llama-3-8B-Instruct`（8B） | gsm8k, commonsenseqa | 12 |

**總執行次數：12 權重 × 2 模型 × 2 資料集 = 48 次執行，每次 1 個種子**

> 本實驗採用單種子（seed=42）而非 RQ1 的 3 種子設計，原因是 RQ3 的核心目的是比較 12 組權重配置的趨勢差異，而非量化條件內變異。48 次執行的規模已足以觀察權重對前沿形狀的系統性影響。**限制**：單種子無法提供信賴區間，結果解讀需謹慎。若發現權重間差異微小，應考慮以 3 種子重跑部分配置。

---

### 3. 固定參數（所有執行共用）

| 參數 | 設定值 | 對應組態欄位 |
|------|--------|-------------|
| LLM 溫度 | 0.3 | `coordinator.llm_temperature` |
| Proxy 評估 | `true` | `evaluation.use_proxy` |
| Proxy 樣本數 | 200 | `evaluation.proxy_samples` |
| 碳排放量測 | `true` | `evaluation.measure_carbon` |
| 碳排放推論次數 | 500 | `evaluation.carbon_inference_count` |
| 搜索方法 | bandit | `search.method` |
| 收斂耐心 | 5 | `termination.convergence_patience` |
| 時間預算 | 4.0 小時 | `termination.budget_hours` |
| 隨機種子 | 42 | `--seed`（需完成 P1） |

---

### 4. 實驗執行方法

#### 4.1 組態檔案生成

**需完成 P8**。為每個權重配置建立 YAML 檔案，放置於 `config/rq3/` 目錄：

```
config/rq3/
├── W01_accuracy_extreme.yaml
├── W02_latency_extreme.yaml
├── ...
└── W12_preset_memory.yaml
```

每個 YAML 檔案約 30 行，內容範例（W01）：

```yaml
# W01: accuracy_extreme — 單目標：準確度
coordinator:
  llm_model: gpt-4o
  llm_temperature: 0.3

quantization:
  default_bit_width: 8
  default_method: int8

evaluation:
  use_proxy: true
  proxy_samples: 200
  measure_carbon: true
  carbon_inference_count: 500

search:
  method: bandit

reward:
  accuracy_weight: 2.0
  latency_weight: 0.1
  memory_weight: 0.1
  energy_weight: 0.05
  min_accuracy: 0.95

termination:
  max_episodes: 12
  budget_hours: 4.0
  convergence_patience: 5
```

#### 4.2 Sweep Runner（`scripts/run_rq3_experiment.py`）

**需完成 P7**。新腳本，使用 Click CLI，約 350 行。核心功能：

```bash
# 執行全部 48 次
python scripts/run_rq3_experiment.py sweep --all

# 僅執行特定權重
python scripts/run_rq3_experiment.py sweep --weights W01,W06,W11

# 僅執行特定模型
python scripts/run_rq3_experiment.py sweep --models gpt2

# 斷點續跑（自動跳過已完成的執行）
python scripts/run_rq3_experiment.py sweep --all --resume

# 顯示執行計畫（dry run）
python scripts/run_rq3_experiment.py sweep --all --dry-run
```

**Sweep Runner 職責：**

1. 讀取 `config/rq3/W*.yaml` 配置檔案
2. 建立 manifest（`data/experiments/rq3/manifest.json`）
3. 依序對每個 (weight, model, dataset) 組合呼叫 `run_pipeline.py`
4. 追蹤完成狀態，支援中斷後續跑
5. 收集所有執行結果，生成彙總報告

---

### 5. 分析指標（7 項）

| 指標 | 說明 | 越好的方向 |
|------|------|-----------|
| 超體積 (HV) | 前沿在目標空間中支配的體積（需完成 P5） | 越高越好 |
| \|PF\| | Pareto 最佳解的數量 | 越多越好 |
| 展幅 (Spread) | 前沿上各目標的 max–min 範圍 | 越寬越好 |
| 間距 (Spacing) | 連續解之間距離的標準差（解的均勻度） | 越低越好 |
| 策略熵 (Strategy Entropy) | 壓縮方法選擇分佈的 Shannon 熵 | 越高 = 越多樣 |
| 綠色效率 (Green Efficiency) | `best_accuracy / best_co2` 比值 | 越高越好 |
| 收斂 Episode | HV 達到最終值 95% 所需的 episode 數 | 越少越好 |

**HV 參考點**：`(accuracy=0, latency=max_observed×1.1, memory=max_observed×1.1, CO₂=max_observed×1.1)`

> 準確度為越高越好，HV 計算時需取反（1 − accuracy），使所有目標方向一致（均為越低越好）。

---

### 6. 視覺化計畫（6 張圖，存為 PDF + PNG）

| 編號 | 名稱 | 說明 |
|------|------|------|
| Fig.1 | **Pareto 前沿疊加圖** | 2D 散佈圖，每個 model-dataset 組合一個面板，疊加 12 組權重配置的前沿。子圖：Acc vs Latency、Acc vs CO₂、Acc vs Size、Latency vs CO₂ |
| Fig.2 | **超體積熱圖** | 列 = W01–W12，行 = model-dataset 組合，格值 = HV，顏色深淺表示數值高低 |
| Fig.3 | **策略分佈堆疊長條圖** | x 軸 = 權重配置，y 軸 = 方法選擇頻率，顏色 = 壓縮方法（GPTQ、AWQ、INT8 等） |
| Fig.4 | **收斂曲線** | HV vs episode number，12 組權重配置疊加，按 model-dataset 分面板 |
| Fig.5 | **雷達圖** | 蜘蛛網圖，展示各配置最佳平衡解在 5 個目標上的表現 |
| Fig.6 | **權重敏感度圖** | 針對每個目標，展示從極端權重到均衡權重時，該目標最佳指標的變化趨勢 |

---

### 7. LaTeX 表格（3 張）

| 編號 | 名稱 | 內容 |
|------|------|------|
| Tab.1 | **完整權重配置結果表** | 每個 model-dataset 組合下，12 組配置的 HV、\|PF\|、策略熵 |
| Tab.2 | **策略選擇頻率矩陣** | 行 = 權重配置，列 = 壓縮方法，格值 = 選擇次數；附卡方檢定 p 值 |
| Tab.3 | **各配置最佳解表** | 每個權重配置下，各目標的最佳值與對應壓縮策略 |

---

### 8. 輸出目錄結構

```
data/experiments/rq3/
├── manifest.json                    # 執行計畫與完成狀態
├── gpt2_gsm8k_W01/
│   ├── config.json                  # 使用的組態
│   ├── model_spec.json              # 推論出的模型規格
│   ├── pareto_frontier.json         # Pareto 最佳解集合
│   ├── final_results.json           # 最終摘要
│   ├── pareto_visualization.html    # 互動式圖表
│   └── episode_XXX/
│       ├── strategy.json            # 壓縮策略
│       └── results.json             # 評估結果
├── gpt2_gsm8k_W02/
├── ...
├── llama3_8b_commonsenseqa_W12/
└── ...

results/rq3_analysis/
├── figures/
│   ├── pareto_overlay_gpt2_gsm8k.pdf
│   ├── pareto_overlay_gpt2_commonsenseqa.pdf
│   ├── pareto_overlay_llama3_gsm8k.pdf
│   ├── pareto_overlay_llama3_commonsenseqa.pdf
│   ├── hypervolume_heatmap.pdf
│   ├── strategy_distribution.pdf
│   ├── convergence_curves.pdf
│   ├── radar_chart.pdf
│   └── weight_sensitivity.pdf
├── tables/
│   ├── hypervolume_table.tex
│   ├── strategy_frequency_table.tex
│   └── best_solutions_table.tex
└── summary.json
```

---

### 9. 需新建的檔案

| 檔案 | 說明 | 預估行數 | 對應前置作業 |
|------|------|----------|-------------|
| `config/rq3/W01_accuracy_extreme.yaml` ~ `W12_preset_memory.yaml` | 12 個 YAML 組態檔 | 各約 30 行 | P8 |
| `scripts/run_rq3_experiment.py` | Sweep runner，Click CLI | ~350 行 | P7 |
| `scripts/analyze_rq3.py` | 分析腳本：計算指標、生成圖表與表格 | ~600 行 | P9 |

---

### 10. 分析腳本設計（`scripts/analyze_rq3.py`）

**需完成 P9**。核心功能：

```bash
# 完整分析
python scripts/analyze_rq3.py --input data/experiments/rq3 --output results/rq3_analysis

# 僅生成圖表
python scripts/analyze_rq3.py --input data/experiments/rq3 --output results/rq3_analysis --figures-only

# 僅生成表格
python scripts/analyze_rq3.py --input data/experiments/rq3 --output results/rq3_analysis --tables-only
```

**分析流程：**

1. 載入所有 48 次執行的 `final_results.json` 與 `pareto_frontier.json`
2. 計算 7 項分析指標
3. 生成 6 張圖（matplotlib + seaborn）
4. 生成 3 張 LaTeX 表格
5. 輸出 `summary.json` 彙總

---

### 11. 統計分析

1. **卡方檢定**：測試壓縮方法選擇頻率是否與權重配置獨立（Tab.2）
2. **Kruskal-Wallis 檢定**：比較不同權重群組（單目標、混合、Preset）的 HV 分佈
3. **Spearman 相關**：各權重值與對應目標最佳指標的等級相關
4. **定性分析**：比較 W09–W12（Preset 對照）與 W01–W08 的行為一致性，檢測權重傳播問題

---

### 12. 驗證計畫

| 驗證項目 | 方法 |
|----------|------|
| 組態正確傳遞 | 每次執行記錄實際使用的組態（`config.json`），事後比對 |
| Mock 偵測 | 檢查結果中 `is_mock: false`，排除 mock 執行 |
| 可復現性 | 固定 seed=42，抽樣重跑 3 組確認結果一致 |
| Preset 一致性 | W09–W12 結果應與直接使用 `--preset` 的結果相似 |
| 權重傳播檢查 | 記錄協調器 prompt 中實際使用的權重，與組態檔比對 |

---

### 13. 計算資源估計

| 項目 | GPT-2 (24 次) | Llama-3-8B (24 次) |
|------|--------------|-------------------|
| 12 episodes × 24 次 | ~12 小時 | ~96 小時 |
| 分析 | ~0.5 小時 | ~0.5 小時 |
| **合計** | **~12.5 小時** | **~96.5 小時（約 4 天）** |

**OpenAI API 費用**：約 $29 USD（48 次執行 × 12 episodes × ~$0.05/次 GPT-4o 呼叫）

---

### 14. 參考的程式碼檔案

| 檔案 | 在 RQ3 中的角色 | 需修改？ |
|------|-----------------|----------|
| `scripts/run_pipeline.py` | 被 sweep runner 呼叫執行各配置 | 是（P1: `--seed`） |
| `scripts/run_rq3_experiment.py` | Sweep runner | 新建（P7） |
| `scripts/analyze_rq3.py` | 分析腳本 | 新建（P9） |
| `src/common/config.py` | `AdvancedConfig`、`RewardConfig` 定義 | 否（RQ3 使用 YAML 覆蓋） |
| `config/rq3/*.yaml` | 12 組權重配置 | 新建（P8） |
| `config/default.yaml` | 預設組態範本 | 否 |
| `src/coordinator/langgraph_coordinator.py` | 使用權重進行決策 | 建議修改（權重傳播） |
| `src/coordinator/pareto.py` | Pareto 前沿與 HV 計算 | 是（P4, P5） |
| `src/agents/search_agent.py` | Bandit 搜索 | 是（P3） |
| `src/agents/evaluation_agent.py` | 評估與碳排放量測 | 否 |

---

## RQ 4：協調器 LLM 對壓縮最佳化效能之影響

### 研究問題

**RQ 4:** How does the choice of coordinator LLM—varying in model size, capability, and compression level—affect the quality of multi-agent compression optimization, and can models compressed by the framework itself serve as effective coordinators, forming a recursive, self-applicable compression pipeline?

驅動多代理壓縮框架的協調器 LLM，在模型大小、能力與壓縮程度的不同選擇下，如何影響壓縮最佳化的品質？進一步，由本框架自身壓縮產出的模型能否作為有效的協調器，形成遞迴式、可自我應用的壓縮管線？

### 假說

1. **能力假說**：較大、較強的 LLM（GPT-4o）會產出更高品質的 Pareto 前沿，但邊際效益遞減 — GPT-4o-mini 或中型本地模型可達到 GPT-4o 80% 以上的 HV，而成本大幅降低。
2. **壓縮容忍假說**：本系統壓縮後的 Llama-3-8B（如 GPTQ-4bit）作為協調器時，其 HV 損失不超過未壓縮版本的 15%，證明壓縮管線可自我應用（self-applicable）。
3. **效率假說**：考量協調器推論成本後，中型本地模型在「每美元 HV」或「每焦耳 HV」指標上優於雲端 API 旗艦模型。

---

### 1. 實驗條件（自變數）— 協調器 LLM

#### 1.1 雲端 API 模型

| 編號 | 條件 | 模型 | 預估參數量 | 每次呼叫成本 | 說明 |
|------|------|------|-----------|-------------|------|
| L1 | GPT-4o（旗艦） | `gpt-4o` | ~200B (MoE) | ~$0.05 | 當前預設，最強能力 |
| L2 | GPT-4o-mini（輕量） | `gpt-4o-mini` | ~8B | ~$0.002 | 成本為 L1 的 1/25 |
| L3 | GPT-3.5-turbo（傳統） | `gpt-3.5-turbo` | ~20B | ~$0.005 | 舊世代基線 |

#### 1.2 本地模型（OpenAI-compatible API via vLLM/Ollama）

| 編號 | 條件 | 模型 | 參數量 | 推論方式 | 說明 |
|------|------|------|--------|----------|------|
| L4 | Llama-3-8B（未壓縮） | `Meta-Llama-3-8B-Instruct` FP16 | 8B | vLLM on GPU | 本地全精度基線 |
| L5 | Llama-3-8B（GPTQ-4bit）| RQ1 最佳 GPTQ-4bit checkpoint | 8B→~4GB | vLLM on GPU | **本系統壓縮產物** |
| L6 | Llama-3-8B（GPTQ-4bit + LoRA）| RQ1 S9 pipeline checkpoint | 8B→~4.5GB | vLLM on GPU | **本系統壓縮+微調產物** |
| L7 | Phi-2（小型） | `microsoft/phi-2` FP16 | 2.7B | vLLM on GPU | 小型模型下界 |

#### 1.3 控制條件

| 編號 | 條件 | 說明 |
|------|------|------|
| L0 | 隨機協調器 | 均勻隨機取樣壓縮策略（與 RQ1 A1 共用） |

> **L5/L6 的遞迴性**：L5 和 L6 使用的壓縮模型是本系統在 RQ1 中產出的 checkpoint。這創造了一個遞迴結構：系統壓縮出的模型，反過來驅動系統去壓縮其他模型。此設計直接測試壓縮管線的「自我應用性」（self-applicability）。

---

### 2. 目標模型（被壓縮的模型）

| 模型 | 參數量 | 選擇理由 |
|------|--------|----------|
| `gpt2` | 124M | 所有協調器 LLM 皆可快速完成，便於比較 |
| `meta-llama/Meta-Llama-3-8B-Instruct` | 8B | 真實工作負載；L4–L6 作為協調器時與目標模型同家族，測試自我壓縮場景 |

> **資源衝突處理**：當 L4–L7（本地模型）作為協調器時，需同時使用 GPU 進行協調器推論與目標模型壓縮/評估。解決方案：(1) 若有多 GPU，協調器與壓縮分置不同 GPU；(2) 若僅有單 GPU，序列化 — 先完成協調器推論（釋放 VRAM）再執行壓縮/評估；(3) L7 (Phi-2) 可在 CPU 上執行（推論約 1–3 秒/次，可接受）。

---

### 3. 固定參數

| 參數 | 設定值 | 說明 |
|------|--------|------|
| 預設 | `balanced` | 統一使用 balanced preset |
| Episodes | 15 | 與 RQ1 D1 相同 |
| 搜索方法 | bandit | `search.method=bandit` |
| 評估 | proxy (`use_proxy=True`, 200 samples) | 動態迴圈中使用 proxy |
| Benchmark | gsm8k, hellaswag, truthfulqa | 與 RQ1 相同 3 個 benchmark |
| 種子 | 42, 123, 456 | 3 個種子 |
| LLM 溫度 | 0.7 | 所有協調器統一溫度 |

---

### 4. 應變數

#### 4.1 主要指標 — 壓縮最佳化品質

| 指標 | 定義 | 說明 |
|------|------|------|
| HV（超體積） | 目標模型 Pareto 前沿支配的體積 | 直接衡量壓縮品質 |
| \|PF\| | Pareto 最佳解數量 | 前沿多樣性 |
| 最佳準確度 | 前沿上最高準確度的解 | 單目標上限 |
| 最佳延遲 | 前沿上最低延遲的解 | 單目標上限 |
| 最佳 CO₂ | 前沿上最低碳排的解 | 單目標上限 |

#### 4.2 次要指標 — 協調器效率

| 指標 | 定義 | 說明 |
|------|------|------|
| 每次協調器呼叫延遲 (ms) | 協調器從送出 prompt 到收到回覆的時間 | 反映 LLM 推論速度 |
| 總 API/推論成本 ($) | 雲端 API 費用 或 本地 GPU-hours × 電費 | 經濟成本 |
| 協調器能耗 (J) | 協調器推論本身的能量消耗 | 用 CodeCarbon/PyNVML 量測本地模型 |
| 管線總碳排 (g CO₂) | 協調器碳排 + 壓縮/評估碳排 | 端到端綠色指標 |

#### 4.3 第三指標 — 決策品質

| 指標 | 定義 | 說明 |
|------|------|------|
| JSON 有效率 (%) | 協調器產出合法 JSON 的比例 | 小型模型可能產出無效格式 |
| 策略多樣性（Shannon 熵） | 壓縮方法選擇分佈的熵 | 測試探索能力 |
| 重複策略率 (%) | 完全重複先前策略的比例 | 測試是否陷入局部最優 |
| Pareto 改進率 | 成功改進 Pareto 前沿的 episode 比例 | 探索效率 |

#### 4.4 綜合效率指標（核心創新）

| 指標 | 定義 | 說明 |
|------|------|------|
| HV / 協調器成本 | 每美元 HV | 經濟效率 |
| HV / 管線總碳排 | 每克 CO₂ 的 HV | 綠色效率 |
| HV / 管線總時間 | 每小時 HV | 時間效率 |

---

### 5. 實驗流程

#### 階段 1：準備壓縮後的協調器模型

從 RQ1 結果中取得 L5、L6 所需的 checkpoint：

```bash
# L5: 取 RQ1 S1 (GPTQ-4bit) 的 Llama-3-8B checkpoint
# L6: 取 RQ1 S9 (GPTQ-4bit + LoRA) 的 Llama-3-8B checkpoint
# 若 RQ1 尚未完成，可先以預設參數獨立執行壓縮
python scripts/run_static_baseline.py \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --condition S1 --seed 42
```

#### 階段 2：部署本地 LLM 服務

```bash
# L4: Llama-3-8B FP16
vllm serve meta-llama/Meta-Llama-3-8B-Instruct \
    --port 8001 --dtype float16

# L5: Llama-3-8B GPTQ-4bit（本系統壓縮產物）
vllm serve data/checkpoints/llama3_8b_gptq4bit/ \
    --port 8002 --quantization gptq

# L6: Llama-3-8B GPTQ-4bit + LoRA
vllm serve data/checkpoints/llama3_8b_gptq4bit_lora/ \
    --port 8003 --quantization gptq

# L7: Phi-2
vllm serve microsoft/phi-2 \
    --port 8004 --dtype float16
```

#### 階段 3：執行壓縮最佳化

對每個協調器 LLM × 目標模型 × 3 種子：

```bash
# L1: GPT-4o（雲端 API，預設行為）
python scripts/run_pipeline.py -m TARGET_MODEL -d gsm8k -e 15 \
    --preset balanced --seed SEED \
    --llm-model gpt-4o \
    -n "rq4_L1_TARGET_seedSEED"

# L2: GPT-4o-mini
python scripts/run_pipeline.py -m TARGET_MODEL -d gsm8k -e 15 \
    --preset balanced --seed SEED \
    --llm-model gpt-4o-mini \
    -n "rq4_L2_TARGET_seedSEED"

# L4: Llama-3-8B 本地（需 P10 支援 --llm-base-url）
python scripts/run_pipeline.py -m TARGET_MODEL -d gsm8k -e 15 \
    --preset balanced --seed SEED \
    --llm-model meta-llama/Meta-Llama-3-8B-Instruct \
    --llm-base-url http://localhost:8001/v1 \
    -n "rq4_L4_TARGET_seedSEED"

# L5: 壓縮後 Llama-3-8B（本系統產物）
python scripts/run_pipeline.py -m TARGET_MODEL -d gsm8k -e 15 \
    --preset balanced --seed SEED \
    --llm-model llama3-8b-gptq4bit \
    --llm-base-url http://localhost:8002/v1 \
    -n "rq4_L5_TARGET_seedSEED"
```

#### 階段 4：事後標準化評估

所有條件的最佳 Pareto 解以完整 benchmark 重新評估（非 proxy）。

#### 階段 5：協調器能耗量測

- 雲端 API（L1–L3）：以 token 數 × 每 token 能耗估算（文獻參考值）
- 本地模型（L4–L7）：以 CodeCarbon/PyNVML 直接量測推論過程能耗
- 計算管線總碳排 = 協調器碳排 + 壓縮碳排 + 評估碳排

---

### 6. 實驗矩陣

| 維度 | 數值 | 數量 |
|------|------|------|
| 協調器 LLM | L0–L7 | 8 |
| 目標模型 | gpt2, Llama-3-8B | 2 |
| 種子 | 42, 123, 456 | 3 |

**總執行次數：8 條件 × 2 目標模型 × 3 種子 = 48 次執行**

> **注意**：L5/L6 的 checkpoint 來自 RQ1。若 RQ1 尚未完成，可先以固定參數獨立壓縮 Llama-3-8B 作為替代。

---

### 7. 所需程式碼變更

#### 7.1 `CoordinatorConfig` 擴充（P10）

```python
class CoordinatorConfig(BaseModel):
    llm_model: str = "gpt-4o"
    llm_temperature: float = 0.7
    llm_provider: str = Field(
        default="openai",
        description="LLM provider: 'openai', 'openai_compatible', 'ollama'"
    )
    llm_base_url: Optional[str] = Field(
        default=None,
        description="Base URL for OpenAI-compatible API (e.g., http://localhost:8001/v1)"
    )
    worker_model: str = "gpt-4o"
    worker_temperature: float = 0.0
    max_retries: int = 3
```

#### 7.2 LLM 工廠函式（P10）

```python
def create_coordinator_llm(config: CoordinatorConfig) -> BaseChatModel:
    """Create LLM instance based on provider configuration."""
    if config.llm_provider == "openai":
        return ChatOpenAI(
            model=config.llm_model,
            temperature=config.llm_temperature,
            max_retries=config.max_retries,
        )
    elif config.llm_provider == "openai_compatible":
        return ChatOpenAI(
            model=config.llm_model,
            temperature=config.llm_temperature,
            max_retries=config.max_retries,
            openai_api_base=config.llm_base_url,
            openai_api_key="not-needed",  # local server
        )
    else:
        raise ValueError(f"Unknown provider: {config.llm_provider}")
```

#### 7.3 輸出強健化（P11）

```python
def parse_coordinator_output(raw_output: str, max_retries: int = 3) -> dict:
    """Parse coordinator output with fallback strategies."""
    # Strategy 1: Direct JSON parse
    try:
        return json.loads(raw_output)
    except json.JSONDecodeError:
        pass

    # Strategy 2: Extract JSON from markdown code block
    match = re.search(r'```json?\s*(.*?)\s*```', raw_output, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Strategy 3: Extract first {...} block
    match = re.search(r'\{.*\}', raw_output, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Cannot parse coordinator output: {raw_output[:200]}")
```

#### 7.4 新增 CLI 參數

```bash
# run_pipeline.py 新增：
--llm-base-url     # 本地 LLM 伺服器的 base URL
--llm-provider     # LLM 提供者（openai, openai_compatible）
```

---

### 8. 統計分析方案

#### 主要分析：協調器能力 vs 壓縮品質

1. **Kruskal-Wallis 檢定**：比較 8 個條件的 HV 分佈
2. **事後兩兩比較**：Dunn's test（Bonferroni 校正），特別關注：
   - L1 (GPT-4o) vs L2 (GPT-4o-mini)：旗艦 vs 輕量雲端
   - L4 (Llama FP16) vs L5 (Llama GPTQ-4bit)：壓縮對協調能力的影響
   - L5 (Llama GPTQ-4bit) vs L0 (隨機)：壓縮模型是否仍優於隨機
3. **Cliff's delta 效果量**：量化各對之間的實際差異大小

#### 次要分析：成本效益

4. **Pareto 效率前沿**：在 (HV, 成本) 空間中繪製各協調器的 Pareto 前沿
5. **Spearman 相關**：模型參數量 vs HV（測試規模效應）
6. **配對分析**：L4 vs L5 配對比較（同模型、壓縮前後），量化壓縮對協調能力的精確影響

#### 第三分析：決策品質

7. **卡方檢定**：JSON 有效率在各條件間是否有顯著差異
8. **策略多樣性分析**：各條件的 Shannon 熵比較

**顯著水準**：α = 0.05，多重比較使用 Bonferroni 校正。

---

### 9. 預期圖表

**表格**
1. 所有條件 × {HV, \|PF\|, 最佳準確度, 最佳延遲, 最佳 CO₂, 協調器成本, JSON 有效率}，含 ± 95% CI
2. 成本效益彙總：每美元 HV、每焦耳 HV、每小時 HV
3. L4 vs L5 配對比較（壓縮前 vs 壓縮後協調器）

**圖**
1. **HV 長條圖**：8 個條件的 HV（含誤差條），按 HV 排序，顏色區分雲端/本地/壓縮
2. **成本-品質散佈圖**：x = 協調器總成本, y = HV，點大小 = 模型參數量
3. **收斂曲線比較**：各協調器的 HV vs episode，觀察收斂速度差異
4. **決策品質堆疊長條圖**：JSON 有效率、策略多樣性、重複率
5. **管線總碳排分解圖**：堆疊長條，分解為協調器碳排 + 壓縮碳排 + 評估碳排
6. **自我壓縮效果圖**：L4（FP16）→ L5（GPTQ-4bit）→ L6（GPTQ+LoRA）的 HV 變化趨勢

---

### 10. 效度威脅（RQ4 特有）

| 威脅 | 緩解措施 |
|------|----------|
| 本地模型 vs API 模型的推論環境差異 | 統一使用 OpenAI-compatible API 格式，確保 prompt 一致 |
| vLLM 服務穩定性 | 每次呼叫設定超時（60 秒），失敗自動重試 |
| 小型模型 JSON 失敗率高 | 記錄並報告失敗率；失敗的 episode 以隨機策略替代（記錄） |
| GPU 資源競爭（本地協調器 + 壓縮） | 序列化執行或雙 GPU 配置（見第 2 節） |
| L5/L6 checkpoint 品質依賴 RQ1 結果 | 若 RQ1 未完成，以固定參數獨立壓縮作為替代 |
| 雲端 API 成本估算不精確 | 記錄每次呼叫的 token 數，以官方定價精確計算 |

---

### 11. 計算資源估計

| 項目 | GPT-2 目標（×3 種子） | Llama-3-8B 目標（×3 種子） |
|------|----------------------|--------------------------|
| L1–L3 雲端 API（3 條件） | ~9 hr | ~54 hr |
| L4–L7 本地模型（4 條件） | ~12 hr | ~72 hr |
| L0 隨機（1 條件） | ~3 hr | ~18 hr |
| 事後評估 | ~2 hr | ~8 hr |
| **小計** | **~26 hr** | **~152 hr** |

**總計**：約 178 小時（約 7.4 天）

**API 費用**：
- L1 (GPT-4o): 2 目標模型 × 3 種子 × 15 ep × ~$0.05 = ~$4.5
- L2 (GPT-4o-mini): ~$0.18
- L3 (GPT-3.5-turbo): ~$0.45
- L4–L7 本地: $0（GPU 電費另計）
- **API 總計**：~$5.1

**本地 GPU 額外需求**：L4（FP16 Llama-3-8B）需要 ~16GB VRAM 用於 vLLM 服務。若僅有一張 GPU，需序列化協調器推論與壓縮操作。

---

### 12. 參考的程式碼檔案

| 檔案 | 在 RQ4 中的角色 | 需修改？ |
|------|-----------------|----------|
| `src/coordinator/langgraph_coordinator.py` | 核心協調器，需支援多 LLM 後端 | 是（P10, P11） |
| `src/common/config.py` | `CoordinatorConfig` 需擴充 | 是（P10） |
| `scripts/run_pipeline.py` | 新增 `--llm-base-url`, `--llm-provider` CLI 參數 | 是（P10） |
| `scripts/run_rq4_experiment.py` | Sweep runner | 新建（P12） |
| `scripts/run_static_baseline.py` | 準備 L5/L6 的壓縮 checkpoint | 依賴（P6） |

---

## 附錄 A：共通效度威脅

以下效度威脅適用於所有四個 RQ，在此統一說明以避免重複。RQ4 特有的威脅另見 RQ4 第 10 節。

### 內部效度

| 威脅 | 緩解措施 | 適用 RQ |
|------|----------|---------|
| LLM 非確定性 | 每條件 3 個種子（RQ1, RQ2, RQ4），報告信賴區間 | RQ1, RQ2, RQ4 |
| 單種子限制 | RQ3 使用單種子；若結果差異微小則以 3 種子重跑關鍵配置 | RQ3 |
| Mock 回退 | 檢查結果中 `is_carbon_mock: false`，排除使用 mock 的執行 | 全部 |
| Proxy 偏差 | 動態迴圈中使用 proxy 加速，最終報告使用完整評估 | 全部 |
| 碳排放量測變異 | 所有條件使用一致的量測方法與推論次數 | 全部 |
| GPU 熱狀態 | 各次執行間以 `torch.cuda.empty_cache()` 重置 + 冷卻間隔 | 全部 |
| 權重傳播缺口 | P3, P4 修正前，記錄實際生效的權重以供事後比對 | RQ2, RQ3 |

### 外部效度

| 威脅 | 緩解措施 | 適用 RQ |
|------|----------|---------|
| 模型覆蓋範圍 | RQ1: 2 模型；RQ2: 4 模型跨家族（GPT-2, Phi, Mistral, Llama-3）；RQ3: 2 模型 | 全部 |
| Benchmark 代表性 | RQ1: 3 個（推理、常識、事實）；RQ2: 5 個（+程式碼、多任務）；RQ3: 2 個 | 全部 |
| 單一 GPU | 報告精確硬體規格，便於重現 | 全部 |
| LLM 服務穩定性 | GPT-4o API 可能有版本更新；記錄 API 呼叫時間與模型版本 | 全部 |
| 本地 LLM 服務穩定性 | vLLM/Ollama 可能 OOM 或崩潰；設定自動重啟與呼叫超時 | RQ4 |

### 建構效度

| 威脅 | 緩解措施 | 適用 RQ |
|------|----------|---------|
| HV 取決於參考點 | 同時報告 IGD、覆蓋率、前沿大小等多項指標 | RQ1, RQ3 |
| 動態優勢可能來自更多計算預算 | 消融 A6（5 ep）與靜態聯集比較；收斂曲線分析 | RQ1 |
| 能耗量測精確度 | 使用 CodeCarbon → PyNVML 階層，排除 mock 結果 | RQ2 |
| 代理優勢可能來自更多計算預算 | 靜態有 6–10 種策略，代理有 15 輪迭代；報告逐策略比較 | RQ1, RQ2 |
| 雲端 vs 本地推論環境差異 | 統一 OpenAI-compatible API 格式，同一 prompt 模板 | RQ4 |

---

## 附錄 B：計算資源總覽

### 各 RQ 計算需求彙總

| RQ | GPU 時間 | API 費用 |
|----|----------|---------|
| RQ1 | ~189 小時（GPT-2: 15h + Llama-3: 174h） | ~$31 |
| RQ2 | ~185.5 小時（4 模型合計） | ~$19 |
| RQ3 | ~109 小時（GPT-2: 12.5h + Llama-3: 96.5h） | ~$29 |
| RQ4 | ~178 小時（GPT-2: 26h + Llama-3: 152h） | ~$5 |
| **總計** | **~661.5 小時（約 27.6 天）** | **~$84** |

### 並行化建議

- RQ3 可獨立先行（不依賴 RQ1/RQ2 基線）
- RQ1 的 gpt2 基線與 RQ2 的 gpt2 基線可共用（節省 ~3 小時）
- RQ1 的 Llama-3 基線與 RQ2 的 Llama-3 基線可共用（節省 ~24 小時）
- 若有多 GPU，RQ1 消融（Llama-3 only）與 RQ2 代理執行（phi-2, Mistral-7B）可並行
- RQ4 的雲端 API 條件（L1–L3）可與 Phase 4/5 並行；本地模型條件（L4–L7）需等 RQ1 產出 checkpoint
- RQ4 的 API 費用極低（~$5），主要成本在 GPU 時間

### 風險緩衝

建議預留 20% 的時間緩衝（約 132 小時），用於：
- 因 Mock 回退需重跑的執行
- GPU OOM 後調整 batch_size 重跑
- 碳排放量測異常值排除後的補充量測
- vLLM 服務崩潰重啟與重跑（RQ4 本地模型條件）
- 小型模型 JSON 解析失敗過多時的 prompt 工程迭代
