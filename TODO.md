# TODO

剩餘待實作項目清單。

## 高優先級

### 整合 lm-eval ✅
- [x] 替換自訂 evaluators 為 lm-eval harness
- [x] 支援更多 benchmark (MMLU, HellaSwag, ARC, WinoGrande, etc.)
- [x] 統一評估介面

### 完善 Evolutionary Search
- [x] `src/agents/search_agent.py` - evolutionary_search 的 fitness 目前是隨機值
- [x] 需要連接真實評估結果作為 fitness

### Pruning Agent ✅
- [x] 實作 magnitude pruning
- [x] 實作 structured pruning
- [x] 整合到 LangGraph workflow

### LoRA/QLoRA Agent ✅
- [x] 實作 LoRA fine-tuning wrapper
- [x] 實作 QLoRA (4-bit + LoRA)
- [x] 整合到壓縮策略選項

## 中優先級

### Spec Inference 改進
- [ ] 從 HuggingFace Hub API 動態取得模型資訊
- [ ] 支援更多模型架構的參數估算
- [ ] 移除硬編碼的 MODEL_SIZE_DATABASE

### Search Agent 改進
- [ ] Multi-Armed Bandit 整合真實 reward
- [ ] 實作 Population-based Training (PBT)
- [ ] 加入 early stopping 策略

### 評估系統改進
- [x] TruthfulQA evaluator 使用真實 MC scoring (via lm-eval)
- [x] BigBench Hard 支援更多子任務 (via lm-eval)
- [ ] 加入 perplexity 評估

## 低優先級

### MLflow 整合
- [ ] 記錄實驗參數和結果
- [ ] 模型 registry
- [ ] 實驗比較 dashboard

### Streamlit Dashboard
- [ ] 即時監控壓縮進度
- [ ] Pareto frontier 視覺化
- [ ] 實驗結果比較

### Docker 部署
- [ ] Dockerfile for CPU
- [ ] Dockerfile for GPU (CUDA)
- [ ] docker-compose 設定

### 蒸餾支援
- [ ] 實作 knowledge distillation wrapper
- [ ] Teacher-student training loop
- [ ] 整合到壓縮策略

## 已完成

- [x] LangGraph 重構
- [x] GPT-4o 協調器
- [x] 真實量化工具 (AutoRound, GPTQ, AWQ, INT8)
- [x] BenchmarkRunner 整合
- [x] LatencyEvaluator 真實測量
- [x] codecarbon 能源追蹤
- [x] HumanEval 程式碼執行
- [x] VRAM 估算 (HF Hub API)
- [x] Pareto frontier 追蹤
- [x] CLI 介面
- [x] lm-eval harness 整合 (支援 MMLU, HellaSwag, ARC, WinoGrande 等)

## 檔案對照

| 待改進檔案 | 問題 |
|-----------|------|
| `src/common/model_utils.py` | 靜態 MODEL_SIZE_DATABASE (已整合到共用模組) |

### 新增檔案 (lm-eval 整合)

| 檔案 | 說明 |
|-----|------|
| `src/evaluation/evaluators/lm_eval_evaluator.py` | lm-eval harness wrapper |
