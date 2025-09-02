# 🧮 RTX 4090 Math Problem Solving Deployment

A production-ready deployment architecture for intelligent math problem solving using a cascading model approach with carbon efficiency gates (CRE). This system deploys your trained 1.5B math model alongside escalation tiers (7B, GPT-5) with comprehensive verification, iterative repair, and energy monitoring.

## 🏗️ Architecture Overview

This deployment implements a smart routing system that maximizes accuracy while minimizing computational carbon cost:

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌─────────────┐
│   Client    │ ── │    Router    │ ── │ 1.5B Model  │ ── │ PoT Sandbox │
│  Question   │    │  (CRE Gates) │    │ (vLLM+AWQ)  │    │ (Executor)  │
└─────────────┘    └──────────────┘    └─────────────┘    └─────────────┘
                           │                    │                 │
                    ┌──────┴──────┐     ┌──────┴──────┐  ┌───────┴───────┐
                    │ 7B Model    │     │ PRM Server  │  │ Unit Testing  │
                    │(Speculative)│     │ (Step Valid)│  │ & Verification│
                    └─────────────┘     └─────────────┘  └───────────────┘
                           │
                    ┌──────┴──────┐
                    │   GPT-5     │
                    │ (Ultimate)  │
                    └─────────────┘
```

## 🔧 Core Components

### 1. Router (`router/`)
**The intelligent orchestrator** that manages the entire problem-solving pipeline:

- **Difficulty Assessment**: Lexical analysis using mathematical keywords to classify problems as "easy", "medium", or "hard"
- **Budget Planning**: Dynamically adjusts k-sampling, temperature, and token limits based on difficulty
- **Cascading Inference**: 1.5B → 7B → GPT-5 with carbon efficiency gates
- **Iterative Repair**: StepCo-style iterative improvement for low-confidence solutions
- **Energy Monitoring**: Real-time GPU power consumption tracking via NVML
- **Carbon Efficiency Ratio (CRE)**: Blocks escalation if accuracy gain per gram CO₂ is too low

**Key Features**:
- Smart budgeting: `easy` problems use k=1, `hard` problems use k=4
- Self-consistency voting across multiple generations
- Confidence scoring combining execution success, PRM scores, and majority voting
- Rolling energy averages for accurate carbon budgeting

### 2. PoT Sandbox (`pot_sandbox/`)
**Secure Python execution environment** for Program-of-Thought (PoT) code:

- **Isolated Execution**: Sandboxed Python environment with resource limits
- **Unit Test Support**: Executes optional assertion-based tests provided with problems
- **SymPy Verification**: Mathematical equivalence checking using symbolic computation
- **Security Hardening**: CPU/memory limits, process isolation, timeout protection

**Template Structure**:
```python
# Imports: math, fractions, sympy, etc.
{user_code}

def _ci_entry():
    {tests}  # Run unit tests first
    val = solve()  # Call user's solve() function
    print("Answer:", val)
```

### 3. PRM Server (`prm_server/`)
**Process Reward Model** for step-by-step solution validation:

- **Step-Level Scoring**: Evaluates individual reasoning steps for mathematical validity
- **GPU/CPU Flexible**: Automatically detects CUDA availability, falls back to CPU
- **Batch Processing**: Efficient batch inference for multiple solution steps
- **Memory Optimized**: Uses fp16 on GPU, fp32 on CPU

**Input Format**: Question + step pairs formatted for binary classification
**Output**: Confidence score [0.0, 1.0] for solution validity

### 4. vLLM Servers
**High-performance model serving** with advanced optimizations:

#### 1.5B Solver (`vllm-solver-1p5b`)
- **Purpose**: Primary problem solver using your trained math model
- **Configuration**: Conservative GPU utilization (35%) to share resources
- **Features**: Continuous batching, chunked prefill, AWQ quantization

#### 7B Escalator (`vllm-escalate-7b`)
- **Purpose**: Higher-capacity model for complex problems
- **Configuration**: Speculative decoding using 1.5B as draft model
- **Features**: Aggressive GPU utilization (80%), draft token speculation

## 🌍 Carbon Efficiency System

### Carbon Efficiency Ratio (CRE)
The system implements a novel CRE metric to balance accuracy and environmental impact:

```
CRE = Δ Accuracy (%) / Carbon Cost (grams CO₂)
```

**Escalation Rules**:
- **1.5B → 7B**: Only escalate if CRE > 0.05%/g (configurable via `CRE_THRESHOLD_PCT_PER_G`)
- **7B → GPT-5**: Similar threshold with API-based carbon estimates

### Energy Monitoring
- **Real-time Power Sampling**: NVML-based GPU power monitoring at 200ms intervals
- **Rolling Averages**: Learns actual energy consumption patterns over time
- **Grid Integration**: Optional ElectricityMaps API for real-time carbon intensity

## 📊 Problem Solving Pipeline

### Phase 1: Initial Solve (1.5B)
1. **Difficulty Assessment**: Keyword analysis + length heuristics
2. **Budget Planning**: Set k, temperature, max_tokens based on difficulty
3. **Multi-candidate Generation**: Self-consistency sampling
4. **Verification**: Execute code in sandbox, check with PRM
5. **Confidence Scoring**: Weighted combination of execution success, PRM score, majority vote

### Phase 2: Iterative Repair
If confidence < 0.80:
1. **Repair Prompt**: "Fix the wrong steps in this solution"
2. **Multi-round Repair**: 1.5B for early rounds, 7B for final round
3. **Progressive Improvement**: Only keep repairs that improve confidence

### Phase 3: Escalation Gates
If still low confidence after repair:

#### 7B Gate
1. **CRE Calculation**: Estimate accuracy delta vs. energy cost
2. **Gate Decision**: Block if CRE below threshold
3. **Speculative Execution**: Use 1.5B as draft model for 7B

#### GPT-5 Gate (Optional)
1. **Policy-based CRE**: Estimate API carbon cost
2. **Ultimate Escalation**: High-capacity commercial model

## 🛠️ Configuration

### Environment Variables (`.env`)
```bash
# Carbon monitoring
EM_TOKEN=your_electricity_maps_token
EM_ZONE=TW  # Your grid region

# CRE policy
CRE_THRESHOLD_PCT_PER_G=0.05  # Minimum efficiency for escalation
UPLIFT_7B=0.25               # Expected accuracy improvement 1.5B→7B
UPLIFT_GPT5=0.45             # Expected accuracy improvement 7B→GPT-5

# Model paths (inside containers)
SOLVER_MODEL=/models/math-1p5b-awq
ESCALATE_MODEL=/models/qwen2.5-math-7b-awq

# Performance tuning
SOLVER_GPU_UTIL=0.35    # Conservative for resource sharing
ESCALATE_GPU_UTIL=0.80  # Aggressive for performance
```

### Model Requirements
1. **1.5B Math Model**: Your trained model in AWQ quantized format
2. **7B Math Model**: Base mathematical reasoning model (e.g., Qwen2.5-Math-7B-AWQ)
3. **PRM Model**: Process reward model for step validation (optional)

## 🚀 Deployment

### Prerequisites
- Docker & Docker Compose
- NVIDIA Container Runtime
- RTX 4090 with 24GB VRAM
- Models downloaded to `/home/you/models/`

### Quick Start
```bash
# Clone and configure
cd deploy-4090
cp .env.example .env
# Edit .env with your settings

# Update model paths in docker-compose.yml
# Point volumes to your actual model directory

# Launch all services
docker compose --env-file .env up -d

# Verify services
curl -s http://localhost:8001/v1/models | jq .   # 1.5B
curl -s http://localhost:8002/v1/models | jq .   # 7B
curl -s http://localhost:8080/docs               # Router UI
```

### Test Query
```bash
curl -s -X POST http://localhost:8080/solve \
  -H "Content-Type: application/json" \
  -d '{
    "question": "If 2x + 3 = 11, what is x?",
    "repair_rounds": 2
  }' | jq .
```

## 📈 Performance Optimization

### Single RTX 4090 Tuning
- **Memory Allocation**: 35% for 1.5B, 80% for 7B when active
- **Speculative Decoding**: Uses 1.5B as draft model for 7B acceleration
- **Continuous Batching**: Maximizes throughput via vLLM optimizations

### Scaling Strategies
- **Vertical**: Increase `MAX_BATCHED_TOKENS` for higher throughput
- **Horizontal**: Deploy multiple instances with load balancing
- **Model Optimization**: Consider Medusa multi-head speculation

## 🔍 Monitoring & Observability

### Response Structure
```json
{
  "route": "1.5B+repair",
  "confidence": 0.87,
  "verified": true,
  "answer": "4",
  "text": "# Solution code...",
  "energy_wh_1p5b": 2.4,
  "gco2_1p5b": 1.08,
  "difficulty": "easy",
  "escalation_blocked": null
}
```

### Key Metrics
- **Route Distribution**: Which models are being used
- **CRE Block Rate**: How often escalation is blocked for efficiency
- **Energy Consumption**: Real-time and rolling averages
- **Verification Success**: Code execution and mathematical correctness

## 🧪 Advanced Features

### Unit Test Integration
The sandbox supports executing unit tests alongside solution code:

```python
# Problem with tests
{
  "question": "Find the roots of x^2 - 5x + 6 = 0",
  "tests": [
    "assert len(roots) == 2",
    "assert 2 in roots and 3 in roots"
  ]
}
```

### Custom Difficulty Override
Force specific solving approaches:
```python
{
  "question": "Complex problem...",
  "mode": "cot",        # Force Chain-of-Thought
  "k": 8,               # Override sampling count
  "temperature": 0.1    # Override temperature
}
```

## 🔧 Troubleshooting

### Common Issues

**Out of Memory (OOM)**
- Reduce `MAX_MODEL_LEN` to 4096 or lower
- Temporarily stop 1.5B server to load 7B exclusively
- Lower `SOLVER_GPU_UTIL` to 0.2 or below

**Low Speculative Acceptance**
- Math problems may have low draft acceptance rates
- Try reducing `SPEC_DRAFT_TOKENS` to 3
- Consider disabling speculation for pure accuracy

**Energy Monitoring Unavailable**
- Install nvidia-ml-py3: `pip install nvidia-ml-py3`
- Ensure NVIDIA drivers support NVML
- System falls back to zero energy if unavailable

## 📚 Research Context

This deployment implements several cutting-edge techniques:

- **Cascading Inference**: Tiered model deployment for cost-performance optimization
- **Carbon Efficiency Ratios**: Novel metric balancing accuracy and environmental impact
- **Iterative Repair**: StepCo-inspired iterative solution improvement
- **Program-of-Thought**: Code-based mathematical reasoning with execution verification
- **Process Reward Models**: Step-by-step solution validation

## 🤝 Contributing

To extend this system:

1. **Add New Models**: Update docker-compose.yml with additional vLLM services
2. **Custom Verification**: Extend sandbox with domain-specific test runners
3. **Enhanced PRM**: Train specialized Process Reward Models for your domain
4. **Monitoring**: Add Prometheus endpoints for production observability

## 📄 License

This deployment template is provided as-is for educational and research purposes. Ensure compliance with model licenses and API terms of service.