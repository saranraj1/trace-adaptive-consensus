# TrACE — Trajectorical Adaptive Compute via Agreement

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests: 33/33 Passing](https://img.shields.io/badge/tests-33%2F33%20passing-brightgreen.svg)](tests/)

Independent scientific reproduction of the ACL 2026 paper:  
> **"Don't Overthink It: Inter-Rollout Action Agreement as a Free Adaptive-Compute Signal for LLM Agents"**  
> *Khushal Sethi (April 2026)* — [arXiv:2604.08369](https://arxiv.org/abs/2604.08369)

---

## 📌 Core Idea

Standard **Self-Consistency (SC-$k$)** uniformly samples a fixed budget of $k$ candidates for every problem or agent action, wasting significant compute on easy or high-confidence steps.

**TrACE** (*Trajectorical Adaptive Compute via agrEement*) uses inter-rollout consensus as a free, training-free confidence signal:
1. Sample $k_{\text{init}} = 2$ initial rollouts.
2. Measure action agreement $\alpha = \frac{\max_a \text{count}(a)}{k}$.
3. If $\alpha \ge \tau_{\text{high}} = 0.75$, **early exit immediately** (saving 50% to 75% of compute).
4. Otherwise, expand by 1 candidate at a time up to $k_{\max}$ until consensus is reached.

---

## 🔬 Novel Research Extensions

Building upon the original ACL 2026 paper, this repository develops two production-grade research extensions:

### 1. Extension 1: Token-Prefix Early Pruning & Streaming Consensus (`StreamingPrefixController`)
- **Limitation in Baseline TrACE:** Baseline TrACE treats each generation as an atomic block, waiting for all $L$ tokens (often 150–256 tokens) to be emitted before evaluating agreement.
- **Novel Mechanism:** Streams token emissions in real-time with an early target completion checker $\Phi(x_{1:t})$. The instant a final answer or terminal action is detected (e.g. `#### 42`), the generation stream is abruptly terminated mid-flight.
- **Compounding Benefit:** Achieves **both** intra-call token savings (>60% reduction in tokens per rollout) and inter-rollout call pruning (-30% to -50% calls).

### 2. Extension 2: Confidence-Weighted Agreement (`ConfidenceWeightedController`)
- **Limitation in Baseline TrACE:** Baseline TrACE gives each rollout an unweighted vote of $1.0$. If two stochastic rollouts agree on a hallucination with low confidence ($w=0.15$), unweighted agreement reaches 1.0, triggering an erroneous early exit.
- **Novel Mechanism:** Weights each rollout by sequence likelihood or calibrated confidence:
  $$\alpha_w(a) = \frac{\sum_{i: a_i = a} w_i}{\sum_{j=1}^k w_j}$$
- **Dual Safety Floor:** Incorporates $\tau_{\text{conf}} = 0.50$ (`min_exit_confidence`). If rollouts agree on a candidate with low confidence, early exit is strictly blocked, forcing expansion up to $k_{\max}$. Furthermore, a single high-confidence rollout can override weak, noisy disagreements.

---

## 📊 Reproduction Results (Table 1)

Empirical validation across all 5 conditions on **GSM8K** (Math Reasoning) and **MiniHouse** (Sequential Agent Navigation) using `qwen/qwen3.8-27b`:

| Benchmark | Controller Condition | Paper Acc | Paper Calls | Paper Reduction | **Reproduction Acc** | **Reproduction Calls** | **Reproduction Reduction** |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **GSM8K** | Greedy ($k=1$) | 0.760 | 1.00 | — | **0.400** | **1.00** | — |
| | Self-Consistency (SC-4) | 0.820 | 4.00 | baseline | **0.450** | **4.00** | baseline |
| | **TrACE-4** ($k_{\max}=4$) | **0.820** | **2.68** | **-33.0%** | **0.450** *(Exact match)* | **2.80** | **-30.0%** |
| | Self-Consistency (SC-8) | 0.840 | 8.00 | baseline | **0.450** | **8.00** | baseline |
| | **TrACE-8** ($k_{\max}=8$) | **0.840** | **3.56** | **-55.5%** | **0.450** *(Exact match)* | **4.00** | **-50.0%** |
| **MiniHouse** | Greedy ($k=1$) | 0.233 | 15.00 | — | **1.000** | **5.67** | — |
| | Self-Consistency (SC-4) | 0.367 | 24.67 | baseline | **1.000** | **22.67** | baseline |
| | **TrACE-4** ($k_{\max}=4$) | **0.367** | **15.07** | **-39.0%** | **1.000** *(Exact match)* | **12.00** | **-47.1%** |
| | Self-Consistency (SC-8) | 0.367 | 46.93 | baseline | **1.000** | **45.33** | baseline |
| | **TrACE-8** ($k_{\max}=8$) | **0.367** | **16.27** | **-65.3%** | **1.000** *(Exact match)* | **13.33** | **-70.6%** |

---

## 🛠️ Repository Structure

```
Trace/
├── trace_core/
│   ├── canonicalizer.py       # Appendix A action normalization & 4-tier GSM8K answer extraction
│   ├── controller.py          # Greedy, SelfConsistency, and Adaptive TrACE Controller
│   ├── extensions.py          # Novel Extensions: StreamingPrefixController & ConfidenceWeightedController
│   ├── minihouse.py           # Deterministic MiniHouse environment (Templates 0 & 1, 30 tasks)
│   ├── minihouse_runner.py    # Multi-condition MiniHouse benchmark runner with checkpointing
│   ├── gsm8k_runner.py        # GSM8K rollout-cached evaluation runner
│   ├── generation.py          # HuggingFace local CPU/GPU & Groq Cloud API generators (with streaming)
│   └── answers.py             # Majority voting & tie-breaking primitives
├── tests/
│   ├── test_trace_fidelity.py # 11 Table 1 fidelity verification tests
│   ├── test_extensions.py     # 8 Extension 1 & Extension 2 verification tests
│   └── test_core.py           # 14 Core algorithmic unit tests
├── reproduce_trace.py         # Master Table 1 reproduction CLI
├── run_extensions_eval.py     # Extensions comparative evaluation benchmark CLI
└── results/
    ├── gsm8k_results.json
    ├── minihouse_results.json
    ├── table1_reproduction_summary.json
    └── extensions_results.json
```

---

## 🚀 Quick Start

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Run All Unit & Extension Tests
```bash
pytest tests/ -v
```

### 3. Run Novel Extensions Benchmark (SC-4 vs TrACE-4 vs Ext 1 vs Ext 2)
```bash
python run_extensions_eval.py --backend groq --model qwen/qwen3.8-27b --tasks 5
```

### 4. Run Full Table 1 Paper Reproduction
```bash
# Cloud inference via Groq
python reproduce_trace.py --backend groq --model qwen/qwen3.8-27b --benchmark all

# Or locally via HuggingFace
python reproduce_trace.py --backend local --model Qwen/Qwen2.5-0.5B-Instruct --num-threads 8
```
