## AegisPlan Ablation Benchmark Results (25 Clinical Briefs)
*NVIDIA Hackathon Engineering Telemetry & Code Compliance Evaluation*

| System Variant | Syntax Validity (%) | Pre-Repair Overlap (%) | Post-Repair Overlap (%) | ADA / FGI Rule Pass Rate (%) | Lock Preservation (%) | P50 / P95 Latency (s) | Cost per Accepted Plan ($) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Baseline: OpenAI GPT-4o-mini (Unconstrained JSON)** | 82.5% | 21.4% | 21.4% | 36.0% | 62.0% | ~2.5s / 4.2s | $0.0004 |
| **2. + EBNF Grammar (FML-DSL)** | **100.0%** | 14.8% | 14.8% | 54.0% | 85.0% | ~1.7s / 2.8s | $0.0003 |
| **3. + Module 2 Validator & CP-SAT Repair** | **100.0%** | 14.8% | **0.0%** | **100.0%** | **100.0%** | ~0.037s / 0.123s | $0.0009 *(Best-of-3)* |
| **4. + Fine-Tuned Model (Ours) + CP-SAT** | **100.0%** | **1.4%** | **0.0%** | **100.0%** | **100.0%** | **~0.4s / 0.4s** | **$0.0000 (100% Local)** |

### Conflict Core Diagnostic Evaluation (5 Stress Tests):
- **Infeasible Diagnostics Success Rate**: **5 / 5 (100.0%)**
- All 5 impossible designs correctly isolated the conflicting requirements (e.g., `BOUNDARY_CAPACITY_EXCEEDED`, `LOCKED_ROOM_corridor_1`) in $<5	ext{ ms}$ with 0 false completions.
