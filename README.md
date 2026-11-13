# AegisPlan: Neuro-Symbolic Healthcare Spatial Synthesis Engine
**Deterministic Layout Generation & Statutory Compliance Verification for Healthcare Outpatient Facilities**  
*Fine-tuned NVIDIA Nemotron Nano 9B v2 • 4-Reward GRPO Alignment • Google OR-Tools CP-SAT Combinatorial Repair*

[![Hugging Face Models](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Merged%20Model-blue)](https://huggingface.co/VenuArvind/aegisplan-nemotron-nano-9b)
[![Hugging Face GGUF](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-GGUF%20Quantized-green)](https://huggingface.co/VenuArvind/aegisplan-nemotron-nano-9b-gguf)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Key Result:** AegisPlan couples an autoregressive spatial drafter (fine-tuned Nemotron Nano 9B) with combinatorial constraint programming (OR-Tools CP-SAT), eliminating inter-room overlap to **0.0%** and reaching **100.0% ADA / FGI statutory code compliance** with a mean CP-SAT solve time of **6.88 ms** across 25 standardized benchmark briefs and 8 out-of-distribution clinical specialties.

---

## Quickstart (3 Commands)

```bash
# 1. Install package in editable mode with dependencies
pip install -e .

# 2. Download quantized GGUF weights (5.5 GB) into checkpoints/
python scripts/download_from_hf.py

# 3. Generate floorplan and audit statutory compliance from CLI
python -m aegisplan.inference_cli \
  --brief "Primary care clinic requiring central circulation corridor, 4 outpatient exam rooms (>= 11.15 m2), ADA restroom, and clean/soiled utility support." \
  --boundary-w 18000 --boundary-h 12000 \
  --out clinic_plan.json --svg clinic_plan.svg
```

Launch the interactive 2D CAD canvas and 3D WebGPU walkthrough:
```bash
uvicorn aegisplan.server:app --host 127.0.0.1 --port 8000
```

---

## Empirical Benchmarks & Ablation Studies

### 1. Primary Benchmark (25 Standardized Briefs)

Evaluated across **25 clinical briefs** (20 feasible outpatient layouts + 5 conflict stress tests designed to induce geometric/code infeasibility):

| System Variant | Syntax Validity | Pre-Repair Overlap | Post-Repair Overlap | ADA / FGI Pass Rate | Lock Preservation | Median Latency | Cost / Plan |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Baseline: OpenAI GPT-4o-mini (Unconstrained JSON)** | 82.5% | 21.4% | 21.4% *(no repair)* | 36.0% | 62.0% | ~2.5s | $0.0004 |
| **2. + EBNF Grammar (`fml_dsl.ebnf`)** | **100.0%** | 14.8% | 14.8% *(no repair)* | 54.0% | 85.0% | ~1.7s | $0.0003 |
| **3. + Validator & CP-SAT Repair** | **100.0%** | 14.8% | **0.0%** | **100.0%** | **100.0%** | ~37 ms *(solver)* | $0.0009 |
| **4. + Nemotron Nano 9B GRPO (Ours)** | **100.0%** | **1.2%** | **0.0%** | **100.0%** | **100.0%** | **< 15 ms *(solver)*** | **$0.0000 (Local)** |

> **Evaluation Leakage Prevention Note:** All 25 evaluation briefs are disjoint from the 400 SFT training examples, 50 validation examples, and 500 GRPO prompt briefs (0 exact text matches and 0 token-normalized collisions). Verified via `tests/test_evaluation_leakage.py` (`3 passed in 0.52s`).

#### Conflict Core Extraction (Infeasible Stress Tests)
For contradictory requirements (e.g., placing 10 exam rooms in a 10m x 10m boundary), CP-SAT activates assumption literals (`model.AddAssumptions()`) and extracts the **Minimal Unsatisfiable Core** in **< 5 ms**, identifying the conflicting room constraints with a **100.0% diagnostic detection rate (5/5)**.

---

### 2. Generalization on Out-of-Distribution (OOD) Clinical Specialties

To test generalization beyond the primary training distribution, AegisPlan was evaluated on **8 unseen clinical facility archetypes** (`dataset/ood_clinical_briefs.json`), spanning emergency triage, endoscopy, pediatric operatory rooms, cancer infusion bays, behavioral health consult suites, hemodialysis water treatment, ultrasound suites, and sports rehabilitation:

| Facility Specialty | Boundary | Pre-Repair Overlap | Post-Repair Overlap | ADA / FGI Pass Rate | CP-SAT Solve Time | Solver Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| High-Density Urgent Care & Triage | 22m x 14m | 0.0% | **0.0%** | **100.0% (8/8)** | 7.00 ms | OPTIMAL |
| Rural Ambulatory Surgery & Endoscopy | 26m x 16m | 0.0% | **0.0%** | **100.0% (8/8)** | 7.00 ms | OPTIMAL |
| Pediatric Dental & Orthodontic Clinic | 18m x 12m | 0.0% | **0.0%** | **100.0% (8/8)** | 7.00 ms | OPTIMAL |
| Outpatient Oncology Infusion Center | 24m x 15m | 0.0% | **0.0%** | **100.0% (8/8)** | 7.00 ms | OPTIMAL |
| Behavioral Health & Crisis Clinic | 20m x 12m | 0.0% | **0.0%** | **100.0% (8/8)** | 7.00 ms | OPTIMAL |
| Hemodialysis Treatment Center | 25m x 15m | 0.0% | **0.0%** | **100.0% (8/8)** | 7.00 ms | OPTIMAL |
| Diagnostic Imaging & Ultrasound Suite | 22m x 14m | 0.0% | **0.0%** | **100.0% (8/8)** | 6.00 ms | OPTIMAL |
| Sports Medicine & Rehabilitation | 28m x 18m | 0.0% | **0.0%** | **100.0% (8/8)** | 7.00 ms | OPTIMAL |
| **OOD Aggregate Mean** | — | **0.0%** | **0.0%** | **100.0%** | **6.88 ms** | **100% Feasible** |

*Raw telemetry saved to `benchmarks/ood_benchmark_results.json`.*

---

### 3. GRPO Multi-Seed Variance & Reward Ablation Study

Evaluated across random seeds (42, 137, 2026) across 20 feasible clinical briefs (`scripts/benchmark_reward_ablation.py`):

| Reward Formulation | Pre-Repair Overlap | ADA/FGI Rule Pass Rate | CP-SAT Feasibility |
| :--- | :---: | :---: | :---: |
| **Full Compound Reward ($R_{\text{all}}$)** | **1.4 ± 0.07%** | **100.0 ± 0.00%** | **100.0%** |
| Overlap Reward Alone ($R_{\text{overlap}}$) | 2.1 ± 0.09% | 68.4 ± 0.75% | 95.0% |
| Regulatory Compliance Alone ($R_{\text{reg}}$) | 8.7 ± 0.24% | 96.0 ± 0.35% | 90.0% |
| Connectivity & Egress Alone ($R_{\text{conn}}$) | 11.3 ± 0.31% | 71.9 ± 0.65% | 85.0% |
| Area Fidelity Alone ($R_{\text{area}}$) | 13.6 ± 0.40% | 63.9 ± 0.82% | 80.0% |
| SFT Baseline (No GRPO Alignment) | 14.9 ± 0.35% | 53.9 ± 0.89% | 80.0% |

*Findings:* Isolating $R_{\text{overlap}}$ minimizes geometric collision rates but neglects door circulation and ADA turning clearance compliance. Isolating $R_{\text{reg}}$ ensures dimension minimums but allows spatial packing collisions (8.7% overlap). Multi-objective compound alignment achieves 100% solver feasibility. Raw output saved to `benchmarks/reward_ablation_results.json`.

---

### 4. Asynchronous Non-Blocking Streaming Pipeline Benchmark (SSE)

To eliminate head-of-line UI blocking during multi-candidate neural drafting and CP-SAT interval solving, AegisPlan implements an asynchronous event stream (`POST /api/generate/stream`) over HTTP Server-Sent Events (`text/event-stream`). The pipeline emits granular lifecycle events (`brief_validated`, `draft_synthesized`, `solver_progress`, `candidate_evaluated`, and `synthesis_complete`) directly across the wire as stages complete.

Evaluated over 50 concurrent requests against a local socket Uvicorn server (`benchmarks/benchmark_async_throughput.py`):

| Pipeline Architecture | TTFB (P50) | TTFB (P90) | TTFB (P95) | Throughput | Drop Rate |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Sync Blocking** (`/api/generate`) | 17.71 ms | 17.92 ms | 18.06 ms | 55.7 req/s | 0.0% |
| **Async SSE Streaming** (`/api/generate/stream`) | **1.87 ms** | **1.93 ms** | **1.94 ms** | **57.0 req/s** | **0.0%** |
| **Improvement** | **-89.4%** | **-89.2%** | **-89.3%** | **+2.3%** | **Rock-solid** |

*Key Systems Takeaways:*
- **Time-to-First-Byte (TTFB) slashed by 89.4%** (P50 drops from 17.71 ms to 1.87 ms), providing immediate UI responsiveness and real-time solving telemetry to architects.
- **Concurrent Throughput sustained at 57.0 req/s** under sustained async load with zero connection drops or socket starvation.
- Verified wire frames emit structured JSON payloads adhering to the `StreamingEvent` schema. Raw benchmark telemetry is saved to `benchmarks/async_streaming_results.json`.

---

## Architecture Flow

```
                      [Natural Language Clinical Brief + Locked Boundaries]
                                               │
                                               ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ TIER 1: CONSTRAINED NEURAL DRAFTING (NVIDIA Nemotron Nano 9B v2 GGUF)                  │
│  • Architecture: Hybrid Mamba-2 SSM + Transformer Attention (nemotron_h)               │
│  • Training: SFT warmup + 4-Reward GRPO (RLVR) with FML-DSL decimeter grid tokens      │
│  • Inference: llama-cpp-python with Metal acceleration (< 6 GB unified memory)         │
└────────────────────────────────────────────────────────────────────────────────────────┘
                                               │
                                               ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ TIER 2: DETERMINISTIC REGULATORY AUDITOR (ADA, FGI, IBC)                               │
│  • ADA §404.2.3: Door clear opening (≥ 915mm clinical, ≥ 815mm standard)               │
│  • ADA §403.5.1: Corridor minimum clearance (≥ 1830mm / 6 ft circulation)              │
│  • ADA §304.3: Inscribed wheelchair turning circle (d ≥ 1525mm / 60-inch)              │
│  • FGI §2.1-2.2: Exam rooms ≥ 11.15 m² (120 sq ft) with min dimension ≥ 3048mm        │
│  • FGI §2.1-3: Strict topological separation between clean and soiled utilities        │
│  • IBC Table 1017.2: Graph BFS egress travel distance ≤ 45,720mm (150 ft)              │
└────────────────────────────────────────────────────────────────────────────────────────┘
                                               │
                                               ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ TIER 3: COMBINATORIAL CP-SAT REPAIR (Google OR-Tools)                                  │
│  • 2D Interval Disjointness (AddNoOverlap2D) on 100mm modular grid                     │
│  • Preserves locked structural rooms; minimizes L1 perturbation from draft             │
│  • Unsat Diagnostics: Assumption-based Minimal Unsatisfiable Core extraction (< 5 ms)  │
└────────────────────────────────────────────────────────────────────────────────────────┘
                                               │
                                               ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ EXPORTS & CAD CONSUMPTION (plan.json Building Contract)                                │
│  • 2D CAD Canvas (interactive room drag & lock) • 3D Three.js WebGPU Walkthrough       │
│  • Regulatory Audit Matrix (Pass/Fail citations) • Scalable Vector Graphics (SVG)      │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Published Checkpoints

| Repository | Format | Size | Description |
| :--- | :--- | :--- | :--- |
| [`VenuArvind/aegisplan-nemotron-nano-9b-gguf`](https://huggingface.co/VenuArvind/aegisplan-nemotron-nano-9b-gguf) | **GGUF Q4_K_M** | **5.5 GB** | Recommended local model for Apple Silicon Metal |
| [`VenuArvind/aegisplan-nemotron-nano-9b-gguf`](https://huggingface.co/VenuArvind/aegisplan-nemotron-nano-9b-gguf) | **GGUF Q8_0** | **9.2 GB** | Near-lossless precision quantized GGUF |
| [`VenuArvind/aegisplan-nemotron-nano-9b`](https://huggingface.co/VenuArvind/aegisplan-nemotron-nano-9b) | **SafeTensors (BF16)** | **18 GB** | Standalone merged base + LoRA model with tokenizer |

---

## Assumptions & Limitations

1. **Synthetic Training Distribution:** Training layouts and GRPO rollouts were generated using procedural outpatient templates. Performance on brownfield renovations featuring irregular structural column grids, load-bearing shear walls, or multi-story mechanical/plumbing riser penetrations requires explicit locked room declarations in the brief.
2. **Statutory Scope:** The deterministic compliance auditor checks federal baseline standards: ADA 2010 Chapters 3 and 4, FGI 2022 Outpatient Facility Guidelines, and IBC Table 1017.2 egress metrics. Municipal amendments (e.g., NYC Building Code Chapter 11, California OSHPD/HCAI standards) require localized constraint rule extensions.
3. **Orthogonal 2D Bounding Boxes:** The CP-SAT formulation enforces 2D axis-aligned rectangular room intervals on a 100mm decimeter grid. Non-orthogonal boundaries (e.g., chamfered corridors or curved exterior curtain walls) must be represented via bounding box enclosures or polygon decompositions.
4. **Local Hardware Constraints:** When running quantized GGUF inference on CPU, token generation requires 70–100 seconds per candidate; GPU/Metal acceleration is recommended for real-time interactive synthesis.

---

## License

Licensed under the [MIT License](LICENSE).
