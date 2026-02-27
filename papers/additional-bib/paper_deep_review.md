# FedLoRA Additional Papers: Detailed Review Notes

This document summarizes the 15 papers extracted in `papers/additional-bib/src_extracted`.
For each paper: summary, pros, cons/limitations, future directions, experimental setting, code status, and actionable takeaways for your `base-flower` work.

## 1) FlexLoRA (2024) — arXiv:2402.11505

### Summary
FlexLoRA targets heterogeneous client resources/tasks in federated LoRA fine-tuning. Instead of forcing a single rank for all clients (bucket effect), it lets clients use larger feasible local ranks, then synthesizes/redistributes global LoRA via SVD.

### Pros
- Uses heterogeneous client capacity instead of bottlenecking to smallest rank.
- Strong empirical gains in heterogeneous client/task settings.
- The method is plug-in friendly with existing LoRA+FL pipelines.

### Cons / Limitations
- Server-side SVD/decomposition can be computationally heavy.
- Communication pattern can be heavier than simple FedAvg-style LoRA averaging.
- Authors explicitly note missing large-scale LLaMA-3 + thousands-client tests.

### Future Work / Improvements
- Validate at larger model scales and larger client counts.
- Better rank-allocation policies than simple “use maximum feasible rank”.

### Experimental Setting Snapshot
- Models: DataJuicer-1.3B; also LLaMA-3 (8B) in extended study.
- Data: Natural Instructions (task-heterogeneous); Dolly-15K (Dirichlet split).
- Setup includes large client pools and heterogeneous resource distributions.

### Code
- GitHub: https://github.com/alibaba/FederatedScope/tree/FlexLoRA

### Relevance To Your Project
- Strong baseline for heterogeneous-rank federated LoRA.
- Useful when you want to compare exactness/performance vs server compute overhead.

---

## 2) HetLoRA (2024) — arXiv:2401.06432

### Summary
HetLoRA introduces heterogeneous LoRA ranks across clients plus rank self-pruning and sparsity-weighted aggregation, focused on on-device foundation model federated fine-tuning.

### Pros
- Directly addresses system heterogeneity.
- Better communication/computation efficiency than full fine-tuning.
- Improves convergence/performance over homogeneous-rank LoRA baselines.

### Cons / Limitations
- Assumes rank distribution is independent from data distribution.
- Rank assignment is not fully optimized (open question in paper).
- May still underperform full fine-tuning in some settings.

### Future Work / Improvements
- Correlated data-resource/rank modeling.
- Theoretical convergence/generalization analysis for heterogeneous-rank LoRA.

### Experimental Setting Snapshot
- Tasks include multi-session chat and text summarization.
- Metrics include perplexity (chat) and RougeL (summarization).
- Local settings include mini-batch and local iteration grids; client sampling per round.

### Code
- No clear official repo link in extracted source.

### Relevance To Your Project
- Classic heter-rank baseline.
- Good control baseline before more advanced aggregation-bias methods.

---

## 3) FLoRA (2024) — arXiv:2409.05976

### Summary
FLoRA explicitly targets aggregation noise in FedLoRA. Core idea: stack local LoRA adapters to achieve noise-free aggregation and support heterogeneous ranks.

### Pros
- Directly addresses mathematical aggregation mismatch from averaging A and B separately.
- Supports heterogeneous ranks naturally.
- Strong gains vs FedIT and zero-padding variants in reported experiments.

### Cons / Limitations
- Server-to-client broadcast can become heavy (stacked adapters).
- Experiments mainly on LLaMA-family models.

### Future Work / Improvements
- Generalization to broader model families and larger settings.
- Further communication optimization on top of exact aggregation.

### Experimental Setting Snapshot
- Models: TinyLlama (1.1B), LLaMA (7B), LLaMA2.
- Data: Dolly, Alpaca, Wizard, ShareGPT.
- Eval: MMLU and MT-Bench.
- Typical setup: non-IID client sampling, few rounds due high LLM cost.

### Code
- GitHub: https://github.com/ATP-1010/FederatedLLM

### Relevance To Your Project
- Core baseline for the `E[AB] != E[A]E[B]` issue.
- Important reference when comparing exactness vs communication overhead.

---

## 4) FedSA-LoRA (ICLR 2025) — arXiv:2410.01463

### Summary
FedSA-LoRA argues A and B play asymmetric roles in FL: A captures more shared/general knowledge, B captures more local-specific knowledge. It aggregates only A globally and keeps B local.

### Pros
- Personalized-by-design behavior while retaining global sharing.
- Lower communication than sending full adapter state.
- Extended to rsLoRA and VeRA (generalized paradigm claim).

### Cons / Limitations
- Depends on the A/B asymmetry assumption being stable across tasks/models.
- Strong local personalization may trade off some global consistency.

### Future Work / Improvements
- Broader PEFT variants and larger-scale heterogeneous deployments.

### Experimental Setting Snapshot
- NLU: GLUE tasks (MNLI, SST2, QNLI, QQP, RTE).
- NLG: LLaMA3-8B with GSM8K and code generation settings.
- Includes sensitivity to non-IID severity, client count, and LoRA rank.

### Code
- GitHub: https://github.com/Pengxin-Guo/FedSA-LoRA

### Relevance To Your Project
- Strong baseline if you want “partial LoRA communication” (`A-only` sharing).

---

## 5) FedEx-LoRA (2024) — arXiv:2410.09432

### Summary
FedEx-LoRA adds a residual error term to correct inexact federated LoRA aggregation, aiming to recover exact/ideal updates with minimal extra system cost.

### Pros
- Targets exactness without full stacking of all local adapters.
- Simple correction mechanism; broad benchmark improvements reported.
- Strong applicability framing across tasks/models.

### Cons / Limitations
- Additional residual management/state complexity.
- Privacy-preserving/DP setting is proposed as future extension, not fully completed.

### Future Work / Improvements
- DP/federated privacy setting validation.
- Extension to ViT/VLM-type foundation models.

### Experimental Setting Snapshot
- NLU (RoBERTa + GLUE), NLG (GPT-2 + E2E NLG), plus broader reasoning evaluations.
- LoRA rank and local-epoch sweeps included.

### Code
- GitHub: https://github.com/RaghavSinghal10/fedex-lora

### Relevance To Your Project
- Good “exact aggregation with moderate overhead” baseline next to FLoRA/FlexLoRA.

---

## 6) LoRA-FAIR (2024/2025) — arXiv:2411.14961

### Summary
LoRA-FAIR focuses on two issues jointly: (1) server-side aggregation bias and (2) client-side initialization lag. It uses server correction and initialization refinement to better approximate ideal updates while preserving shared information.

### Pros
- Joint treatment of two often separated failure modes.
- Good empirical gains over multiple LoRA-FL baselines in CV settings.
- Strong practical insight: initialization consistency matters in FL LoRA.

### Cons / Limitations
- Experiments centered on vision benchmarks.
- Heterogeneous rank support not native in base method (uses extra adaptation with padding/truncation).

### Future Work / Improvements
- Extend to NLP/LLM and native heterogeneous-rank settings.

### Experimental Setting Snapshot
- Models: ViT and MLP-Mixer.
- Data: DomainNet and NICO++.
- Includes feature non-IID analysis and local-epoch sensitivity.

### Code
- No clear official implementation link in extracted source.

### Relevance To Your Project
- Useful if your runs suffer from unstable local re-initialization behavior.

---

## 7) LoRA-A2 (2024/2025) — arXiv:2410.22815

### Summary
LoRA-A2 introduces alternating freeze (between factors) plus adaptive rank selection, targeting robustness under low-rank and high heterogeneity.

### Pros
- Very strong communication reduction claims (up to ~99.8% uploaded parameters vs full FT).
- Robustness under low-rank/high heterogeneity where many baselines degrade.
- Adaptive module/rank selection behavior observed empirically.

### Cons / Limitations
- Heavy focus on classification/NLP-classification style tasks.
- Simulated heterogeneity dominates; limited real-world dataset evidence.
- Code release is described as future/planned.

### Future Work / Improvements
- Extend to generation tasks and richer real-world heterogeneous datasets.

### Experimental Setting Snapshot
- Models: RoBERTa-base/large, DistilBERT.
- Data: BANKING77, 20 Newsgroups; Dirichlet and pathological partitions.
- Includes DP variant experiments.

### Code
- No official public repo link found in extracted source.

### Relevance To Your Project
- Strong candidate for low-rank stress testing baselines.

---

## 8) AFLoRA (2025) — arXiv:2505.24773

### Summary
AFLoRA is an adaptive resource-efficient FedLoRA framework: diagonal-based dynamic rank assignment, decoupled updates, and rank-aware aggregation for heterogeneous clients.

### Pros
- Dynamic rank adaptation tied to local learning/resource states.
- Attempts to balance efficiency and performance in practical heterogeneous settings.
- Includes mechanisms beyond static heter-rank assignment.

### Cons / Limitations
- Multi-component pipeline increases tuning complexity.
- No explicit official code repository in extracted source.

### Future Work / Improvements
- Simpler adaptive controllers with fewer hyperparameters.
- Broader reproducibility and open implementation.

### Experimental Setting Snapshot
- Models: GPT-2, TinyLlama-1.1B, Qwen2.5-3B.
- Data: WizardLM, FinGPT-Sentiment, Dolly-15K, AG News (+ small Alpaca public subset).
- IID and non-IID comparisons with client-side cost reporting.

### Code
- No explicit code link found in extracted source.

### Relevance To Your Project
- Good reference if you want adaptive rank policies inside `flcore/lora/methods.py`.

---

## 9) FedRPCA (2025) — arXiv:2506.01194

### Summary
FedRPCA decomposes client LoRA updates into common low-rank + client-specific sparse components via Robust-PCA; then averages common parts and scaled-averages sparse parts.

### Pros
- Addresses client-specific knowledge dilution under vanilla FedAvg.
- Server-side method only (no change required to client training objective).
- Reported faster convergence and better final performance across vision + language.

### Cons / Limitations
- Extra server compute for RPCA (though reported moderate overhead).
- Requires scaling-factor strategy (beta scheduling) for best results.

### Future Work / Improvements
- Deeper theory and adaptive scaling rules.
- Faster/parallel RPCA implementations.
- Extend idea to non-LoRA PEFT modules.

### Experimental Setting Snapshot
- Vision: CLIP ViT-B/32 on SVHN/EuroSAT/DTD/Stanford Cars.
- Language: GPT-2 (20News), T5-base (MRQA).
- Rank and heterogeneity ablations; wall-clock and convergence analyses.

### Code
- No explicit public code link found in extracted source.

### Relevance To Your Project
- Highly relevant to “common vs client-specific signal” decomposition for aggregation bias.

---

## 10) EcoLoRA (EMNLP 2025) — arXiv:2506.02001

### Summary
EcoLoRA is communication-centric: round-robin segment sharing of LoRA, adaptive sparsification, and lossless encoding to reduce uplink bottlenecks.

### Pros
- Large communication-time and total-time reductions with limited accuracy loss.
- Robust under non-IID and varying participation/client-scale settings.
- Practical network-condition simulation included.

### Cons / Limitations
- Built and evaluated only on LoRA; generalization to other PEFT not yet validated.
- No explicit official repo link in extracted source.

### Future Work / Improvements
- Extend same communication design to adapter/prefix PEFT families.
- More standardized open-source implementation for reproducibility.

### Experimental Setting Snapshot
- LLaMA2-7B focus; QA and value-alignment tasks.
- Datasets include Dolly, Alpaca, UltraFeedback (DPO-style evaluation), ARC/MT-Bench/MMLU.
- Extensive ablations on sparsity, segments, client count, local updates, rank.

### Code
- No clear official code repo found in extracted source.

### Relevance To Your Project
- Very useful if you need to scale many experiments under limited upload bandwidth.

---

## 11) FLoRIST (2025) — arXiv:2506.09199

### Summary
FLoRIST aims for exact/accurate aggregation with lower server cost. It avoids building full dense global updates and performs efficient SVD in a compact intermediate space; two variants trade off accuracy vs communication.

### Pros
- Strong performance-efficiency balance.
- Much lower server FLOPs than full decomposition approaches (reported large margin vs FlexLoRA).
- Supports heterogeneous settings and explicit threshold-based control.

### Cons / Limitations
- Threshold choice is a major sensitivity axis.
- Still requires SVD-based machinery and per-setting threshold tuning.

### Future Work / Improvements
- Automated threshold/rank selection, including layer-wise rank policies.

### Experimental Setting Snapshot
- Models: LLaMA-3.2-1B, TinyLlama (1.1B), LLaMA-7B.
- Data: Dolly, Alpaca, Wizard.
- Eval: MMLU; homogeneous and heterogeneous client-rank settings.

### Code
- Anonymous project link appears in extracted material (temporary/anonymous hosting).

### Relevance To Your Project
- Strong candidate for next baseline beyond FLoRA when server compute is constrained.

---

## 12) Aggregation-Broadcast SP vs PS Theory (2025) — arXiv:2508.01348

### Summary
This work formalizes LoRA-FL aggregation through Aggregation-Broadcast Operators (ABO), unifying SP (Sum-Product) and PS (Product-Sum) views and deriving weak/strong convergence conditions.

### Pros
- Provides rigorous theoretical language for comparing aggregation schemes.
- Clarifies when/why SP vs PS differ in convergence sensitivity (rank/epochs).
- Useful for principled method design and theorem-driven ablations.

### Cons / Limitations
- Mostly theoretical + smaller-scale empirical validation.
- Not a plug-and-play large-LLM system contribution.

### Future Work / Improvements
- New ABO designs with improved rates and practical system constraints.
- Extension from MLP-scale experiments to modern FM/LLM federated setups.

### Experimental Setting Snapshot
- LoRA-augmented MLP on MNIST/FMNIST.
- Extreme label-skew non-IID (single-class clients).
- Systematic sweeps over LoRA rank ratio and local epochs.

### Code
- No explicit code link found in extracted source.

### Relevance To Your Project
- Best theoretical framing for your aggregation-bias research narrative.

---

## 13) ILoRA (2025) — arXiv:2511.16069

### Summary
ILoRA proposes a unified framework for three challenges: unstable initialization, rank-incompatible aggregation, and non-IID client drift. It combines QR-based initialization, concatenated QR aggregation, and rank-aware control variates.

### Pros
- End-to-end framework (init + aggregation + optimization correction).
- Strong empirical and theoretical support.
- Handles heter-rank and non-IID jointly.

### Cons / Limitations
- Algorithmic complexity is higher than simple FedAvg-style methods.
- No clear official code link in extracted source.

### Future Work / Improvements
- Extend to broader PEFT families and stronger resource constraints.

### Experimental Setting Snapshot
- CV: ViT/Swin on CIFAR-10/100 and Tiny-ImageNet.
- NLP: RoBERTa on multiple text classification benchmarks.
- Non-IID sweeps (Dirichlet alpha), homogeneous/heterogeneous rank settings.

### Code
- No explicit official public repo link found in extracted source.

### Relevance To Your Project
- A strong candidate if you need stability and heterogeneity robustness together.

---

## 14) WinFLoRA (2026) — arXiv:2602.01126

### Summary
WinFLoRA addresses privacy-heterogeneous FL by estimating client DP-noise from uploaded adapters and applying noise-aware aggregation weights as incentive and quality control.

### Pros
- Aligns client utility and global objective without external monetary mechanism.
- Handles privacy heterogeneity directly.
- Strong gains in global accuracy and client utility in reported settings.

### Cons / Limitations
- Depends on accurate enough noise estimation/ranking.
- Focused on DP-noise scenario; may need adaptation for other quality proxies.

### Future Work / Improvements
- Broader benchmark diversity and robustness to estimation/system shifts.
- Extensions to mixed quality signals beyond noise.

### Experimental Setting Snapshot
- Models: TinyLlama, GPT2-Large.
- Data: AGNews, DBpedia, 20 Newsgroups.
- Studies over client scale, non-IID levels, privacy-preference distributions.

### Code
- GitHub: https://github.com/koums24/WinFLoRA.git

### Relevance To Your Project
- Important if your next phase includes DP/noisy client updates and incentive-aware aggregation.

---

## 15) raFLoRA (2026) — arXiv:2602.13486

### Summary
raFLoRA identifies rank collapse in heterogeneous-rank FedLoRA and proposes rank-partitioned aggregation weighted by effective client contributors per rank partition.

### Pros
- Directly targets rank-collapse root cause with principled aggregation.
- Strong robustness across rank configurations and non-IID settings.
- Favorable communication-performance tradeoff vs strong baselines.

### Cons / Limitations
- Additional aggregation complexity and rank-partition bookkeeping.
- No explicit official code link in extracted source.

### Future Work / Improvements
- More automation in partitioning/weighting policies.
- Broader evaluations and reproducible reference implementations.

### Experimental Setting Snapshot
- Vision: ViT-base on CIFAR100.
- NLP: RoBERTa-base on 20 Newsgroups.
- Reasoning: LLaMA-3.x on GSM8K + Commonsense15K.
- Includes pathological and Dirichlet non-IID analyses, rank sensitivity, client participation tests.

### Code
- No explicit official public repo link found in extracted source.

### Relevance To Your Project
- One of the most aligned papers for your `E[AB]`/aggregation-bias + heter-rank direction.

---

## Cross-Paper Action Plan (Practical)

### A) Baseline Ladder (recommended order)
1. **FedAvg + plain LoRA** (your current reference).
2. **FedSA-LoRA** (`A`-only sharing baseline).
3. **FLoRA** (stacking-based exactness baseline).
4. **FedEx-LoRA / FedRPCA** (exact/corrected server aggregation with moderate overhead).
5. **raFLoRA / FLoRIST** (heter-rank robust, better efficiency-accuracy tradeoff).

### B) What to log in every run
- Aggregation error proxy between ideal merged update and implemented global update.
- Communication per round (upload/download params + bytes).
- Server FLOPs / runtime per round.
- Per-rank energy distribution over rounds (for rank-collapse diagnosis).

### C) Minimal experiments that make your paper stronger
- **Homogeneous ranks** vs **heterogeneous ranks**.
- Non-IID sweep (at least 3 alpha values).
- Rank sweep (low/medium/high).
- Ablation on local epochs and participating clients per round.
- Accuracy + convergence speed + communication + server compute.

### D) Positioning tip for your own method
Frame your contribution as balancing:
- **mathematical faithfulness** to ideal update,
- **communication efficiency**,
- **server/client compute scalability**,
- **heterogeneity robustness** (rank + data + possibly privacy).
