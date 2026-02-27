# Theory/Proof-Focused Review for Federated LoRA Papers

This file is a focused complement to `paper_deep_review.md`.
It only tracks papers with explicit theory/proof content (or strong semi-formal theory) and explains:

- what each paper claims mathematically,
- how the theory is built (assumptions -> objective -> bound/theorem -> proof idea),
- what is reusable for your own submission (especially for aggregation bias in LoRA-FL).

## 1) Scope and Selection Rule

I marked papers as theory-relevant if at least one of the following appears in source:

- explicit theorem/lemma/proposition + proof, or
- a dedicated convergence analysis section with formal assumptions and derived bound.

I did **not** treat pure empirical discussion as theory.

## 2) Quick Triage (Theory Strength)

Legend:

- `A`: formal theorem(s) + explicit proof section.
- `B`: semi-formal theorem/convergence claim; proof sketch or weaker formalism.
- `C`: light theoretical statement only (not suitable as core citation for proof-heavy section).

| Paper | Level | Main Theory Topic | Where in source |
|---|---|---|---|
| FlexLoRA (2402.11505) | A | Generalization/sample bound under SVD error | `subfile/3_method.tex`, `subfile/6_append.tex` |
| FedSA-LoRA (2410.01463) | A | A/B asymmetry lemma + nonconvex convergence theorem | `main.tex`, `appendix.tex` |
| Aggregation-Broadcast SP vs PS (2508.01348) | A | Weak/strong convergence conditions for LoRA aggregation operators | `convergence.tex` |
| WinFLoRA (2602.01126) | A | Existence of stationary Markov equilibrium (game-theoretic) | `main.tex` appendix theoretical analysis |
| raFLoRA (2602.13486) | A | Rank-collapse theorem with geometric decay | `Sections/3-problem_analysis.tex` (+ appendix refs) |
| EcoLoRA (2506.02001) | B+ | Convergence bound with compression/segmentation error term | `method.tex`, `appendix.tex` |
| FLoRA (2409.05976) | B | Theorem-style convergence statement under unbiased LoRA gradient assumption | `neurips_2024.tex` |
| LoRA-A2 (2410.22815) | B- | Proposition on parameter-space inclusion | `4_methodology.tex`, `appendix_d.tex` |
| LoRA-FAIR (2411.14961) | B-/C* | Correction-term approximation and convergence statements in supplement | `sec/X_suppl.tex` |
| ILoRA (2511.16069) | C* | Many theory claims; needs strong verification before relying | `sec/3_finalcopy.tex`, `sec/X_suppl.tex` |

`*` For LoRA-FAIR/ILoRA: theory content exists, but rigor style appears mixed (supplement/rebuttal-like in places). Use carefully and verify with final camera-ready versions before citing as primary theorem backbone.

## 3) Deep Review by Paper

## 3.1 FlexLoRA (2402.11505)

### What it claims
- Under Lipschitz assumptions and bounded SVD approximation error, global model generalization gap is bounded with high probability.
- Required sample size scales with intrinsic dimension and SVD error term.

### Core mathematical object
- Client receives rank-truncated SVD approximation:
$$
\lVert \mathrm{SVD}(W_g, r^i)-W_g\rVert \le \phi^i.
$$
- Generalization/sample condition is then written as a bound on $\tilde N$ (their theorem).

### Assumptions used
- Loss/hypothesis Lipschitz in LoRA-weight space.
- Bounded LoRA weights and bounded approximation error $\phi^i$.

### Proof construction (high-level)
1. Build function-space distance and expected-risk gap.
2. Use Lipschitz continuity to turn parameter distance into loss difference.
3. Insert bounded SVD approximation error term.
4. Derive sample complexity condition for target $\epsilon,\delta$.

### Why it matters for your topic
- It gives a direct way to include approximation terms (e.g., truncation/correction error) into a high-probability generalization statement.
- You can mirror this pattern with an aggregation-bias term for LoRA products.

### Limitation
- Bound depends on $\phi^i$; not a direct optimization dynamics proof for non-IID stochastic FL.

## 3.2 FedSA-LoRA (2410.01463)

### What it claims
- In a linearized least-squares view, optimizing with fixed $A$ vs fixed $B$ yields closed-form optima and motivates sharing only one factor.
- Under smoothness and boundedness assumptions, method converges in nonconvex FL at order $O(1/\sqrt{T})$ (as gradient norm bound).

### Core mathematical object
- LoRA update:
$$
W = W_0 + BA.
$$
- Lemma provides forms for $B^*$ and $A^*$ under one-factor freezing.

### Assumptions used
- $L$-smooth local objectives.
- Bounded stochastic gradients.
- Bounded factor norms and matrix-alignment lower bounds.

### Proof construction (high-level)
1. Express one-step local updates with periodic synchronization.
2. Apply smoothness inequality.
3. Decompose terms caused by local drift and factorization.
4. Sum over rounds and choose step size to obtain average gradient norm bound.

### Why it matters for your topic
- This is one of the cleaner templates for LoRA-specific FL convergence with explicit assumptions.
- Useful baseline when you compare full two-factor aggregation vs one-factor/shared-factor strategies.

### Limitation
- The A/B role asymmetry may depend on data/model regime; verify empirically across tasks.

## 3.3 Aggregation-Broadcast SP vs PS Theory (2508.01348)

### What it claims
- Defines general aggregation-broadcast operators $(\mathcal P,\mathcal Q)$.
- Gives:
  - weak convergence condition (local model convergence),
  - strong convergence condition (global model convergence),
  - optimality conditions linking aggregation design to convergence speed.

### Core mathematical object
- Global LoRA form from operator:
$$
W^{(t)} = W_0 + \mathcal P(\cdot)\,\mathcal Q(\cdot).
$$
- Conditions are bounds on distances between operator outputs and client factors/products.

### Assumptions used
- Standard smoothness, bounded stochastic gradients, bounded factor norms.

### Proof construction (high-level)
1. Start from descent inequality in LoRA factor space.
2. Upper-bound aggregation/broadcast mismatch by constants ($R$, or $P,Q$).
3. Convert to averaged gradient norm bounds over rounds.
4. Derive optimality equations by minimizing mismatch objective.

### Why it matters for your topic
- Very close to your “aggregation bias” thesis.
- Provides a direct language to define and optimize bias-inducing operator mismatch.

### Limitation
- Conditions may be sufficient but not necessary; practical tightness can vary.

## 3.4 raFLoRA (2602.13486)

### What it claims
- Vanilla FedLoRA can suffer rank collapse under heterogeneous client ranks.
- Energy outside shared rank decays geometrically:
$$
1-\rho_{r_1}^{(t)} \le C\gamma^t,\quad \gamma<1.
$$

### Core mathematical object
- Singular-direction energy ratio $\rho_{r_1}^{(t)}$ of shared rank.

### Assumptions used
- Fixed singular basis.
- Direction-preserving client updates.

### Proof construction (high-level)
1. Write per-direction expected update recursion.
2. Show shared directions receive full support, higher-rank directions only partial coverage.
3. Derive multiplicative attenuation for high-rank directions.
4. Convert to geometric decay in high-rank energy ratio.

### Why it matters for your topic
- Strongly aligned with heterogeneous-rank LoRA bias story.
- Gives an interpretable mechanism-level theorem, not only optimization-rate bound.

### Limitation
- Relies on simplifying assumptions (fixed basis / direction preservation).

## 3.5 WinFLoRA (2602.01126)

### What it claims
- Models client noise-choice interaction as a stochastic aggregative Markov game.
- Proves existence of a stationary Markov equilibrium (SME).

### Core mathematical object
- Long-run average utility:
$$
\bar U_i(\pi)=\lim_{T\to\infty}\frac1T\mathbb E_\pi\Big[\sum_{t=1}^T U_i^t\Big].
$$

### Assumptions used
- Polish state space, finite actions.
- Aggregative structure with bounded/continuous maps.
- Feller dynamics.
- Foster-Lyapunov drift.

### Proof construction (high-level)
1. Convert strategy profile to induced Markov kernel.
2. Establish invariant/occupation measures (positive Harris recurrence).
3. Define best-response correspondence (nonempty/convex/u.h.c.).
4. Apply Kakutani fixed-point theorem to obtain SME existence.

### Why it matters for your topic
- If your contribution includes incentive/weighting policies, this is a rigorous proof template for existence/stability arguments.

### Limitation
- Equilibrium existence does not directly imply global-optimal learning performance.

## 3.6 EcoLoRA (2506.02001)

### What it claims
- With smoothness/bounded gradients/contractive compression assumptions, obtains:
$$
\frac1T\sum_{t=0}^{T-1}\|\nabla F(P_t)\|^2 = O(T^{-1/2}).
$$
- Additional constants account for segmentation/compression errors.

### Proof construction (high-level)
1. Start from smoothness descent.
2. Decompose update into true gradient + structured error.
3. Bound inner-product and error-norm terms.
4. Sum over rounds, select $\eta=O(T^{-1/2})$.

### Why it matters
- Useful for adding communication/compression modules while keeping a standard FL convergence guarantee.

### Limitation
- Formal style is weaker than full theorem-lemma structure; assumptions/step-size conditions need careful consistency checks.

## 3.7 FLoRA (2409.05976)

### What it claims
- Stacked aggregation removes the “intermediate noise” from averaging factors directly.
- Gives theorem-style convergence bound under a key extra assumption: unbiased LoRA gradient mapping to SGD-like full-model gradient.

### Why it matters
- Closely connected to your $E[AB]\neq E[A]E[B]$ problem framing.

### Limitation
- The unbiased-LoRA-gradient assumption is strong and may be hard to justify in real nonconvex LoRA training.

## 3.8 LoRA-A2 (2410.22815)

### What it claims
- Proposition on parameter-space inclusion:
$$
\Omega_{\text{FFA-LoRA}} \subsetneq \Omega_{\text{FL+LoRA}}=\Omega_{\text{FlexLoRA}} \subset \Omega_{\text{LoRA-A}^2}.
$$

### Why it matters
- Good for expressivity/capacity argument (what update space your algorithm can represent).

### Limitation
- This is not a convergence/generalization theorem; use as supporting proposition, not main theoretical backbone.

## 3.9 LoRA-FAIR (2411.14961) and ILoRA (2511.16069) — Use with Caution

Both contain theorem-like content in extracted source, but style/placement suggests mixed maturity.

- LoRA-FAIR: main useful piece is residual correction bound for reducing approximation error from $\bar B\bar A$ to ideal $\Delta W$ in supplement.
- ILoRA: many claims spanning convergence/stability/communication; verify final published proof quality before citing as foundational theorem source.

Recommendation: cite only after manual verification against final official versions.

## 4) What This Means for Your Submission (Aggregation Bias Focus)

For your topic
$$
\mathbb E[AB] \neq \mathbb E[A]\mathbb E[B],
$$
the strongest reusable theory patterns are:

1. **Operator-mismatch framework** (SP-vs-PS paper): define aggregation operator error and tie it to convergence constants.
2. **Correction-term approximation bound** (LoRA-FAIR style): show your correction reduces product mismatch norm.
3. **Energy/rank dynamics argument** (raFLoRA): explain failure mode under heterogeneity as geometric attenuation/collapse.
4. **Standard nonconvex descent scaffold** (FedSA/EcoLoRA): convert mismatch/correction term into $O(1/\sqrt{T})$-type bound with explicit bias term.

## 5) Practical Theory Blueprint You Can Implement

Use this as a writing checklist.

## Step A: Problem setup
- Define client LoRA factors $(A_i,B_i)$ and ideal aggregate:
$$
\Delta W^\star = \sum_i p_i A_iB_i.
$$
- Define your server reconstruction $\widehat{\Delta W}$.
- Define aggregation bias:
$$
\mathcal B_t := \|\widehat{\Delta W}_t-\Delta W_t^\star\|_F^2.
$$

## Step B: Main theorem target
- Theorem form (nonconvex FL):
$$
\frac1T\sum_{t=1}^T \mathbb E\|\nabla F(W_t)\|^2
\le O(T^{-1/2}) + c_1\cdot \frac1T\sum_{t=1}^T \mathbb E[\mathcal B_t].
$$
- If you add correction term, show:
$$
\mathbb E[\mathcal B_t^{\text{corr}}] \le \kappa\cdot \mathbb E[\mathcal B_t^{\text{base}}],\quad \kappa<1.
$$

## Step C: Proof skeleton
1. Smoothness descent on global objective.
2. Decompose update into unbiased descent + drift + aggregation bias.
3. Bound drift using standard FL heterogeneity assumptions.
4. Bound aggregation part using your operator/correction lemma.
5. Telescope sum; choose step size.

## Step D: Minimum empirical alignment
- Plot $\mathcal B_t$ over rounds (directly).
- Report correlation between $\mathcal B_t$ and accuracy drop.
- Ablate correction strength and show monotonic impact on $\mathcal B_t$ and convergence.

## 6) Suggested Citation Priority (for theory section)

If your section is theorem-heavy, prioritize:

1. `2508.01348` (operator-level convergence conditions),
2. `2410.01463` (LoRA-specific FL convergence template),
3. `2602.13486` (rank-collapse mechanism theorem),
4. `2402.11505` (generalization/sample complexity with approximation term),
5. `2506.02001` (compression-aware convergence style).

Use `2411.14961` and `2511.16069` selectively after final-version verification.

## 7) Final Advice for A*-style Theory Section

- Make one theorem the centerpiece, not many weak theorems.
- Keep assumptions minimal and interpretable; map each to an experiment.
- Expose one explicit “failure mode” theorem (bias/collapse) and one “improvement” theorem (your correction).
- Include one finite-sample/non-asymptotic bound that carries your method-specific term.
- Add a short proof sketch in main paper; move full proof to appendix with clean lemma dependency graph.

