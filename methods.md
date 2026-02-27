# Methods Guide: FedAvg, LoRA (Plain/Diag), and FedMUD DMU

This file explains three things in one place:
- how `FedAvg` is computed in this project,
- how `LoRALinear` and `LoRAConv2d` are computed in `base-flower` (with concrete numeric examples),
- how the methods in `/Users/mitu/Desktop/data/math/fedmud-icml25` compute updates, initialize factors, and map tensor shapes.

## 1) Scope and code locations

`base-flower`:
- `flcore/server_app.py` (FedAvg and other FL strategies)
- `flcore/lora/modules.py` (`LoRALinear`, `LoRAConv2d`)
- `flcore/lora/methods.py` (`plain`, `diag` composition)

`fedmud-icml25`:
- `decompose/dmu.py` (DMU decomposition and update logic)
- `decompose/models.py` (module replacement and skip layers)
- `train/client.py`, `train/server.py` (when/why `push_reset_update` is called)

## 2) FedAvg in this codebase

At round $t$, selected clients $S_t$ train locally and return updated weights. The server computes weighted average by sample count:

$$
W^{t+1} = \sum_{k \in S_t}\frac{n_k}{\sum_{j \in S_t} n_j} W_k^{t+1}
$$

where $n_k$ is the number of samples used by client $k$.

### Practical behavior in this repository
- Strategy name is configured by `strategy-name`.
- For plain federated baseline, use `strategy-name="fedavg"`.
- Client-side returns model arrays and metrics.
- Server aggregates arrays and evaluates global model on centralized test set.

### Pros
- Very simple baseline.
- Reproducible and easy to compare across methods.
- Strong default starting point.

### Cons
- Sensitive to non-IID client drift.
- Can degrade with large local epochs or too-large LR.

## 3) LoRA in `base-flower`

Both LoRA wrappers use the same high-level idea:

$$
W_{\text{effective}} = W_{\text{base}} + \Delta W
$$

$$
\Delta W = \text{compose}(A, B)\cdot\frac{\alpha}{r}
$$

where:
- $r$ is LoRA rank,
- $\alpha$ is LoRA scaling,
- `compose` is `plain` or `diag`.

Forward structure:

$$
y = f(x;W_{\text{base}}) + f(\text{dropout}(x);\Delta W)
$$

Note: dropout is applied only on the LoRA branch input.

### 3.1) `LoRALinear`: shape and formula

For a base linear layer with input dim $d_{in}$ and output dim $d_{out}$:
- $A \in \mathbb{R}^{d_{out}\times r}$,
- $B \in \mathbb{R}^{r\times d_{in}}$,
- $\Delta W \in \mathbb{R}^{d_{out}\times d_{in}}$.

`plain` method:

$$
\text{compose}(A,B)=AB
$$

`diag` method with learnable $s \in \mathbb{R}^{r}$:

$$
\text{compose}(A,B)=(A\odot s)B
$$

where $s$ is broadcast along the row dimension of $A$.

#### Initialization in code
- `lora_a` is Kaiming-uniform initialized.
- `lora_b` is initialized to zeros.
- So initial $\Delta W$ is near zero (exactly zero when `lora_b` is all zeros).

#### Worked numeric example (`plain`)

Choose:
- $d_{in}=3$, $d_{out}=2$, $r=1$, $\alpha=2$ so $\alpha/r=2$.
- $A=\begin{bmatrix}2\\-1\end{bmatrix}$,
- $B=\begin{bmatrix}0.5 & 1.0 & -0.5\end{bmatrix}$.

Then:

$$
AB=
\begin{bmatrix}
1 & 2 & -1\\
-0.5 & -1 & 0.5
\end{bmatrix}
$$

$$
\Delta W = 2(AB)=
\begin{bmatrix}
2 & 4 & -2\\
-1 & -2 & 1
\end{bmatrix}
$$

For input $x=\begin{bmatrix}1 & 2 & -1\end{bmatrix}$, the LoRA branch contribution is:

$$
x\Delta W^\top = \begin{bmatrix}12 & -6\end{bmatrix}
$$

and final output is base output plus this LoRA contribution.

### 3.2) `LoRAConv2d`: shape and formula

For base conv weights $W_{\text{base}} \in \mathbb{R}^{C_{out}\times C_{in}\times k_h\times k_w}$:

$$
d_{flat}=C_{in}k_hk_w
$$

- $A \in \mathbb{R}^{C_{out}\times r}$,
- $B \in \mathbb{R}^{r\times d_{flat}}$,
- compose result is reshaped to
  $\Delta W \in \mathbb{R}^{C_{out}\times C_{in}\times k_h\times k_w}$.

Forward:

$$
y = \text{Conv}(x,W_{\text{base}},b)+\text{Conv}(\text{dropout}(x),\Delta W,0)
$$

#### Worked numeric example (`plain`, simple single-channel)

Let:
- $C_{in}=1$, $C_{out}=1$, $k_h=k_w=2$, $r=1$, $\alpha=1$.
- $A=[2]$,
- $B=[0.5,-0.5,1.0,0.0]$.

Compose:

$$
AB=[1,-1,2,0]
$$

Reshape into conv kernel:

$$
\Delta W=
\begin{bmatrix}
\begin{bmatrix}
1 & -1\\
2 & 0
\end{bmatrix}
\end{bmatrix}
$$

Use base kernel:

$$
W_{\text{base}}=
\begin{bmatrix}
\begin{bmatrix}
1 & 0\\
0 & -1
\end{bmatrix}
\end{bmatrix}
$$

Input:

$$
x=
\begin{bmatrix}
1 & 2 & 0\\
0 & 1 & 3\\
2 & 1 & 0
\end{bmatrix}
$$

For valid stride-1 convolution:
- base output:

$$
\begin{bmatrix}
0 & -1\\
-1 & 1
\end{bmatrix}
$$

- LoRA output:

$$
\begin{bmatrix}
-1 & 4\\
3 & 0
\end{bmatrix}
$$

- final output (sum):

$$
\begin{bmatrix}
-1 & 3\\
2 & 1
\end{bmatrix}
$$

### 3.3) `plain` vs `diag` (LoRA method variants)

`plain`:
- $\Delta W = AB\cdot\alpha/r$
- Pros: simplest, fastest baseline, minimal extra parameters.
- Cons: less expressive at fixed rank.

`diag`:
- $\Delta W = (A\odot s)B\cdot\alpha/r$, with learnable $s\in\mathbb{R}^r$.
- Pros: rank-wise scaling can improve flexibility with only $+r$ parameters per adapted layer.
- Cons: slightly more optimization complexity.

## 4) FedMUD DMU methods (`fedmud-icml25/decompose/dmu.py`)

FedMUD DMU uses a frozen base weight plus an update module, similar in spirit to additive adapters:

$$
W_{\text{effective}} = W_{\text{base}} + \Delta W
$$

but the update families and initialization logic are different from LoRA.

### 4.1) `DMU_Linear`

Base:
- `weight` shape $(o,i)$, non-trainable.

Update family selected by `dmu_type`:
- `mat`: uses `Mat_Update(i,o,config)`
- `kron`: uses `Kron_Update(i,o,config)`

Forward:

$$
y = x(W_{\text{base}} + \Delta W)^\top
$$

### 4.2) `DMU_Conv2d`

Base conv weight:
- shape $(C_{out}, C_{in}, k, k)$, non-trainable.

Update backend is built on matrix shape:

$$
i_{shape} = C_{in}\cdot k,
\quad
o_{shape} = C_{out}\cdot k
$$

So matrix update has shape $(o_{shape},i_{shape})$ and element count:

$$
(o_{shape}\cdot i_{shape})=(C_{out}k)(C_{in}k)=C_{out}C_{in}k^2
$$

which exactly matches the number of conv-weight elements.

Then FedMUD reshapes matrix update to:

$$
\Delta W_{conv}\in\mathbb{R}^{C_{out}\times C_{in}\times k\times k}
$$

and uses standard `conv2d` with $W_{\text{base}} + \Delta W_{conv}$.

### 4.3) `mat` update family: `Mat_Weight` and `Mat_Update`

For target update matrix $\Delta W \in \mathbb{R}^{o\times i}$:
- left factor $L \in \mathbb{R}^{o\times r}$,
- right factor $R \in \mathbb{R}^{r\times i}$,
- update:

$$
\Delta W = LR
$$

Rank formula in code (`ratio = \rho`):

$$
\text{num\_weights}=oi
$$

If pattern is `FAB`:

$$
r = \left\lceil\frac{oi\rho}{i}\right\rceil
$$

Else (`AB` or `FAB+CFD`):

$$
r = \left\lceil\frac{oi\rho}{o+i}\right\rceil
$$

and always $r\ge 1$.

Pattern behavior:
- `AB`: both left and right are trainable.
- `FAB`: only right is trainable, left is fixed random after init.
- `FAB+CFD`: two branches are summed:
  - branch 1: right-trainable,
  - branch 2: left-trainable,
  - total update is branch1 + branch2.

#### Worked DMU linear shape/parameter example (`mat`)

Assume:
- input size $i=256$,
- output size $o=128$,
- ratio $\rho=0.1$.

For `AB`:
$$
r=\left\lceil\frac{oi\rho}{o+i}\right\rceil
=\left\lceil\frac{128\cdot 256\cdot 0.1}{128+256}\right\rceil
=\left\lceil 8.53\right\rceil=9
$$

Shapes:
- $L:(128,9)$,
- $R:(9,256)$,
- $\\Delta W:(128,256)$.

Trainable parameter count:
$$
128\cdot 9 + 9\cdot 256 = 3456
$$

For `FAB` (right trainable only), trainable count is:
$$
9\cdot 256=2304
$$
(left side is fixed random after init).

For `FAB+CFD`, branch 1 trains right and branch 2 trains left, so trainable count becomes:
$$
2304 + 1152 = 3456
$$
same total as `AB`, but split across two one-sided branches.

### 4.4) `kron` update family: `Kron_Weight` and `Kron_Update`

Each branch keeps two 3D tensors:
- $L\in\mathbb{R}^{r\times s\times s}$,
- $R\in\mathbb{R}^{r\times s\times s}$.

`kron_3d` computes (per rank index) a Kronecker-like expansion via `einsum` and reshapes to:

$$
\text{kron\_3d}(L,R)\in\mathbb{R}^{r\times s^2\times s^2}
$$

Then FedMUD flattens, truncates to needed count $oi$, and reshapes back to $(o,i)$.

Rank/size selection:

$$
\text{num\_weights}=oi
$$

If pattern is `FAB`, code first uses $\rho \leftarrow 2\rho$.

$$
r = \left\lceil\frac{\rho^2\cdot oi}{4}\right\rceil,
\quad
s = \left\lceil\left(\frac{oi}{r}\right)^{1/4}\right\rceil
$$

Then:

$$
\Delta W = \text{reshape}(\text{flatten}(\text{kron output})[:oi],(o,i))
$$

### 4.5) FedMUD initialization: why update starts at zero

FedMUD parses `init_type_mag` such as:
- `nor_0.1` / `nor-0.1`,
- `uni_0.1` / `uni-0.1`.

For both `Mat_Weight` and `Kron_Weight`:
- if `right` is trainable:
  - initialize left with random values (normal or uniform),
  - initialize right as zeros;
- if `left` is trainable:
  - initialize right random,
  - initialize left zeros.

So initial multiplicative update is exactly zero:

$$
\Delta W_0 = 0
$$

For `FAB+CFD`, two branches are initialized with different seeds (`seed` and `seed+42`) to reduce branch correlation.

### 4.6) Push-reset in FedMUD (`dmu_push_reset_update`)

`dmu_push_reset_update` is called on clients when receiving server config.

For each DMU layer:
- if `push=True`:
  1) merge current update into base,
     $$W_{\text{base}}\leftarrow W_{\text{base}}+\Delta W$$
  2) reinitialize update factors (`init_parameters(seed)`).
- if `push=False`: do not merge; keep using current update factors.

Server sets `push` periodically via `dmu_interval`.

### 4.7) Concrete DMU shape example (conv)

Suppose a conv layer has:
- $C_{in}=16$, $C_{out}=32$, $k=3$.

Then:

$$
i_{shape}=16\cdot 3=48,
\quad
o_{shape}=32\cdot 3=96
$$

Total update entries:

$$
96\cdot 48 = 4608 = 32\cdot 16\cdot 3\cdot 3
$$

For `mat` with $\rho=0.1$ and pattern `AB`:

$$
r = \left\lceil\frac{4608\cdot 0.1}{96+48}\right\rceil
= \left\lceil 3.2\right\rceil = 4
$$

Shapes:
- $L: (96,4)$,
- $R: (4,48)$,
- $\Delta W = LR$ then reshape to $(32,16,3,3)$.

## 5) Pros and cons summary

### FedAvg
Pros:
- simple and robust baseline.

Cons:
- non-IID drift and local-training sensitivity.

### LoRA `plain`
Pros:
- simplest low-rank baseline, low overhead.

Cons:
- may underfit when rank is small.

### LoRA `diag`
Pros:
- extra rank-wise control with tiny parameter increase.

Cons:
- slightly harder optimization.

### FedMUD `mat`
Pros:
- explicit low-rank matrix factorization with pattern control (`AB`, `FAB`, `FAB+CFD`).

Cons:
- behavior depends strongly on pattern and push schedule.

### FedMUD `kron`
Pros:
- structured update that can capture repeated/block interactions.

Cons:
- rank/size heuristics and flatten-truncate step can make behavior less intuitive.

## 6) References

- FedAvg: McMahan et al., 2017  
  https://proceedings.mlr.press/v54/mcmahan17a.html

- LoRA: Hu et al., 2021  
  https://arxiv.org/abs/2106.09685

- Non-IID client drift motivation: SCAFFOLD, Karimireddy et al., 2020  
  https://proceedings.mlr.press/v119/karimireddy20a.html

## 7) Deep related works: aggregation bias in FedLoRA (updated: 2026-02-27)

This section focuses on the core mismatch you called out:
- ideal aggregation is average of products,
- most practical FedLoRA pipelines compute product of averages.

### 7.1) Mathematical formulation of the bias

Let client $i$ produce a low-rank update:

$$
\Delta W_i = A_i B_i
$$

with client weight $p_i \ge 0$, $\sum_i p_i = 1$.

Two common aggregation forms are:

$$
\Delta W_{\text{SP}} = \sum_{i=1}^{N} p_i A_i B_i
$$

$$
\Delta W_{\text{PS}} = \left(\sum_{i=1}^{N} p_i A_i\right)\left(\sum_{i=1}^{N} p_i B_i\right) = \bar A \bar B
$$

The aggregation bias is:

$$
\mathcal E_{\text{agg}} = \Delta W_{\text{SP}} - \Delta W_{\text{PS}}
$$

Equivalent symmetric decomposition:

$$
\mathcal E_{\text{agg}}
= \frac{1}{2}\sum_{i=1}^{N}\sum_{j=1}^{N} p_i p_j (A_i - A_j)(B_i - B_j)
$$

This makes the non-IID effect explicit: if client factors are diverse and correlated, the cross terms grow.

Practical normalized metric per layer $\ell$:

$$
\text{bias\_ratio}_\ell
=
\frac{\left\|
\sum_i p_i A_i^{(\ell)}B_i^{(\ell)}
-
\left(\sum_i p_i A_i^{(\ell)}\right)\left(\sum_i p_i B_i^{(\ell)}\right)
\right\|_F}
{\left\|\sum_i p_i A_i^{(\ell)}B_i^{(\ell)}\right\|_F + \varepsilon}
$$

### 7.2) Taxonomy of methods relevant to this bias

Note:
- For several very recent papers (especially 2025-2026), categorization below is partly inferred from abstracts when full derivations are not fully exposed in the abstract page.
- Many are arXiv preprints, so peer-review status may change.

#### A) Exact or near-exact product aggregation (reduce $E[AB] \neq E[A]E[B]$ directly)

1. FedEx-LoRA (2024):  
   https://arxiv.org/abs/2410.09432  
   - Adds a residual error term on server-side frozen weights to correct inexact FedAvg-style LoRA merging.
   - Strength: explicitly targets the inexact aggregation issue.
   - Weakness: extra state/tracking overhead at server.

2. FLoRIST (2025):  
   https://arxiv.org/abs/2506.09199  
   - Uses stacked-adapter SVD/thresholding ideas to approach mathematically accurate aggregation without forming dense global updates.
   - Strength: better balance of accuracy and communication than naive stacking/full reconstruction.
   - Weakness: still needs decomposition logic and threshold tuning.

3. FLoRA (2024):  
   https://arxiv.org/abs/2409.05976  
   - Handles heterogeneous ranks and client capabilities with low-rank adapter coordination.
   - Strength: heterogeneity-aware federated LoRA.
   - Weakness: global synthesis/decomposition can still be costly at scale.

#### B) Correction/refinement around PS aggregation

4. LoRA-FAIR (2024/2025):  
   https://arxiv.org/abs/2411.14961  
   - Explicitly names two problems: server-side aggregation bias and client-side initialization lag.
   - Introduces correction/refinement strategy on top of federated LoRA.
   - Strength: directly framed around your target problem.
   - Weakness: correction overhead and assumptions can be fragile in high heterogeneity.

5. FedRPCA (2025):  
   https://arxiv.org/abs/2506.01194  
   - Decomposes local LoRA updates into common low-rank + sparse parts via Robust PCA, then aggregates differently.
   - Strength: separates shared vs client-specific information; often more robust to heterogeneity/noise.
   - Weakness: decomposition cost and hyperparameter sensitivity.

#### C) Selective sharing/personalization (avoid averaging both factors equally)

6. FedSA-LoRA (ICLR 2025):  
   https://arxiv.org/abs/2410.01463  
   https://proceedings.iclr.cc/paper_files/paper/2025/hash/f53a37f820d5be5930415d964f4a0187-Abstract-Conference.html  
   - Shares only one factor (A) and keeps the other (B) local.
   - Strength: lower upload cost, personalization-friendly, avoids full PS mismatch.
   - Weakness: less globally consistent update; may under-transfer global knowledge.

7. FedALT (AAAI 2026 in arXiv comments):  
   https://arxiv.org/abs/2503.11880  
   - Personalized training with Rest-of-World LoRA and adaptive mixing, instead of pure FedAvg-style global adapter.
   - Strength: mitigates cross-client interference.
   - Weakness: more personalized pipeline complexity.

#### D) Heterogeneous-rank and resource-aware aggregation

8. HetLoRA (2024):  
   https://arxiv.org/abs/2401.06432  
   - Early strong baseline for heterogeneous client resources/ranks in on-device FM fine-tuning.

9. FlexLoRA (2024):  
   https://arxiv.org/abs/2402.11505  
   - Synthesizes full-size LoRA and redistributes via SVD under heterogeneous tasks/resources.

10. LoRA-A2 (2024/2025):  
    https://arxiv.org/abs/2410.22815  
    - Alternating freeze + adaptive rank selection for robustness at low rank under heterogeneity.

11. AFLoRA (2025):  
    https://arxiv.org/abs/2505.24773  
    - Decouples shared/client-specific updates and uses rank-aware aggregation with refinement.

12. ILoRA (2025):  
    https://arxiv.org/abs/2511.16069  
    - Targets initialization misalignment, rank incompatibility, and drift with QR-based and rank-aware aggregation.

13. raFLoRA (2026):  
    https://arxiv.org/abs/2602.13486  
    - Highlights rank collapse under heterogeneous ranks; partitioned rank-aware aggregation.

14. WinFLoRA (2026):  
    https://arxiv.org/abs/2602.01126  
    - Client-adaptive aggregation under privacy heterogeneity and incentives.

#### E) Communication-first FedLoRA methods (indirectly impact bias handling choices)

15. EcoLoRA (EMNLP 2025):  
    https://arxiv.org/abs/2506.02001  
    - Segment sharing, sparsification, and encoding for communication reduction.
    - Important baseline because communication constraints often force approximate aggregation.

#### F) Theory focused on SP vs PS

16. Convergence Analysis of Aggregation-Broadcast in LoRA-enabled Distributed Fine-Tuning (2025):  
    https://arxiv.org/abs/2508.01348  
    - Defines SP (sum-product) vs PS (product-sum) and aggregation-broadcast operator.
    - Useful for theoretical framing and convergence analysis language.

### 7.3) What this means for your stated limitation (Vision vs LLM, high non-IID)

In many easier vision regimes:
- low-rank factor subspaces across clients are closer,
- $\mathcal E_{\text{agg}}$ stays moderate,
- linear correction can be enough.

In high non-IID LLM regimes:
- subspace mismatch and rank heterogeneity increase,
- cross terms in $\mathcal E_{\text{agg}}$ can dominate,
- correction approximations break more often,
- and communication-efficient approximations can further amplify mismatch.

This is exactly why recent papers increasingly combine:
- rank-aware aggregation,
- subspace alignment/QR/SVD,
- personalized or selective sharing,
- plus correction terms.

## 8) Step-by-step research plan toward A* conference quality and large models

Target: produce a publishable contribution on aggregation bias in FedLoRA, with convincing evidence from controlled math-to-systems progression.

### Phase 0 (1-2 weeks): solid experimental harness in your codebase

1. Implement explicit aggregation modes in `flcore/lora/methods.py` and server aggregation path:
   - `ps_avg`: aggregate $\bar A,\bar B$ then multiply.
   - `sp_exact`: aggregate $\sum p_i A_i B_i$ in full matrix space (small models first).
   - `sp_svd_r`: aggregate in full then compress back to rank $r$ via SVD.
   - `share_a_only` or `share_b_only`.
   - `ps_plus_corr`: PS plus correction term module.

2. Add instrumentation metrics each round:
   - layer-wise `bias_ratio`.
   - cosine similarity between client subspaces.
   - per-client rank-energy contribution.

3. Log to W&B with strict monotonic step:
   - `server/bias_ratio_mean`,
   - `server/bias_ratio_max`,
   - `server/agg_mode`,
   - `server/comm_bytes_up/down`.

### Phase 1 (2-4 weeks): controlled synthetic verification

1. Use linear toy setup where $A_i, B_i$ are generated with controllable covariance.
2. Validate empirical scaling:

$$
\|\mathcal E_{\text{agg}}\|_F
\uparrow
\quad \text{as heterogeneity/correlation increases}
$$

3. Show when PS is close enough and when it fails.
4. Use this to motivate your method mathematically, not only empirically.

### Phase 2 (4-8 weeks): vision benchmarks (cheap, fast iteration)

1. Datasets: CIFAR-10/100, Tiny-ImageNet, GTSRB.
2. Non-IID axes:
   - label skew (Dirichlet $\alpha$),
   - quantity skew,
   - feature/domain skew (if available).
3. Compare:
   - FedAvg full,
   - PS LoRA baseline,
   - SP exact/SVD,
   - your correction method.
4. Report:
   - accuracy vs rounds,
   - communication cost,
   - compute overhead,
   - bias metrics correlation with final accuracy.

### Phase 3 (6-10 weeks): medium LLM scale

1. Models: 0.5B-3B then 7B (LoRA/QLoRA).
2. Tasks:
   - instruction tuning (heterogeneous datasets),
   - classification and generation mix.
3. Focus results:
   - where vision-style approximation fails,
   - where your correction/rank-aware method keeps stability.

### Phase 4 (8-12 weeks): large-model validation

1. Move to 7B-13B+ with realistic client heterogeneity:
   - variable rank budgets,
   - variable local epochs,
   - stragglers/partial participation.
2. Add memory-safe estimators for bias (avoid dense matrices):
   - random projection/Hutchinson estimators:

$$
\|M\|_F^2 = \mathbb E_{z}\|Mz\|_2^2
$$

3. Demonstrate method quality under strict communication budgets.

### Phase 5 (paper shaping): make it A* ready

1. Required claims:
   - precise bias formalization,
   - theoretical proposition/bound,
   - method that improves bias-accuracy-efficiency tradeoff,
   - robust large-model evidence.

2. Minimal ablations expected by top venues:
   - no correction vs correction,
   - correction rank/temperature/sparsity,
   - non-IID severity sweep,
   - homogeneous vs heterogeneous ranks,
   - failure cases.

3. Reproducibility package:
   - configs, seeds, scripts, exact commit hash,
   - cost accounting (GPU hours, peak memory, network bytes),
   - deterministic eval pipeline.

### Phase 6 (execution order for your current repository)

1. Start with your existing `base-flower` LoRA plain/diag setup.
2. Add `aggregation-mode` config key.
3. Implement `ps_avg` vs `sp_svd_r` first (most informative baseline pair).
4. Add `ps_plus_corr` as your first novel method.
5. Validate on CIFAR100 non-IID.
6. Move to Tiny-ImageNet and then LLM.

## 9) Additional bibliography (FedLoRA-focused)

- FlexLoRA (2024): https://arxiv.org/abs/2402.11505  
- HetLoRA (2024): https://arxiv.org/abs/2401.06432  
- FLoRA (2024): https://arxiv.org/abs/2409.05976  
- FedSA-LoRA (ICLR 2025): https://arxiv.org/abs/2410.01463  
- FedEx-LoRA (2024): https://arxiv.org/abs/2410.09432  
- LoRA-FAIR (2024/2025): https://arxiv.org/abs/2411.14961  
- LoRA-A2 (2024/2025): https://arxiv.org/abs/2410.22815  
- AFLoRA (2025): https://arxiv.org/abs/2505.24773  
- FedRPCA (2025): https://arxiv.org/abs/2506.01194  
- EcoLoRA (EMNLP 2025): https://arxiv.org/abs/2506.02001  
- FLoRIST (2025): https://arxiv.org/abs/2506.09199  
- Aggregation-Broadcast SP vs PS theory (2025): https://arxiv.org/abs/2508.01348  
- ILoRA (2025): https://arxiv.org/abs/2511.16069  
- WinFLoRA (2026): https://arxiv.org/abs/2602.01126  
- raFLoRA (2026): https://arxiv.org/abs/2602.13486  
- FedLoRA survey (IJCAI 2025): https://www.ijcai.org/proceedings/2025/1196
