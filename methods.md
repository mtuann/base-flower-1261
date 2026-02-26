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
