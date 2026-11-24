# AION Plasticity Core (APC) — Mathematical Specification

**Status:** Permanent specification (owner canon, 2026-11-24: *the plastic neuron mathematics is its own permanent research discipline — mathematical specification, reference implementation, numerical tests and versioned benchmarks*).
**Reference implementation:** `src/nnfs/connections/plastic.py` (frozen: `freeze/aion_learning_baseline_v2_20261125.tar.gz`, SHA-256 `ea802b9603aaf4a9e91816a8d8b59ddabeedb30a2ca36906762084d9f38734af`)
**Numerical suite:** `tests/test_connections_learning.py` · **Gate runner:** `scripts/diagnostic_learning.py`

---

## 1. Identity

The plastic neuron is the heart of AION — the part that makes it an organism that develops, rather than a model that runs.

Three lines hold the whole design together:

**The brain is replaceable.**
**The teacher is replaceable.**
**The ability to learn is fundamental.**

And the defining property of the organism follows from the third line: AION is not measured by how much it knows at birth. It is measured by its ability to change through experience. A substrate that cannot change is a corpse with good posture. A substrate that can change is an organism, whatever its starting knowledge.

APC is that ability, written down. Everything in this document serves it: a forward map that carries experience into transmitted signal, a learning law that follows the actual computation, a consolidation step that turns fast learning into persistent knowledge without disturbing what is already transmitted, five protected properties that must survive every future change, and a learning contract that any host substrate must pass before a claim may be made on it.

---

## 2. The forward map

State, per connection $ij$ (pre neuron $i$, post neuron $j$), at context time $t$:

- $u_{ij}^{\text{fast}}$ — the **fast state**: rapidly adapting, error-driven, the substrate of learning. (`plastic.py` `u_fast`, line 303; `requires_grad=True`.)
- $\theta_{ij}^{\text{slow}}$ — the **slow state**: consolidated knowledge, written only by consolidation. (`theta_slow`, line 306; `requires_grad=False`.)
- $\psi_{ij}(t) = \psi(h_i, h_j, c_t)$ — the **relationship function**: a learned modulation factor per pre/post pair, optionally gated by a context signal $c_t$. (`RelationshipFunction.forward`, lines 189–261.)

**Effective connection strength** (dual-timescale blend, `get_combined_weight`, lines 504–508):

$$
a_{ij} \;=\; b\cdot u_{ij}^{\text{fast}} \;+\; (1-b)\cdot \theta_{ij}^{\text{slow}}
$$

with the canonical blend $b = 0.7$, i.e. $a_{ij} = 0.7\,u_{ij}^{\text{fast}} + 0.3\,\theta_{ij}^{\text{slow}}$. The blend is a live parameter: the frozen pilot calibration runs $b = 1.0$ and anneals it toward its floor during consolidation (§4); the unit-test suite uses the canonical $b = 0.7$.

**Forward map** (`PlasticRelationalConnection.forward`, lines 743–749, 822):

$$
m_{ij}(t) \;=\; a_{ij}\cdot \psi_{ij}(h_i, h_j, c_t)
\qquad\qquad
\hat y_j(t) \;=\; \sum_i m_{ij}(t) \;=\; \sum_i a_{ij}\,\psi_{ij}
$$

The predicted next state of post neuron $j$ is the sum of transmitted signals over its incoming connections. In code: `weight = get_combined_weight(blend)` → `psi = relationship(pre, post, context)` → `transmitted = weight.unsqueeze(0) * psi` → `predicted = transmitted.sum(dim=-1)` (line 765). The forward output used for the error **is** the actual forward output — this is a protected property (§5.1), not a convention.

**Loss** — mean-squared on the actual output:

$$
L \;=\; \tfrac12\,\big(y_j - \hat y_j\big)^2
\qquad\qquad
L_B \;=\; \tfrac12\,\operatorname{mean}_{B}\big(y_j - \hat y_j\big)^2
$$

where $B$ is the batch (the implementation trains against $L_B$; `plasticity_step`, lines 341–344).

**Relationship families** (`RelationshipFunction`, lines 156–188): `bilinear` ($\psi_{ij} = h_j^{\top} W\, h_i \cdot \text{gate}(c_t)$), `post` ($\psi_{ij} = h_j \cdot W_{ij}$, constant along pre), `mlp`, `attention`. Context enters through a sigmoid gate `torch.sigmoid(context_gate(c_t))` (lines 227–229, 236–238). **The frozen calibration point uses `relation_type="post"`** — the bilinear pre-factor zeroes the expected error-driven gradient under zero-mean states (gradient starvation; `freeze/AION_LEARNING_BASELINE_20261124.md`, "Calibration settings"; rationale in `plastic.py` lines 164–170).

---

## 3. The learning law

$$
\boxed{\;u_{ij}^{\text{fast}} \;\leftarrow\; u_{ij}^{\text{fast}} \;+\; 0.7\,\eta\,(y_j - \hat y_j)\,\psi_{ij}\;}
$$

in canonical form; implemented as (`plasticity_step`, lines 319–369)

$$
u^{\text{fast}} \leftarrow u^{\text{fast}} + b\,\eta\,\operatorname{mean}_{B}\!\big[(y - \hat y)\,\psi\big]
$$

with frozen $\eta = 0.05$, followed by weight decay $u \leftarrow (1-\lambda)u$ ($\lambda = 10^{-5}$) and the safety clamp $u \leftarrow \mathrm{clip}(u, -10, +10)$ (lines 361–362). The update touches only connections that participate in the forward pass this step: active mask $=\mathbb{1}\!\left[\operatorname{mean}_B|\psi_{ij}| > 10^{-4}\right]$ (line 791).

### 3.1 Derivation sketch

The rule is exact gradient descent on $L_B$ with respect to $u^{\text{fast}}$, not an approximation of it.

1. **Error derivative of the loss.** For $L = \tfrac12 (y_j - \hat y_j)^2$ and $\hat y_j = \sum_i a_{ij}\psi_{ij}$:

$$
\frac{\partial L}{\partial a_{ij}} \;=\; -(y_j - \hat y_j)\,\psi_{ij}
$$

2. **Chain through the blend.** From $a_{ij} = b\,u_{ij}^{\text{fast}} + (1-b)\,\theta_{ij}^{\text{slow}}$ with $\theta$ held fixed:

$$
\frac{\partial a_{ij}}{\partial u_{ij}^{\text{fast}}} = b = 0.7
\qquad\Rightarrow\qquad
\frac{\partial L}{\partial u_{ij}^{\text{fast}}} \;=\; -\,0.7\,(y_j - \hat y_j)\,\psi_{ij}
$$

3. **Gradient step, sign-flipped into a descent update.**

$$
u_{ij}^{\text{fast}} \leftarrow u_{ij}^{\text{fast}} - \eta\,\frac{\partial L}{\partial u_{ij}^{\text{fast}}}
= u_{ij}^{\text{fast}} + 0.7\,\eta\,(y_j - \hat y_j)\,\psi_{ij}
$$

The $0.7$ is not a heuristic damping factor bolted onto a rule: it is $\partial a / \partial u^{\text{fast}}$, and it enters the update **exactly once** — as the explicit Jacobian factor of the blend (`plastic.py` lines 330–335 and 357–358: "the blend factor (0.7) enters the gradient explicitly, exactly once"). A batch mean replaces the per-sample sum because the implementation minimises $L_B$; `mean_B[(y-\hat y)\psi]` *is* $\partial L_B/\partial a$.

### 3.2 The scope condition — first-class mathematics

The chain in §3.1 used one assumption, and it must be stated as part of the law, not as a footnote:

$$
\frac{\partial \hat y_j}{\partial u_{ij}^{\text{fast}}} = b\,\psi_{ij}
\quad\text{holds}\quad
\textbf{while } \psi_{ij} \text{ is independent of the fast state.}
$$

In the current design this holds: $\psi$ is a function of $(h_i, h_j, c_t)$ and the relationship weights $W$ — never of $u^{\text{fast}}$ (§2). The gradient with respect to $W$ is a *different* parameter's gradient and belongs to $W$'s own training path (autograd; the frozen calibration leaves $W$ at its init gain $g = 1.0$).

**If a future relationship function depends on trainable plastic state** — $\psi = \psi(h_i, h_j, c_t; u^{\text{fast}}, \theta, \ldots)$ — the exact gradient gains an additional term, and the law extends to:

$$
\frac{\partial \hat y_j}{\partial u_{ij}^{\text{fast}}}
= b\,\psi_{ij} \;+\; \sum_{k} a_{kj}\,\frac{\partial \psi_{kj}}{\partial u_{ij}^{\text{fast}}}
\qquad\Rightarrow\qquad
u_{ij}^{\text{fast}} \leftarrow u_{ij}^{\text{fast}}
+ \eta\,(y_j - \hat y_j)\Big(b\,\psi_{ij} + \underbrace{\textstyle\sum_k a_{kj}\,\partial \psi_{kj}/\partial u_{ij}^{\text{fast}}}_{J_\psi}\Big)
$$

i.e. the Jacobian $J_\psi$ of the relationship function with respect to the plastic state joins the update. Running the §3.1 rule while $\psi$ depends on the fast state is **not** an approximation of gradient descent — it is a different (biased) rule — and would require an explicit mathematical justification plus owner sign-off under §9.

**What is deliberately *not* in the update.** The coactivity eligibility trace does not scale the error-driven step; it only feeds masks, consolidation gates and statistics (Test 4 in `tests/test_connections_learning.py` proves the update is invariant to trace rescaling by $0.5\times$ and $1000\times$). The uncertainty estimate $\sigma_{\text{err}}$ (batch error std) is **reported only** (`uncertainty_scale`, lines 779–782) and does not scale the step.

---

## 4. Consolidation — annealing exact-solve

Consolidation moves knowledge from the fast state into the slow state. Its law is stricter than "transfer": **the transmitted output must not move.**

Per connection, with transfer fraction $f$ (`transfer_fraction`, frozen $0.1$), current blend $b$, and consolidation set $M$ (entries passing the activity mask and an open selective gate, §4.1):

$$
d_{ij} \;=\; f\cdot u_{ij}^{\text{fast}} \quad (ij \in M), \qquad d_{ij} = 0 \text{ otherwise}
$$

**Annealing.** Each consolidation event that transfers anneals the blend:

$$
b' \;=\; \max\!\big(b_{\text{floor}},\; b - \text{anneal}\big)
\qquad\text{(frozen: anneal } 0.05,\ \text{floor } 0.5\text{)}
$$

(`blend_anneal`, `blend_floor`, lines 286–287; the solve, lines 447–457.)

**Exact solve.** The new fast and slow states are solved exactly from $a$-preservation:

$$
a_{\text{before}} = b\,u^{\text{fast}} + (1-b)\,\theta^{\text{slow}}
\qquad
u' = u^{\text{fast}} - d
\qquad
\theta' = \frac{a_{\text{before}} - b'\,u'}{\,1 - b'\,}
$$

(elementwise; `plastic.py` lines 451–457). Because $a_{\text{before}}$ is computed from the pre-event states and $\theta'$ absorbs **both** the transfer and the blend change, the effective connection after the event is

$$
b'\,u' + (1-b')\,\theta' \;=\; a_{\text{before}}
$$

identically, for **every** entry — masked and unmasked — so $\hat y$ is invariant.

**Reduction to the classic transfer.** With no annealing ($b' = b$) the solve algebraically reduces to the form recorded in the v1 freeze manifest:

$$
\theta' = \theta^{\text{slow}} + \frac{b}{1-b}\,d
\qquad\text{(manifest: } \theta \leftarrow \theta + (\text{blend}/(1-\text{blend}))\cdot d\text{)}
$$

— the same computation; the exact solve is its generalisation under annealing.

**Preservation invariant (protected).**

$$
\big\| \big(b'u' + (1-b')\theta'\big) - \big(b\,u^{\text{fast}} + (1-b)\theta^{\text{slow}}\big) \big\|_\infty \;\leq\; 10^{-6}
\qquad\text{and}\qquad
\|\hat y' - \hat y\|_\infty \leq 10^{-6}
$$

asserted numerically in the test suite (tests 5, 8, 9: `atol=1e-6, rtol=1e-6`).

**Why annealing exists.** At $b = 1.0$ the slow weights carry zero weight in $a$, so no transfer can preserve the effective connection while $b$ stays 1 — the blend=1.0 deadlock (documented in `consolidate()`, lines 392–396, and broken by Test 8: blend anneals $1.0 \to 0.95$ on the first transfer and is written back to the connection, line 817). Annealing walks $b$ down toward its floor $0.5$, never past it and never up (Test 9 runs repeated events at $b \in \{1.0, 0.7, 0.3\}$ and asserts preservation at every step).

### 4.1 Selective gating and controls

- **Consolidation selection:** entries pass if $|u^{\text{fast}}| > 0.1$ (threshold) or the trace activity mask selects them (lines 422–426).
- **Selective gates:** each connection carries a gate $g_{ij} = \sigma(\text{gate param})$; only $g_{ij} > 0.5$ entries transfer (lines 428–438). The gate encodes **stability of exercise, not error level** (`update_evidence`, lines 467–502): the scale-free statistic
  $\;\text{stability} = \bar{|e|}^2 / \overline{|e|^2} \in [0,1]$ (equal to $1/(1+\text{CV}^2)$), smoothed by a $0.99$ EMA, drives the target gate via $\sigma((\text{stability} - 0.5)\cdot 10)$. Intermittent or absent coactivity keeps the gate shut.
- **Controls stay valid:** `consolidation_rate = 0` disables consolidation outright — the interval alone must never fire sweeps (`consolidate()` early-return, lines 403–411; Test 10), otherwise Plastic-D0 controls would silently consolidate and the experiment matrix would be invalid.
- **Scheduling:** periodic consolidation fires every `consolidation_interval` updates (default 1000; line 812–818), before the update counter increments.

---

## 5. The five protected properties

*Preserve forever, or demonstrate a mathematically justified improvement.* (README, "AION Plasticity Core".)

| # | Property | Why it matters | Proven today by |
|---|---|---|---|
| 1 | **Forward-update consistency** — the learning rule follows the actual computation | If the update optimises a different quantity than the forward map computes, "learning" is drift with a good name. Prediction must be the actual forward output, and the update must be its exact gradient. | `test_forward_consistency_finite_difference` (Test 1): one real update equals $-\eta$ times the central finite-difference gradient, `allclose(rtol=1e-3)`; gate-4 network-level finite-difference agreement **max rel err 1.95e-4** (`freeze/AION_LEARNING_BASELINE_20261124.md`). Code: `predicted = transmitted.sum(-1)` (`plastic.py:765`), `plasticity_step` Jacobian comment (`plastic.py:319–369`). |
| 2 | **Temporal credit assignment** — eligibility traces remember contributing activity | A relationship built over many steps must stay creditable after the contributing activity has passed; contributions that never co-occurred must not steal credit. | `test_trace_separation` (Test 4): coactivity trace evolves identically with/without an error target; $u^{\text{fast}}$ moves only with a target; the error-driven update is bit-identical under trace rescaling $0.5\times$/$1000\times$. Code: `EligibilityTrace` (`plastic.py:23–131`, $e \leftarrow \gamma e + (1-\gamma)(\text{post}\otimes\text{pre})$, adaptive decay clamp $[0.8, 0.999]$; half-life ≈ 13.5 updates at $\gamma=0.95$, ≈ 69 at $\gamma=0.99$); network-level supervision split `target / in_degree` (`network.py:202–206`). |
| 3 | **Contextual plasticity** — relationships can change with circumstances | The same pair of neurons may mean different things in different contexts; the substrate must be able to express that without corrupting the context-free map. | Code: context gate $\sigma(W_g c_t)$ modulating $\psi$ per connection (`plastic.py:227–229, 236–238`) and the `ContextualRelation` per-context gate/bias mechanism (`plastic.py:527–589`). Honest note: the frozen calibration point and the connection unit suite run `context_dim = 0`, so this property is currently proven at the code level, not yet by a dedicated numerical test — a gap, recorded as such. |
| 4 | **Dual-timescale memory** — fast learning becomes persistent knowledge | Learning that lives only in the fast state dies with the episode. Consolidation must move it into $\theta^{\text{slow}}$ **without moving $\hat y$** — persistence, not rewriting. | Tests 5, 7, 8, 9, 10: output preservation `atol/rtol 1e-6` (Test 5); gates open under realistic O(1) error and stay shut without coactivity, gate means **> 0.8** vs **< 0.3** (Test 7); blend-1.0 deadlock broken by anneal $1.0 \to 0.95$ (Test 8); invariant holds over repeated events at $b \in \{1.0, 0.7, 0.3\}$ (Test 9); `consolidation_rate = 0` disables (Test 10). Code: exact solve + anneal (`plastic.py:447–457`), stability gate (`plastic.py:467–502`). |
| 5 | **Stability** — learning doesn't destroy previous capabilities | An organism that forgets what it already does every time it learns something new is not learning. Updates must be bounded, sign-symmetric and far from safety clamps in normal operation. | `test_gradient_direction_one_step` (Test 2, parametrised all-positive **and** signed/zero-mean post): loss strictly decreases in both regimes, $\max\|u^{\text{fast}}\| < 5.0$; gate-4: no divergence under negative/zero-mean inputs, plastic-state growth $\max\|u\| = 0.63$ — **93.7 % margin** from the $\pm10$ clamp; pilot rotation retention: plastic $1.0029$ vs static baseline $\sim 1.0$ (error does not grow; `scripts/eval_fix_report.md`). Code: decay + $\pm10$ clamp (`plastic.py:361–362`). |

---

## 6. The learning contract

Before any claim is made on a host substrate — Qwen, Gemma, Pith, or something not yet discovered — it must pass the same exam. The contract is substrate-agnostic by design: it examines the *behaviour*, not the architecture.

**The exam (four gates):**

1. **Controlled-task learning.** On a task with known ground truth ($y = 2x_1 + 3x_2$ is the canonical probe), held-out error must measurably improve. Absolute errors are the primary evidence; similarity-style diagnostics may accompany them, never replace them (reporting law, `scripts/eval_fix_report.md`, "Reporting law").
2. **Coefficient recovery via the EFFECTIVE mapping.** The recovered coefficients come from $a = b\,u^{\text{fast}} + (1-b)\,\theta^{\text{slow}}$ — what the network actually transmits — not from the raw fast state. Similarity (cosine) is reported **alongside absolute errors**, both required:
   - v2 gate-4 result: $a = [2.0003,\ 2.9989]$ vs truth $[2, 3]$; **abs err $[0.0003,\ 0.0011]$**; **$\cos = 0.999998$**.
   - predictions on $y = 2x_1 + 3x_2$: $[2.003,\ 2.993,\ 4.997,\ 7.000]$, **MSE $1.7\mathrm{e}{-5}$**.
   - gain calibration law ($g^2\eta$ invariance): $\|\Delta \hat y\| = 0.000946$ identical across $(g,\eta) = (1, 0.05), (2, 0.0125), (4, 0.003125)$; spread $0.0$.
   - held-out connection prediction MSE: $1.026 \to 0.577$ (**$-43.8\ \%$**); per-seed $-51.6\ \%$, $-49.2\ \%$.
3. **Output-preservation invariants.** Consolidation must preserve $a$ and $\hat y$ within $10^{-6}$ (§4); the substrate exam re-proves this on the host's own consolidation path.
4. **Held-out improvement.** Improvement must be measured on held-out data, with the static/no-learning control genuinely disabled (a control that secretly learns is not a control — the pre-fix harness had exactly that bug, `scripts/eval_fix_report.md` Part 2, blocker 4).

### Versioned benchmark registry

| Version | Date | Archive | SHA-256 (verified) | Contents / delta |
|---|---|---|---|---|
| **v1** | 2026-11-24 | `freeze/aion_learning_baseline_20261124.tar.gz` | `a39013c90fb5f36118a76c35d3a6d5dde01d8c903a576a892d89a75218132daf` | Gate-4 baseline. Manifest `freeze/AION_LEARNING_BASELINE_20261124.md`; frozen hashes `plastic.py` `f3f750e7…82e6`, `network.py` `413e8d45…cfcf`, `consolidation.py` `2aed7ed2…e3bf`; suite **7/7 green** (6 test functions, `test_gradient_direction_one_step` parametrised ×2). |
| **v2** | 2026-11-25 | `freeze/aion_learning_baseline_v2_20261125.tar.gz` | `ea802b9603aaf4a9e91816a8d8b59ddabeedb30a2ca36906762084d9f38734af` | Deadlock-fix baseline. Scope: `src/nnfs/connections/**`, `tests/test_connections_learning.py`, `scripts/diagnostic_learning.py`. Delta vs v1: `plastic.py` (now `9ceaaff8…3d37`) and the test file only — `network.py` and `consolidation.py` unchanged (byte-identical between both archives); suite grows to 10 test functions (adds tests 7–10: gate reachability, blend-anneal transfer, repeated-event invariant, rate-0 controls). |

The v2 calibration point (all contract runs above): `relation_type="post"`, $W$-init gain $g = 1.0$, $\eta = 0.05$, `state_leak_rate` $0.5$, `blend_ratio` $1.0$, `transfer_fraction` $0.1$, consolidation interval default 1000 (v1 manifest, "Calibration settings"; carried into v2).

---

## 7. Reference implementation pointers

**`src/nnfs/connections/plastic.py`** — the mathematics, in code:

| Lines | Symbol | What it is |
|---|---|---|
| 23–131 | `EligibilityTrace` | Coactivity trace $e \leftarrow \gamma e + (1-\gamma)(\text{post}\otimes\text{pre})$; adaptive decay predictor clamped to $[0.8, 0.999]$; activity mask at threshold $10^{-4}$. Masks/gates/statistics only — never the error-driven step. |
| 134–261 | `RelationshipFunction` | $\psi(h_i, h_j, c_t)$; families `bilinear` / `post` / `mlp` / `attention`; sigmoid context gate. |
| 264–524 | `FastSlowPlasticity` | `plasticity_step` (319–369): the learning law; `consolidate` (371–465): annealing exact-solve + rate-0 disable; `update_evidence` (467–502): stability gate; `get_combined_weight` (504–508): $a = b\,u + (1-b)\theta$. |
| 644–853 | `PlasticRelationalConnection` | The composed unit: forward (743–749, 765), error sign convention `error = target − predicted` (770), forward-participation mask (791), periodic consolidation with blend write-back (812–818). |

**`tests/test_connections_learning.py`** — which test proves which property:

| Test | Function | Proves |
|---|---|---|
| 1 | `test_forward_consistency_finite_difference` | Property 1 — update $=$ finite-difference gradient $\times(-\eta)$ |
| 2 | `test_gradient_direction_one_step` (×2 regimes) | Property 5 — descent in both post regimes; clamp margin |
| 3 | `test_independent_inputs_coefficient_recovery` | Contract gate 2 — effective mapping recovers $[2,3]$; $\cos>0.999$, abs err $<0.05$, MSE $<10^{-3}$, $u = a/0.7$ |
| 4 | `test_trace_separation` | Property 2 — trace/error separation; update invariant to trace scaling |
| 5 | `test_consolidation_output_preserving` | Properties 4+5 — $a$, $\hat y$ invariant within $10^{-6}$; both sides move; interval fires |
| 6 | `test_metric_contract_keys` | Measurement hygiene — primary metric exists, readout never masks connection learning, `None` without target |
| 7 | `test_evidence_gate_reachable_at_o1_error` | Property 4 — gates open under O(1) error ($>0.8$), shut without coactivity ($<0.3$) |
| 8 | `test_consolidation_blend_1_anneals_and_transfers` | Property 4 — blend-1.0 deadlock broken ($1.0 \to 0.95$), preservation through transfer+anneal |
| 9 | `test_consolidation_preservation_invariant_random` | Properties 4+5 — invariant over repeated events at $b \in \{1.0, 0.7, 0.3\}$; blend walks down only |
| 10 | `test_consolidation_rate_zero_disables` | Property 5 — control validity: rate 0 means no consolidation, periodic path included |

**`scripts/diagnostic_learning.py`** — the gate runner. Sections `[MEASURE]` ($\psi$ RMS / distribution, $|\Delta u|$ per update), `[G2 LAW]` (calibrated $(g,\eta)$ pairs, invariance + stability boundary), `[GATE-4]` (the seven acceptance criteria, including coefficient recovery and negative/zero-mean stability). Full gate-4 rerun in ~30 s (`freeze/AION_LEARNING_BASELINE_20261124.md`); run as `PYTHONPATH=src python scripts/diagnostic_learning.py`.

---

## 8. Honest status

**Demonstrated, on controlled tasks (v2 freeze evidence):**

- Network-level held-out `connection_prediction_mse` $1.026 \to 0.577$ — **$-43.8\ \%$** (seeds $-51.6\ \%$, $-49.2\ \%$).
- Coefficient recovery through the effective mapping: $a = [2.0003,\ 2.9989]$ vs $[2, 3]$, **absolute errors $[0.0003,\ 0.0011]$**, cosine **$0.999998$**; predictions $[2.003,\ 2.993,\ 4.997,\ 7.000]$, MSE $1.7\mathrm{e}{-5}$.
- Finite-difference gradient agreement max rel err $1.95\mathrm{e}{-4}$; gain-calibration invariance spread $0.0$; no divergence under negative/zero-mean regimes; plastic state at $93.7\ \%$ clamp margin.
- Pilot differentiation on the corrected instrument (`scripts/eval_fix_report.md`): Plastic beats Static on rotation adaptation (phase A $0.7508$ vs $1.0008$, $-25.0\ \%$; phase C $0.7529$ vs $1.0043$) and discovery ($0.9696$ vs $1.0032$).

**Not yet proven — recorded so no one mistakes the map for the territory:**

- **Conversational knowledge acquisition** — the ability to absorb knowledge from dialogue, the mechanism the identity section is ultimately about. AION-1 Phase 1 scored **3/5** (retention, correction, transcript-removal PASS; acquisition and generalisation honest FAIL — Base-model expression gap); **Phase 2 is in flight** (`README.md` stage table, 2026-11-25). Until it passes, the learning contract's conversational extension is unproven.
- **Dreaming benefits** — no configuration has yet shown that dreaming improves learning or retention on a valid instrument: D1 replay is bit-identical to D0 on every field and seed; D2 (imagination) ties D1 on held-out discovery ($0.9694$ vs $0.9696$) and actively hurt rotation retention ($1.0810$ vs $1.0029$); D3 helps rotation adaptation/retention ($0.7213$ / $0.9996$) but not discovery. The Dream Engine **re-implementation is deferred** (owner decision, `README.md` stage 04 row).
- **Network-level consolidation** — the correctness of consolidation is unit-proven (§4), but consolidation behaviour at rate > 0 in a live continual run had not been exercised as of the freeze; the gate-6 pilot found 0 elements transferred across all configs with `gate_mean` $0.062$ — a design-level finding (reported as FAIL per owner directive, not tuned away), since repaired at the unit level by the stability-gate and anneal fixes (tests 7–8). Its live continual behaviour remains unproven.

---

## 9. Change control

1. **The five protected properties (§5) change only via mathematically justified improvement, with owner sign-off.** "It works better in the pilot" is not a mathematical justification; a derivation showing the new rule preserves (or strengthens) the property is.
2. **Any mechanism change — forward map, learning law, consolidation solve, gate statistics, calibration point — requires a new freeze manifest** in `freeze/`, with SHA-256 of the archive and per-file hashes, before the new point becomes citable evidence. Results quoted without a freeze pointer are anecdotes.
3. **The learning contract (§6) is substrate-agnostic and versioned.** A new substrate passes the same exam; a changed exam is a new contract version, and claims made under the old exam are not comparable to claims under the new one.
4. **Honesty is part of the spec.** Negative results (the unlearnable-instrument run, the gate-6 FAIL, the D1=D0 inertness) are records, not embarrassments — they are preserved in `pilot_results/` and cited in `scripts/eval_fix_report.md`. Erasing them is a spec violation.
5. **Scope of this document.** APC governs the plastic-neuron mathematics in `src/nnfs/connections/`. Host adapters (`aion/`), the Dream Engine (`src/nnfs/dreaming/`) and benchmark harnesses are governed by their own documents; where they touch the plasticity mathematics, they inherit §9.1–9.2.
