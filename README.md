
# EVO-STACK-KDE — MVP Execution Plan (1‑Day Build)

**Goal (MVP):** Implement a minimal, working version of **EVO‑STACK‑KDE** and run it on **one real dataset** (UCI *Wine Quality – Red*). The plan is written so an LLM (or automation) can follow it **sequentially without backtracking**: each step produces artifacts and stable interfaces consumed by the next steps.

**Scope constraints (for speed & clarity):**
- Only **Gaussian** kernels in this MVP; diversity comes from **bandwidth** and **subspace** (feature masks).
- **NSGA‑II** with **3 objectives**: accuracy (NLL), stability (bootstrap SD of NLL), complexity (simple proxy).
- No external benchmarking against other KDE variants (yet). No heavy performance tuning.
- Deterministic defaults; single dataset; single entrypoint script.

---

## 0) Repository Bootstrap (Scaffold First)

**Outcome:** a runnable repo skeleton with pinned deps and deterministic settings, ready for coding.

1) **Create structure**
```
evo_stack_kde/
  README.md
  requirements.txt
  .gitignore
  src/
    __init__.py
    data.py           # data I/O, split, scaling, persistence
    kde_expert.py     # KDEExpert: Gaussian KDE over an optional feature subspace
    ensemble.py       # KDEEnsemble: convex combination (log-sum-exp)
    eval.py           # f1: NLL (CV), f2: stability (bootstrap), f3: complexity (proxy)
    genome.py         # configs (dataclasses), encode/decode helpers
    nsga.py           # DEAP NSGA-II setup and loop
    plots.py          # Pareto plots (+ simple hist)
    runner.py         # CLI entry point (end-to-end run)
  data/
    raw/              # place winequality-red.csv here
    processed/        # generated splits + meta
  outputs/            # generated artifacts (git-ignored)
    models/
    figures/
    runs/
    logs/
```
2) **.gitignore**
```
.venv/
__pycache__/
.ipynb_checkpoints/
outputs/
data/processed/
*.pkl
*.npz
```
3) **requirements.txt**
```
numpy>=1.23
scipy>=1.10
pandas>=1.5
scikit-learn>=1.3
deap>=1.3
matplotlib>=3.7
joblib>=1.3
```
4) **Determinism policy**
- Use `RANDOM_SEED = 42` and seed Python, NumPy, and DEAP in `runner.py`.
- Avoid hidden global randomness elsewhere.

5) **README (stub)**
- One-paragraph overview and run commands:
```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m src.runner --csv data/raw/winequality-red.csv --seed 42
```

**Acceptance:** environment installs; repository layout matches; `python -c "import numpy"` works.

---

## 1) Data Ingestion & Preprocessing (Single Dataset)

**Outcome:** scaled **train/val/test** arrays serialized to `data/processed/`, plus `meta.json`. No later step needs to revisit ingestion.

**Implement `src/data.py`:**
- Functions:
  - `load_wine_red(csv_path: str) -> pd.DataFrame`: read UCI CSV (`;` or `,` autodetected).
  - `split_and_scale(df, test_size=0.15, val_size=0.15, seed=42) -> DataSplits`:
    - Split into train vs. temp (val+test), then temp into val/test.
    - Apply `StandardScaler` (fit on train, transform val/test).
  - `persist_splits(splits: DataSplits, out_dir: str)`: save `splits.npz` (X_train, X_val, X_test) + `meta.json`.

**Artifacts produced:**
- `data/processed/splits.npz`
- `data/processed/meta.json` (n_train, n_val, n_test, d, scaler stats)

**Acceptance:** shapes in `meta.json` look correct; all arrays are floats; no labels are used.

---

## 2) Single KDE Expert (Gaussian) with Optional Subspace

**Outcome:** a reusable `KDEExpert` with a stable API used by the ensemble and the evaluator. It supports a feature mask and an `alpha` scale on Scott’s bandwidth.

**Implement `src/kde_expert.py`:**
- `scott_bandwidth(X: np.ndarray) -> float`: compute Scott’s rule \( h = n^{-1/(d+4)} \) on subspace.
- `class KDEExpert(feature_mask: np.ndarray, alpha: float=1.0)`:
  - `fit(X)`: select subspace, compute \( h=\alpha\,h_0 \), fit `KernelDensity(kernel="gaussian", bandwidth=h)`.
  - `logpdf(X)`: return per-sample log density on subspace.

**Acceptance:** smoke test: fit on a small random matrix; `logpdf` returns 1D array of expected length.

---

## 3) Ensemble via Convex Log‑Sum‑Exp

**Outcome:** `KDEEnsemble` combining experts with non‑negative weights summing to 1; numerically stable log-density via log‑sum‑exp.

**Implement `src/ensemble.py`:**
- `class KDEEnsemble(experts, weight_logits)`:
  - `fit(X)`: fit each expert on the same `X` (already scaled).
  - `logpdf(X)`: `softmax(weight_logits)` → weights; compute `logsumexp(logpdf_j + log w_j)` across experts.
- Internal `@staticmethod _softmax(z)` used by complexity (optional entropy) and evaluation.

**Acceptance:** with one expert and `weight_logits=[0]`, ensemble equals the expert up to tolerance.

---

## 4) Objectives (MVP): f1, f2, f3

**Outcome:** pure functions that evaluate a **model factory** (produced from a config) on X. They will be called by NSGA‑II. No mutation of global state.

**Implement `src/eval.py`:**
- `nll_kfold(model_factory, X, k=3, seed=42) -> float`:
  - KFold(k, shuffle=True, random_state=seed): for each split, fit `model_factory()` on train and compute **mean NLL** on val.
- `stability_bootstrap(model_factory, X, B=10, k=3, seed=42) -> float`:
  - For b in 1..B: bootstrap-resample rows of X, compute `nll_kfold` (same k), collect means → return **std** of these NLLs.
- `complexity(config, n, d, lam_entropy=0.0) -> float`:
  - For each expert: `C_j = (d_j / d) * min(1, n/2000)` where `d_j` = subspace size; return `mean(C_j)`.
  - Optional entropy term on weights: `H_norm = -∑ w log w / log m`; add `lam_entropy * H_norm` if lam>0.

**Artifacts used downstream:** callable `nll_kfold`, `stability_bootstrap`, `complexity`.

**Acceptance:** unit calls on tiny arrays return floats; runs quickly.

---

## 5) Genome (Configs) & Decode to Model

**Outcome:** JSON‑serializable model configs and a decoder that returns a **model factory** used by objectives.

**Implement `src/genome.py`:**
- Dataclasses:
  - `ExpertConfig(alpha: float, feature_mask: np.ndarray)`
  - `ModelConfig(weight_logits: np.ndarray, experts: list[ExpertConfig])`
  - `to_jsonable()` to convert masks to 0/1 lists for saving.
- Helpers:
  - `random_mask(d, rng, keep_frac_range=(0.6, 0.8)) -> np.ndarray`
  - `random_model_config(d, rng, m=5) -> ModelConfig` (alphas ∈ [0.5, 1.5], random masks, random logits)
  - `decode_to_model(config: ModelConfig) -> Callable[[], KDEEnsemble]` (model factory, unfitted).

**Acceptance:** `random_model_config` followed by `decode_to_model` produces a factory that fits & scores without error.

---

## 6) NSGA‑II Loop (DEAP)

**Outcome:** a complete NSGA‑II run that returns a Pareto set and writes `pareto.json`. It uses the objectives strictly through their APIs; no hidden coupling.

**Implement `src/nsga.py`:**
- Fixed **m=5** experts for MVP.
- Population/build:
  - `creator.FitnessMin3(weights=(-1,-1,-1))`
  - `creator.Individual(object)`
  - Toolbox:
    - `individual`: returns `ModelConfig` via `random_model_config`
    - `population`: `tools.initRepeat(list, toolbox.individual)`
    - `evaluate(config)`: builds factory, returns `(f1, f2, f3)` using `eval.py`
    - `mate(a, b)`: 1‑point crossover over experts + blend on `weight_logits`
    - `mutate(config)`: multiplicative jitter on `alpha` (log‑space), bit‑flips on masks (ensure ≥1 feature), Gaussian jitter on logits
    - `select`: `tools.selNSGA2`
- Loop:
  - `pop_size=60`, `n_gen=40`, `cxpb=0.9`, `mutpb=0.4`
  - Evaluate initial, then tournament DCD → variation → evaluate → select
- **Outputs:**
  - `pareto.json`: list of configs + fitness for nondominated individuals
  - Return `(pareto, run_dir_path)` for the runner

**Acceptance:** execution finishes within target time on a laptop; `outputs/runs/.../pareto.json` exists and is valid JSON.

---

## 7) Runner (CLI): End‑to‑End Training, Selection, Testing, Saving

**Outcome:** **single command** to: prepare data → run NSGA‑II → pick “knee” solution → test on hold‑out → save artifacts and plots. No earlier step is revisited.

**Implement `src/runner.py`:**
- Parse args: `--csv`, `--seed`, `--outdir`, `--pop`, `--gens`, `--kfold`, `--bootstraps`.
- Seed NumPy, Python, DEAP.
- **Data**: load → split/scale → persist (so later runs can reuse).
- **Evo**: call `run_nsga` with `X_train` only (validation handled inside objectives).
- **Pick knee**:
  - From `pareto` fitness triplets `(f1,f2,f3)`, min‑max normalize each objective to [0,1] (across Pareto set) and pick the point with smallest Euclidean distance to (0,0,0).
- **Build final model**:
  - Decode knee config → fit ensemble on **train+val** (entire available training budget).
- **Evaluate test**:
  - Compute **test NLL** on `X_test`; save to `metrics.json` together with chosen config & all three (f1,f2,f3) for transparency.
- **Save artifacts**:
  - `outputs/models/best_model.pkl` (via `joblib.dump` → store config & scaler)
  - `outputs/runs/<timestamp>/pareto.json`
  - `outputs/figures/pareto_2d.png` (e.g., two projections) and `hist_test_logp.png`

**Acceptance:** one command runs start‑to‑finish, producing model, metrics, and plots.

---

## 8) Plots (Minimal)

**Outcome:** quick visual checks to reason about the run.

**Implement `src/plots.py`:**
- `plot_pareto_2d(pareto_fits, out_png)`: two 2D projections (f1 vs f2, f1 vs f3).
- `plot_hist_neglogp_test(logp, out_png)`: histogram of `-log p(x)` on test set.

**Acceptance:** files exist and open; axes labeled; no special styling required.

---

## 9) Command to Run (End‑to‑End)

```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m src.runner --csv data/raw/winequality-red.csv --seed 42 --outdir outputs
```

**Expected artifacts:**
- `data/processed/splits.npz`, `meta.json`
- `outputs/runs/<timestamp>/pareto.json`
- `outputs/models/best_model.pkl`
- `outputs/figures/pareto_2d.png`, `hist_test_logp.png`
- `outputs/metrics.json`

---

## 10) Validation Checklist (No Backtracking)

- [ ] Repo scaffold matches section **0**; deps install; seed policy in place.  
- [ ] Data split & scaler persisted (**1**).  
- [ ] `KDEExpert` fits/returns logpdf; Scott rule used; masks honored (**2**).  
- [ ] `KDEEnsemble` implements convex log‑sum‑exp; softmax weights (**3**).  
- [ ] Objectives `nll_kfold`, `stability_bootstrap`, `complexity` are pure and fast (**4**).  
- [ ] Genome configs are JSONable; decoder returns factory (**5**).  
- [ ] NSGA‑II loop returns `pareto.json` within time (**6**).  
- [ ] Runner selects knee, fits final model (train+val), evaluates test, saves all (**7**).  
- [ ] Plots render (Pareto projections + histogram) (**8**).  

---

## 11) Future (Post‑MVP) — Parking Lot

- Add kernels: Epanechnikov/Laplace/Student; boundary correction; adaptive bandwidth.
- Make `m` variable; add on/off gene per expert; promote true model sparsity.
- Replace complexity proxy with **measured time** (fit/inference) and/or FLOPs proxy.
- Add third plot: 3D Pareto or parallel coordinates for (f1,f2,f3).
- Optional regularizers: Dirichlet prior on weights; constraints on alpha.
- Expand datasets; add baselines; statistical comparisons.
