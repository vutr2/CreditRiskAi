# Learning notes

A record of the mistakes I made, the concepts I need to master, and checklists for future exercises.

**How to use:** whenever I hit a mistake or learn something new, add an entry to section 1 using the template at the end of this file. Before starting a new exercise, re-read section 3 (checklists).

---

## 1. Mistake log

### 2026-11-12 · CreditRiskAI project (Give Me Some Credit dataset)

#### 1.1 Checking for duplicates while the ID column is still there

- **Symptom:** `df.duplicated().sum()` returned `0`, but there were actually 609 duplicate rows.
- **Cause:** the `Unnamed: 0` column (the row number) is different on every row, so no row counted as a duplicate.
- **Fix:** drop the ID column before checking: `df.drop(columns="Unnamed: 0").duplicated().sum()`.
- **Lesson:** an ID column carries no information about the borrower. Always exclude it before checking duplicates or computing statistics.

#### 1.2 Path that depends on the current folder

- **Symptom:** `load_train()` raised `FileNotFoundError` when run from the project root and from `notebooks/`. It only worked from the folder *above* the project.
- **Cause:** `RAW_DIR = "CreditRiskAI/data/raw"` is a relative path. It is resolved against the current working directory, not against the location of the code file.
- **Fix:** build the path from the file's own location:
  ```python
  RAW_DIR = Path(__file__).resolve().parents[3] / "data" / "raw"
  ```
- **Lesson:** library code (`src/`) must never use relative paths written as strings. Use `Path(__file__)` so the path is correct no matter where the code is run from.

#### 1.3 Using `/` to join `str` values

- **Symptom:** `TypeError: unsupported operand type(s) for /: 'str' and 'str'`.
- **Cause:** I changed `+` to `/`, but `RAW_DIR` was still a `str`. The `/` operator only works with `Path`.
- **Lesson:** decide up front whether a variable is a `str` or a `Path`, and use `Path` for every path. When changing one line, also check the related lines, here the line where the variable is defined.

#### 1.4 Comparing with `None` using `==`

- **Wrong:** `if path == None:`
- **Right:** `if path is None:`
- **Lesson:** some objects (pandas, numpy) redefine `==`. `is None` always means exactly "is this None".

#### 1.5 Error message that doesn't say what went wrong

- **Wrong:** `raise FileNotFoundError("Not found the path")`
- **Right:** `raise FileNotFoundError(f"Data not found: {path}")`
- **Lesson:** an error message must say *what* went wrong and *where*. With the path in the message, mistake 1.2 would have been obvious immediately.

#### 1.6 Fixing one place but forgetting an identical one

- **Symptom:** I fixed the error message in `load_test()`, but `load_train()` kept the old one. A test caught it.
- **Cause:** the two functions contained the same copy-pasted code.
- **Lesson:** when code repeats, move it into one shared function (for example `_read_raw_csv()`). Then one fix covers every caller (DRY: Don't Repeat Yourself).

#### 1.7 `load_test()` forgot to drop the empty target column

- **Symptom:** the test set came back with 11 columns, including a completely empty `SeriousDlqin2yrs`.
- **Consequence if not fixed:** `model.predict(test)` would fail because the columns don't match the ones used in training.
- **Lesson:** train and test output must have **the same feature columns, in the same order**. Write a test that checks it.

#### 1.8 Notebook raised `ModuleNotFoundError` even though the package was installed

- **Symptom:** `No module named 'credit_risk_ai'` in the notebook, while the same import worked in the terminal.
- **Cause:** the kernel was started *before* the package was installed. Python only looks for packages when it starts.
- **Fix:** restart the kernel.
- **Lesson:** always restart the kernel after installing a package. To make the notebook reload code changed in `src/` automatically, put this at the top:
  ```python
  %load_ext autoreload
  %autoreload 2
  ```

#### 1.9 Thinking the code didn't run because the cell showed nothing

- **Symptom:** the cell `df = load_train()` finished without showing anything, and I thought "nothing changed".
- **Cause:** a notebook only displays a value when the last line of the cell is an expression. An assignment (`df = ...`) displays nothing.
- **Lesson:** to see a result, put `df.head()` or `df.shape` on the last line. Check with code (`print(df.shape, df.index.name)`) instead of by eye.

#### 1.10 Thinking scaling belongs in `clean()`

- **First thought:** "clean isn't enough yet, we still need MinMaxScaler/StandardScaler".
- **Reality:** scaling is needed, but it **doesn't go in `clean()`**; it goes in an sklearn `Pipeline`. A scaler has to *learn* the mean/std (or min/max) from the data. Learning it on all the data before the train/validation split leaks data (see 2.1).
- **Lesson:** tell the two kinds of transformation apart (see 2.2).

#### 1.11 Choosing MinMaxScaler when the data has outliers

- **Measured:** the largest `MonthlyIncome` is about 3 million, so after MinMaxScaler 99.5% of incomes were squeezed below 0.01. The feature lost almost all its information.
- **Better:** `log1p` then `StandardScaler`, or `RobustScaler`, or cap the tail first.
- **Lesson:** look at the distribution (`describe()`, histogram) before choosing a scaler. MinMaxScaler only suits data without outliers.

#### 1.12 Code style mistakes (PEP 8)

- `#Function` → `# Function` (a space after `#`)
- `path = None` in a parameter list → `path=None`
- `index_col= 0` → `index_col=0`
- Two blank lines before each module-level `def`
- Typos in comments ("Fucntion")
- **Tip:** install `ruff` to check and fix formatting automatically: `ruff check .` and `ruff format .`

### 2026-11-13 · Notebook 02 (baseline model)

#### 1.13 Importing with a file path

- **Wrong:** `from "../credit_risk_ai/data/load" import load_train, clean` raises `SyntaxError`.
- **Cause:** two mistakes.
  1. `import` uses a **dotted module name**, not a file path in quotes.
  2. `clean` lives in `clean.py`, not in `load.py`. Each file is a separate module.
- **Right:**
  ```python
  from credit_risk_ai.data.load import load_train
  from credit_risk_ai.data.clean import clean, drop_duplicate_rows
  ```
- **Lesson:** turn a file path into a module name:
  `src/credit_risk_ai/data/load.py` → drop `src/`, drop `.py`, replace `/` with `.` → `credit_risk_ai.data.load`.
  Python finds the package because it is installed (`uv pip install -e .`), not because of a path.

### 2026-11-14 · Notebook 03 (cross-validation)

#### 1.14 A "default rate" that measured sizes, not labels

- **Symptom:** every fold printed `default_rate = 0.25`, while the expected value was about 0.067.
- **Cause:** the formula was `len(val_idx) / len(train_idx)` = 29,879 / 119,512 ≈ 0.25. That is the size ratio of validation to train (20% / 80%), and it never looks at `y`.
- **Fix:** the default rate is the share of 1s among the validation labels:
  ```python
  y.iloc[val_idx].mean()  # y is 0/1, so the mean is the share of 1s
  ```
  `.iloc` because `cv.split` returns row positions, not `id` values.
- **Lesson:** compare every output with its expected value before moving on. A number that is far off (0.25 vs 0.067), or suspiciously round (exactly ¼ = 20/80), means the formula measures the wrong thing. Ask: "which data does this formula actually read?"

---

## 2. Concepts to master

### 2.1 Data leakage

Every number *learned from data* (the median used to fill missing values, the mean/std used for scaling, a capping threshold...) **must be learned from the training part only**. If it is learned on all the data before splitting, the validation part has already "seen" information, and the validation score looks better than it really is.

**Rule:** split train/validation **first**, then `fit` anything. Put every step that has a `fit` inside a `Pipeline`; sklearn then guarantees this for you.

### 2.2 Two kinds of data transformation

| Kind | Examples | Where it goes |
|---|---|---|
| **Fixed rules** (learn nothing from the data) | income ≤ 1 → NaN, age < 18 → NaN, add flag columns | The `clean()` function, applied the same way to train and test |
| **Learned transformations** (need `fit`) | median imputation, StandardScaler, capping at a percentile | Inside a `Pipeline`, fitted on train |

**Why is `data/processed/` empty?** Because nothing needs saving yet. `load` + `clean` run in under a second, so calling the functions each time is enough, and the data always matches the latest code. Data that is *already imputed or scaled* must **not be saved ahead of time**: the medians/means would have been learned from all the data, which is leakage (2.1). Only save to `data/processed/` when a *fixed-rule* step is too slow (for example heavy feature engineering), and always create the file with code, never by hand.

### 2.3 When scaling is needed

- **Needed:** linear models (LogisticRegression, Ridge...), distance-based models (KNN, SVM), neural networks. Without scaling, LogisticRegression may not converge (seen here: still not done after 1000 iterations).
- **Not needed:** tree-based models (DecisionTree, RandomForest, GradientBoosting). Trees split on thresholds; changing the units doesn't change the order of the values.
- **Scaling barely changes ROC-AUC** (0.851 → 0.855 in this project). Its main effects are helping the model converge and making regularization penalize every feature fairly.

### 2.4 Imbalanced data

- With a 6.7% default rate, a model that predicts "no default" for everyone still gets 93.3% accuracy. **Accuracy is useless here.**
- Use ROC-AUC and PR-AUC.
- Split with `stratify=y` so train and validation keep the same default rate.

### 2.5 pandas points to remember

- `pd.read_csv(path, index_col=0)`: use the first column as the index.
- `df.duplicated()` **does not compare the index**, only the columns. NaN counts as equal to NaN.
- Any comparison with NaN is `False` (`NaN <= 1` is `False`).
- An `int` column cannot hold NaN; assigning NaN turns it into `float64`.
- Always `df.copy()` inside a function so the caller's DataFrame is not modified.

### 2.6 Read what each column really means

The most important issues in the audit all came from comparing **the definition in the data dictionary** with **the actual values**:
- `RevolvingUtilization` is a ratio (0–1), but has values up to 50,708.
- `DebtRatio` is debt / income, but when income is missing the values look like dollar amounts.
- The "times past due" columns contain 96 and 98: those are special codes (sentinel codes), not counts.

### 2.7 Testing with pytest

- `tmp_path`: a temporary folder, deleted after the test.
- `monkeypatch.chdir()`: temporarily change the current folder inside one test.
- `@pytest.fixture(scope="module")`: prepare data once and share it across the whole file.
- `@pytest.mark.skipif(...)`: skip a test when a condition isn't met (for example, no real data available).
- `@pytest.mark.parametrize`: run one test with several sets of inputs.
- `pytest.raises(Error, match="...")`: check that code raises the right error with the right message.
- **Test the tests:** break the code on purpose and check that a test fails. A test that can never fail protects nothing.
- Always test **boundary values**: if the rule is `> 10`, test both 10 (kept) and a value above 10 (removed).

### 2.8 Python project structure

- `src/` layout: code lives in `src/<package>/` and is installed with `uv pip install -e .` (editable: code changes need no reinstall).
- `pyproject.toml` declares the package and its dependencies.
- **Careful with `uv sync`:** it removes every package not listed in `pyproject.toml`.
- `.gitignore`: always exclude `.DS_Store`, `.venv/`, data, and secret files (`.env`).
- Commit early and often.

### 2.9 sklearn: how to use it, and the golden rule

#### The golden rule

> **`fit` and `fit_transform` run on the training data only.**
> **On validation and test data, use only `transform`, `predict`, `predict_proba`.**

Why: `fit` is where numbers are *learned* from data (medians, means, model coefficients...). Learning on validation/test data is data leakage (2.1). Validation/test data may only *use* what was learned from train.

#### Must know: everything in sklearn works the same way

| Kind | Method | What it does | Use on |
|---|---|---|---|
| **Transformer** (imputer, scaler...) | `fit(X)` | Learns numbers from the data | Train only |
| | `transform(X)` | Applies the learned numbers | Train, validation, test |
| | `fit_transform(X)` | `fit` then `transform` in one call | **Train only** |
| **Model** (estimator) | `fit(X, y)` | Learns from the data and labels | Train only |
| | `predict(X)` | Predicts 0/1 labels (threshold 0.5) | Validation, test |
| | `predict_proba(X)` | Predicts the probability of each class | Validation, test |
| **Pipeline** | same as a model | `fit` = `fit_transform` each step, then `fit` the model; `predict_proba` = `transform` each step, then `predict_proba` | Same as a model |

Once you know whether a tool is a transformer or a model, you know how to use it, because every sklearn tool follows this table.

#### Right and wrong

```python
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)  # RIGHT: learn mean/std from train, then apply
X_val_s = scaler.transform(X_val)          # RIGHT: reuse the mean/std learned from train

X_val_s = scaler.fit_transform(X_val)      # WRONG: re-learns mean/std on validation
```

A `Pipeline` follows the golden rule automatically, so it is the recommended way:

```python
pipe.fit(X_train, y_train)               # learns every step, from train only
proba = pipe.predict_proba(X_val)[:, 1]  # column 1 = probability of class 1 (default)
```

#### Common confusions

- **`predict_proba` instead of `predict`:** credit risk needs the *probability* of default, not just 0/1. `predict_proba` returns 2 columns (class 0, class 1); take column 1 with `[:, 1]`.
- **A classifier's `.score()` is accuracy**, which is useless on imbalanced data (2.4). Use `roc_auc_score(y_val, proba)` instead.
- **Attributes ending in `_`** (`statistics_`, `coef_`, `n_iter_`) hold what was learned during `fit`. For example, `pipe.named_steps["impute"].statistics_` shows the medians the imputer learned. Use it to check that the pipeline only learned from train.
- **Classification and regression use different metrics:**
  - Classification (predicting a label): `roc_auc_score`, `average_precision_score`, `confusion_matrix`.
  - Regression (predicting a number): `mean_absolute_error`, `r2_score`.

#### Commonly used modules

| Task | Module | Tools |
|---|---|---|
| Splitting, cross-validation, tuning | `sklearn.model_selection` | `train_test_split`, `StratifiedKFold`, `cross_validate`, `RandomizedSearchCV` |
| Chaining steps | `sklearn.pipeline` | `Pipeline` |
| Different processing per column group | `sklearn.compose` | `ColumnTransformer` |
| Filling missing values | `sklearn.impute` | `SimpleImputer` |
| Scaling, transforming | `sklearn.preprocessing` | `StandardScaler`, `RobustScaler`, `MinMaxScaler`, `FunctionTransformer`, `OneHotEncoder` |
| Baseline (random-guess) model | `sklearn.dummy` | `DummyClassifier` |
| Linear models | `sklearn.linear_model` | `LogisticRegression`, `LinearRegression`, `Ridge` |
| Tree-based models | `sklearn.ensemble` | `HistGradientBoostingClassifier`, `RandomForestClassifier` |
| Scoring | `sklearn.metrics` | `roc_auc_score`, `average_precision_score`, `RocCurveDisplay` |
| Probability calibration | `sklearn.calibration` | `CalibratedClassifierCV` |

No need to learn all of sklearn. The "everything works the same way" table and the golden rule are enough to use most of its tools.

### 2.10 Metrics that need a threshold

#### Two families of metrics

| Family | Metrics | Use it for |
|---|---|---|
| **Threshold-free** (use the probabilities directly) | ROC-AUC, PR-AUC, Brier score | Comparing models |
| **Threshold-based** (first turn probabilities into 0/1) | Precision, recall, F1, confusion matrix, accuracy | Deciding how to act on the predictions |

#### What each metric answers

- **Precision:** of the borrowers flagged as risky, how many really default?
- **Recall:** of the borrowers who really default, how many were flagged?
- **F1:** a single number combining precision and recall, with **equal** weight for both.
- **Confusion matrix:** the four counts behind precision and recall: true negatives, false positives (good borrowers flagged), false negatives (defaulters missed), true positives.
- **Brier score:** the average of (predicted probability − actual outcome)². Lower is better. Always predicting the base rate p gives p × (1 − p), which is 0.0625 for a 6.7% default rate. A useful model must beat that.

#### The 0.5 trap

`predict()` and every threshold-based metric computed from it silently use a threshold of **0.5**. On imbalanced data very few rows get a probability that high. In CreditRiskAI, HistGradientBoosting at 0.5 caught only **18.8%** of defaulters, even though its ROC-AUC was 0.862. At a threshold of 0.22, recall rose to **51.5%**. The model was fine; the default threshold was the problem.

#### Rules

- Compare models with threshold-free metrics. Use threshold-based metrics only after choosing a threshold, and **always report the threshold** next to precision, recall and F1.
- Lowering the threshold raises recall, lowers precision, and flags more borrowers. There is no free lunch: it is a trade-off.
- Choose the threshold from the business, not by habit:
  - **By cost:** minimize `cost_FN × FN + cost_FP × FP`. In credit risk a missed defaulter (lost principal) usually costs much more than a rejected good borrower (lost interest).
  - **By capacity:** if the team can review 10% of applications, pick the threshold that flags 10%.
  - Max-F1 assumes both errors cost the same, so it is only a neutral starting point.
- Choose the threshold on the **validation set only**, then keep it fixed. Never tune it on the test set (golden rule, 2.9).

### 2.11 Running in parallel: `n_jobs`

`n_jobs` (in `cross_validate`, `GridSearchCV`, `RandomizedSearchCV`, ...) is the **maximum number of tasks that may run at the same time**.

| Value | Meaning |
|---|---|
| `1` | 1 task at a time: the folds run one after another |
| `5` | up to 5 tasks at a time |
| `-1` | **as many as there are CPU cores** (10 on this Mac) |
| `-2` | all cores except one, so the computer stays responsive |

- `-1` does **not** mean "1 core". It is a code for "all cores", so the same code works on any machine.
- The number of tasks also limits it: `cross_validate` with 5 folds makes **5 tasks**. With `n_jobs=-1` on 10 cores, 5 cores work (one per fold) and 5 stay idle.
  ```
  n_jobs=1   ->  fold 0 -> fold 1 -> fold 2 -> fold 3 -> fold 4   (one after another)
  n_jobs=-1  ->  folds 0, 1, 2, 3, 4 all at once on 5 cores
  ```
- The results are **identical** for any value, because each fold is computed independently. Only the run time changes.

---

## 3. Checklists for future exercises

### Data audit

- [ ] Shape, dtypes, which column is the ID
- [ ] Missing values: how many, and is the target rate different between missing and non-missing rows?
- [ ] Duplicates, **after dropping the ID column**
- [ ] Target distribution (is it imbalanced?)
- [ ] Value ranges vs the data dictionary: any impossible values (age 0, ratio > 1...)?
- [ ] Special codes such as 96, 98, 999, -1
- [ ] Columns computed from other columns (ratios, totals): do they still make sense when a source column is missing?
- [ ] Do train and test have the same issues and the same distributions?
- [ ] Record everything in one table: issue, evidence, handling

### Writing code

- [ ] Paths built from `Path(__file__)`, never from relative strings
- [ ] Functions don't modify their input (`df.copy()`)
- [ ] Error messages include enough information (path, value)
- [ ] No copy-pasted code: move it into a shared function
- [ ] Named constants instead of bare numbers (`MAX_UTILIZATION = 10`)
- [ ] Tests: normal case, each rule, boundary values, real data (with `skipif`)
- [ ] Know the expected value before running a cell, and compare the output with it

### Building models

- [ ] Split train/validation **first**, with `stratify` if the data is imbalanced
- [ ] Every step with a `fit` lives inside a `Pipeline`
- [ ] Golden rule: `fit`/`fit_transform` on train only; validation/test only `transform`/`predict_proba`
- [ ] Use `predict_proba` + `roc_auc_score`, not `predict()` or `.score()`
- [ ] Compare models with threshold-free metrics: ROC-AUC, PR-AUC and Brier score (vs the base-rate Brier)
- [ ] Never report precision/recall/F1 at the default 0.5 without checking; choose the threshold on validation (by cost or capacity) and report it alongside
- [ ] Choose a suitable metric (no accuracy on imbalanced data)
- [ ] Build a baseline model first, then try more complex models
- [ ] Only scale when the model needs it; look at the distribution before choosing a scaler

---

## 4. Template for a new entry

```markdown
#### <number> <Short name of the mistake>

- **Symptom:** what I saw (error message, wrong result...)
- **Cause:** why it happened
- **Fix:** how I fixed it (with code if needed)
- **Lesson:** what to remember next time
```
