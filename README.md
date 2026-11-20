# Machine Learning: Optimizers Comparison across Regression, Classification & Random Forest

This repository contains **5 Kaggle notebooks** exploring and comparing **5 different gradient-based optimization algorithms** across three progressive Machine Learning tasks:
1. **Assignment 1:** Polynomial Linear Regression.
2. **Assignment 2:** Binary Logistic Regression.
3. **Assignment 3:** Hybrid **Random Forest + Gradient Optimizers** Stacking Model.

---

## 📂 Repository Structure
* `1-sgd-optimizer.ipynb` — Stochastic Gradient Descent (`SGD`)
* `2-mini-batch-gd-optimizer.ipynb` — Mini-Batch Gradient Descent (`partial_fit`)
* `3-momentum-optimizer.ipynb` — Gradient Descent with Momentum (`momentum=0.9`)
* `4-adagrad-optimizer.ipynb` — Adaptive Gradient (`learning_rate='adaptive'`)
* `5-adam-optimizer.ipynb` — Adaptive Moment Estimation (`Adam`)

---

## 📊 Dataset & Preprocessing
* **Dataset:** [Climate Change: Earth Surface Temperature Data](https://kaggle.com) (`GlobalLandTemperaturesByMajorCity.csv`).
* **Subset:** Monthly average temperatures for **Chicago** (`2000-07-01` to `2001-07-01`), capturing a full seasonal U-shape cycle ($\text{Summer} \rightarrow \text{Winter} \rightarrow \text{Summer}$).
* **Feature Engineering:**
  * Centered time steps (`Months Centered`).
  * Polynomial feature expansion (`degree=2`) to capture non-linear seasonal curves.
  * Feature standardization using `StandardScaler`.
* **Binary Target Formulation (Assignments 2 & 3):**
  * **Class `1` (Warm):** Temperature above the annual mean.
  * **Class `0` (Cold):** Temperature at or below the annual mean.

---

## 🌲 Assignment 3: Random Forest + Optimizers Results
In this phase, a `RandomForestClassifier` (`n_estimators=100`) extracts probabilistic features from the scaled polynomial inputs, which are then fed into each respective optimizer to refine the decision boundary.

### 📈 Evaluation Metrics Comparison

| Notebook | Optimizer | Log Loss | Accuracy | Precision | Recall | F1-Score |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `1-sgd-optimizer.ipynb` | **Stochastic GD (SGD)** | `0.1622` | `1.0000` | `1.0000` | `1.0000` | `1.0000` |
| `2-mini-batch-gd-optimizer.ipynb` | **Mini-Batch GD** | `0.2121` | `1.0000` | `1.0000` | `1.0000` | `1.0000` |
| `3-momentum-optimizer.ipynb` | **GD with Momentum** | `0.1886` | `1.0000` | `1.0000` | `1.0000` | `1.0000` |
| `4-adagrad-optimizer.ipynb` | **AdaGrad** | `0.1610` | `1.0000` | `1.0000` | `1.0000` | `1.0000` |
| `5-adam-optimizer.ipynb` | **Adam** | `0.1706` | `1.0000` | `1.0000` | `1.0000` | `1.0000` |

### 💡 Key Takeaways
* Combining **Random Forest** feature probabilities with gradient-based optimizers achieved **100% classification accuracy (`1.0000`)** across all five notebooks.
* **AdaGrad** achieved the lowest probabilistic error (**Log Loss = `0.1610`**), closely followed by **SGD (`0.1622`)** and **Adam (`0.1706`)**.

---

## 🛠️ Technologies Used
* **Python 3**
* **Pandas & NumPy** — Data manipulation and numerical computations
* **Scikit-Learn** — `RandomForestClassifier`, `SGDRegressor`, `SGDClassifier`, `MLPRegressor`, `MLPClassifier`, `PolynomialFeatures`, `StandardScaler`, `metrics`
* **Matplotlib** — Data distribution and decision boundary visualization
