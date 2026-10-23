# Multinomial-sports-betting-predictive-engine
Predictive ML pipeline for sports betting odds calculation using Multinomial Logistic Regression and Random Forest ensembles.

# ⚽ Sports Betting Predictive Engine & ML Pipeline

## 📌 Business Problem Statement
Sports betting companies struggle to quickly and accurately calculate true match probabilities. Relying on slow manual calculations or outdated statistical formulas leads to inaccurate odds, exposing the business to heavy financial losses if smart bettors exploit those pricing errors.

## 🎯 Business Objective
To build an automated machine learning predictive system that ingests global sports datasets to predict match outcomes (`Home Win`, `Draw`, `Away Win`) and generate optimized, accurate betting probabilities with low latency.

---

## 🔍 Project Pipeline & Methodology

### 1. Exploratory Data Analysis (EDA)
* Analyzed historical team ELO ratings, recent win ratios, shots on target, and environmental conditions.
* Handled missing data points and validated class distributions across the 3 possible match outcomes.

### 2. Data Pre-processing & Feature Engineering
* **Feature Engineering:** Created derived features like `elo_difference` and `shot_differential`.
* **Imputation & Scaling:** Used median imputation for missing values and `StandardScaler` to normalize numeric distributions.
* **Train-Test Split:** Stratified split (80/20) to maintain balanced class representation.

### 3. Model Building & Benchmarking
We implemented and compared two distinct methods:
* **Method 1 (Baseline): Multinomial Logistic Regression**
  * Handled the multi-class target variable (`Away`, `Draw`, `Home`) and provided a fast, interpretable probabilistic baseline.
* **Method 2 (Advanced): Ensemble Technique (Random Forest Classifier)**
  * Captured complex, non-linear team dynamics and interaction effects, outperforming the linear baseline in accuracy and loss reduction.

---

## 💡 Final Business Impact
* **Stops Losing Money:** AI-driven dynamic pricing prevents human calculation errors and stops smart bettors from exploiting mispriced odds.
* **Guarantees Steady Profits:** Automatically embeds a reliable profit margin (vigorish) into every market, securing consistent long-term revenue.
* **Low Latency & Live Betting:** Sub-second model execution enables real-time odds updates during live matches, unlocking new micro-betting revenue streams.

---

## 🛠️ Tech Stack
* **Language:** Python
* **Libraries:** Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn
* **Version Control:** Git & GitHub
