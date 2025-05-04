# Machine Learning Exercises

This repository contains a Jupyter notebook (`machine learning.ipynb`) that implements and explores several fundamental machine learning concepts and algorithms across multiple problems. The notebook is organized into five problems, covering classification, independence, Naive Bayes, maximum likelihood estimation, and linear regression.

## Contents

### Problem I: Classification on Continuous Data

- **Question 1**: Visualization of three continuous datasets (`clsf-a.csv`, `clsf-b.csv`, `clsf-c.csv`) using scatter plots of features **X1** vs. **X2**, with class labels **Y = +1** (blue) and **Y = -1** (red).
- **Question 2**: Training and evaluating three classifiers on each dataset:
  - **Decision Tree** (entropy criterion, max depth=2, random state=42)
  - **Gaussian Naive Bayes**
  - **Logistic Regression**
- Reporting classification accuracy for each classifier on each dataset.

### Problem II: Feature Independence

- **Question 6**: Computing joint probability distribution \(P(X1, X2)\) from a nominal dataset (`indep.csv`).
- **Question 7**: Calculating marginal distributions \(P(X1)\) and \(P(X2)\).

### Problem III: Naive Bayes Classifier for Discrete Data

- **Question 9**: Estimation of class-prior probabilities \(P(Y)\) from `nb.csv`.
- **Question 10**: Computing class-conditional probabilities \(P(X1\mid Y)\) and \(P(X2\mid Y)\).
- **Question 11**: Classification of a sample using the computed Naive Bayes probabilities.

### Problem IV: Maximum Likelihood Estimation

- **Question 12**: Derivation of the log-likelihood function for an exponential distribution.
- **Question 13**: Derivation of the maximum likelihood estimator (MLE) for the rate parameter \(\lambda\).
- **Question 14**: Computing \(\hat{\lambda}\) on a real dataset (`exp.csv`).

### Problem V: Linear Regression and Model Complexity

- Using datasets `regr-1.csv`, `regr-2.csv`, and `regr-3.csv`, each with continuous features **X1**, **X2** and label **Y**.
- Training **Ordinary Least Squares** linear regression models (**no intercept**) with different feature subsets (**X1 & X2**, **X1 only**, **X2 only**, **baseline Y=0**).
- Evaluating and reporting **Mean Squared Error (MSE)** on training and test datasets to observe the effects of sample size and feature choice.

## Requirements

- Python 3.7 or higher
- Jupyter Notebook or JupyterLab
- Packages:
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `scikit-learn`

Install dependencies via pip:

```bash
pip install numpy pandas matplotlib scikit-learn
```


