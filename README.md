This repository presents the illustrations in the paper "A Comprehensive Survey on Imbalanced Regression: Definitions, Solutions, and Future Directions". The notebook used to obtain the results in the paper is made available and the protocol is explained below.


## Experimental Protocol
Imbalanced regression is inherently influenced by multiple factors, including the level of imbalance, sample size, and the complexity of the regression task. This section aims to illustrate and analyze the phenomenon through a series of simulations by varying these characteristics.
We assess the sensitivity of predictive performance to the following factors:
- **Sample size** — 5 ordered levels: [200, 500, 1000, 2000, 5000]  
- **Regression complexity** — 5 unordered levels: [1, 2, 3, 4, 5]. Details below 
- **Imbalance level** — 5 ordered levels: [0.5, 0, –0.5, –1, –1.5]. Details below

To mitigate random variation, each configuration is evaluated across 5 independent runs using different random seeds. To ensure model-agnostic conclusions, we apply 10 models from the H2O AutoML library, covering a range of algorithms such as Distributed Random Forest, Extremely Randomized Trees, Regularized Generalized Linear Models, Gradient Boosting Machines, Extreme Gradient Boosting, and Multi-layer Feedforward Neural Networks.
For each configuration, a dataset of size $n = 10,000$ is generated depending solely on the complexity level. A balanced test set of 1,000 observations (approximately uniform) is drawn. From the remaining 9,000 instances, we sample two training sets:
- A balanced training set (baseline), serving as an ideal reference.
- An imbalanced training set, varying according to both sample size and imbalance level. A level of 0.5 indicates balance, while –1.5 represents strong imbalance.
Each dataset includes 5 Gaussian features and 5 non-linear transformations of them. The target variable $Y \sim \mathcal{N}(\mu, \sigma)$ is constructed using five selected features. At complexity level 1, these are the original Gaussian variables (yielding a simple regression task). At level 5, only the non-linear features are used, making the task highly complex. Intermediate levels (2–4) use a mix of both.

### Regression complexity
Here is the pseudo-algorithm to simulate a synthetic dataset with controlled complexity
Bien sûr ! Voici la **version Markdown** parfaitement adaptée pour intégrer dans un dépôt GitHub (ou tout autre rendu Markdown). Elle est claire, lisible et conforme aux bonnes pratiques académiques :

---

### 🔧 Synthetic Dataset Generation with Controlled Complexity

We simulate a synthetic dataset to evaluate model performance under different levels of functional complexity. Let:

* $n$: the number of observations
* $c \in \{1, 2, 3, 4, 5\}$: the complexity level
* All features $X_j$ are normalized using min-max scaling (denoted as $\tilde{X}_j$)

#### **1. Input Feature Generation**

Each observation is composed of the following variables:

```math
X_0, X_1, X_2, X_3, X_4 \sim \mathcal{N}(0, 1)
```

```math
X_5 \sim \mathcal{N}\left(\frac{\exp(2X_0)}{1 + \exp(2X_0)}, \, 0.01^2\right)
```

```math
X_6 \sim \mathcal{N}(\sin(2X_1), 0.1^2), \quad
X_7 \sim \mathcal{N}(\cos(2X_2), 0.1^2)
```

```math
X_8 \sim \mathcal{N}(X_3^2, 0.5^2), \quad
X_9 \sim \mathcal{N}(X_4^2, 0.5^2)
```

#### **2. Min-Max Normalization**

Each variable $X_j$ is scaled:

```math
\tilde{X}_j = \frac{X_j - \min(X_j)}{\max(X_j) - \min(X_j)}
```

#### **3. Linear Predictor Function Definition Based on Complexity Level $c$**

```math
\text{PL} =
\begin{cases}
\tilde{X}_0 + \tilde{X}_1 + \tilde{X}_2 + \tilde{X}_3 + \tilde{X}_4 & \text{if } c = 1 \\
\tilde{X}_5 + \tilde{X}_1 + \tilde{X}_2 + \tilde{X}_3 + \tilde{X}_4 & \text{if } c = 2 \\
\tilde{X}_5 + \tilde{X}_6 + \tilde{X}_7 + \tilde{X}_3 + \tilde{X}_4 & \text{if } c = 3 \\
\tilde{X}_5 + \tilde{X}_6 + \tilde{X}_7 + 2\tilde{X}_8 + 2\tilde{X}_9 & \text{if } c = 4 \\
\tilde{X}_5 + \tilde{X}_6 \cdot \tilde{X}_7 + \tilde{X}_8 \cdot \tilde{X}_9 & \text{if } c = 5 \\
\end{cases}
```

#### **4. Target Variable Generation**

```math
Y \sim \mathcal{N}(\text{PL}, \, 0.1^2)
```

#### **5. Output**

The final dataset contains $n$ samples of the form:

```math
\mathcal{D} = \{(Y^{(i)}, X_0^{(i)}, \dots, X_9^{(i)}) \}_{i=1}^{n}
```







### Imbalance level
The level of imbalance in the training sample is managed by a train-test splitting controlled by an imbalance force.
This procedure creates **train/test subsets** from a dataset while allowing **imbalance control** in the training subset through **importance weighting** based on the target variable.

---

#### **1. Parameters**

* `data`: Input dataset, including both the target and features
* `test-size` ∈ (0, 1): Proportion of the dataset allocated to the test set
* `train-size` ∈ (0, 1): Proportion allocated to the training subset (optional)
* `w-test`: Sampling distribution for the test set (default: uniform)
* `imbForce` ≥ 0: Controls how much importance is given to rare values in the training set
* `np-seed`: Seed for reproducibility

---

#### **2. Test Set Sampling**

Let *n* be the number of observations. Define:

> *n<sub>test</sub>* = round(*n* × `test-size`)

Then draw *n<sub>test</sub>* indices from the dataset using the sampling weights *w<sub>test</sub>*:

> Test indices ∼ Multinomial(*n<sub>test</sub>*, *w<sub>test</sub>*)

The test set is then formed from the selected indices, and the remaining data is assigned to the training pool.

---

#### **3. Training Set Sampling (Optional)**

If `train-size` is provided, two training subsets are built:

##### • (a) Rebalanced Training Set (`X-train`)

To emphasize rare values, a **relevance-based sampling distribution** is computed via the `IR_weighting` function:

> **IR Weight Function:**
> For a target value *y*, the weight is defined as:
>
> $w(y) = \frac{1}{\hat{f}(y)^{\alpha}} \Big/ \sum_{i=1}^{n} \frac{1}{\hat{f}(y_i)^{\alpha}}$
>
> where $\hat{f}(y)$ is the kernel density estimate (KDE) of the target variable, and $\alpha$ controls the strength of emphasis on rare values.

Then, the training sample is drawn from this relevance-weighted distribution:

> Train indices ∼ Multinomial(*n<sub>train</sub>*, *w<sub>imb</sub>*)

##### • (b) Baseline Balanced Training Set (`X-bal`)

A second sample of equal size is drawn from the uniform weights (same as `w-test`):

> Train indices ∼ Multinomial(*n<sub>train</sub>*, *w<sub>test</sub>*)

---

#### **4. Output**

The function returns the following objects:

* `X-train`: Training set sampled using relevance-weighted distribution
* `X-test`: Randomly sampled test set
* `X-bal`: Baseline training set sampled using uniform distribution









