# Introduction to Statistics

## Index
- [Introduction to Statistics](#introduction-to-statistics)
  - [Index](#index)
  - [1. Data Types and Variables](#1-data-types-and-variables)
    - [Data Classification](#data-classification)
    - [Variable Categories](#variable-categories)
  - [2. Descriptive Statistics](#2-descriptive-statistics)
    - [Measures of Dispersion](#measures-of-dispersion)
      - [Variance](#variance)
      - [Standard Deviation](#standard-deviation)
    - [Measures of Position (Quantiles)](#measures-of-position-quantiles)
      - [Quartiles](#quartiles)
  - [3. Probability Functions](#3-probability-functions)
    - [Probability Mass Functions (PMFs)](#probability-mass-functions-pmfs)
      - [Bernoulli distribution](#bernoulli-distribution)
      - [Binomial distribution](#binomial-distribution)
      - [Poisson distribution](#poisson-distribution)
      - [Geometric distribution](#geometric-distribution)
      - [Hypergeometric distribution](#hypergeometric-distribution)
      - [Negative Binomial distribution *(optional)*](#negative-binomial-distribution-optional)

---

## 1. Data Types and Variables

### Data Classification
* **Qualitative:** Categorical properties (e.g., eye color, fruit type).
* **Quantitative:** Properties that can be counted or measured (e.g., age, blood pressure).

### Variable Categories
* **Nominal Qualitative:** Values segregate the population into mutually exclusive, unordered categories (e.g., marital status).
* **Ordinal Qualitative:** Values possess a natural, meaningful ordering (e.g., patient condition: good, mild, severe).
* **Discrete Quantitative:** Variables that take distinct, countable values within a real interval (e.g., number of cars owned).
* **Continuous Quantitative:** Variables that can take any real value within an interval (e.g., weight, height).

---

## 2. Descriptive Statistics

### Measures of Dispersion

#### Variance

* **Population Variance:** Used when analyzing the complete population ($N$).
$$\text{var}(x) = \sigma^2 = \frac{1}{N} \sum_{i=1}^{N} (x_i - \mu)^2$$
where $\mu$ is the population mean.

* **Sample Variance:** Uses Bessel's correction ($N-1$) to provide an unbiased estimator from a sample.
$$\text{var}(x) = s^2 = \frac{1}{N-1} \sum_{i=1}^{N} (x_i - \bar{x})^2$$
where $\bar{x}$ is the sample mean.

#### Standard Deviation
$$\sigma = \sqrt{\text{var}(x)}$$

### Measures of Position (Quantiles)
A **percentile** is the value below which a given percentage of observations falls (e.g., the 80th percentile is the threshold holding 80% of the data below it).

#### Quartiles
Specific percentiles dividing the dataset into four equal parts:
* **$Q_1$ (First quartile):** 25th percentile.
* **$Q_2$ (Second quartile):** 50th percentile, identical to the **median**.
* **$Q_3$ (Third quartile):** 75th percentile.

---
## 3. Probability Functions


### Probability Mass Functions (PMFs) 

Functions that give the probability $P$ that a descrete random variable $X$ is exactly equal to some value (generally $x$ or $k$).

#### Bernoulli distribution

Models a single trial with binary outcomes: success ($x=1$) with probability $p$, or failure ($x=0$) with probability $1-p$.

$$
P(X=x)=p^x(1-p)^{1-x}, \qquad x\in\{0,1\} \equiv \{\rm failure, success\}
$$


#### Binomial distribution

Models the total number of successes $k$ in $n$ independent Bernoulli trials with a constant success probability $p$.

$$P(X=k) = \binom{n}{k} p^k (1-p)^{n-k}, \qquad k \in \{0, 1, \dots, n\}$$


#### Poisson distribution


Models the probability of $k$ events ocurring in a fixed interval when:
- events occur **independently**,
- events occur at a **constant average rate** $\lambda$,
- and the probability of **two or more events occurring simultaneously is negligible**.

$$P(X=k) = \frac{\lambda^k e^{-\lambda}}{k!}, \qquad k \in \{0, 1, 2, \dots\}$$


#### Geometric distribution

Models the number of trials $k$ required to achieve the first success in a sequence of independent Bernoulli trials.

$$P(X=k) = (1-p)^{k-1}p, \qquad k \in \{1, 2, \dots\}$$

#### Hypergeometric distribution 

Models the number of successes $k$ in a sample size $n$ drawn **without replacement** from a finite population $N$ containing $K$ total successes.
$$P(X=k) = \frac{\binom{K}{k}\binom{N-K}{n-k}}{\binom{N}{n}}$$

---

#### Negative Binomial distribution *(optional)*

Models the probability that the \(r\)-th success occurs on the \(k\)-th trial:

$$
P(X=k)=\binom{k-1}{r-1}p^r(1-p)^{k-r}, \qquad k=r,r+1,\ldots
$$






---

[![Basic Statistics PDF](./docs/basic_statistics.pdf)](./docs/basic_statistics.pdf)