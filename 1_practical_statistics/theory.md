# Introduction to Statistics

---
> # Class 1

## Data Types

* Qualitative: categorical properties (e.g. person's eye color, type of fruit, ...)
* Quantitative: properties that can be counted or measured (e.g. age, blood pressure, ...)

## Variable Types

* Nominal Qualitative: the values taken by the variable allow to segregate the population into mutually exclusive categories (e.g. marital status)
* Ordinal Qualitative: the values taken by the variable can be ordered (e.g. the condition of a patient in a hospital: good, mild, severe)
* Discrete Quantitative: the values taken by the variable can only take certain values within (an interval in) the real numbers (e.g. number of cars someone owns)
* Continuous Quantitative: the values taken by the variable can take all possible values within (an interval in) the real numbers (e.g. weight, height)

## Areas of Statistics
* Descriptive: It is dedicated to the description, visualization, and summarization of data originating from the phenomena under study. Its objective is to organize and describe the characteristics of a data set in order to facilitate interpretation.

* Inferential: It is dedicated to the generation of models and predictions associated with the phenomena in question, taking into account the randomness of the observations.

---

> # Class 2

## Variance and Standard Deviation

### Variance 

- If using the total population:

$$\text{var}(x) = \sigma^2 = \frac{1}{N} \; \displaystyle{\sum_{i=1}}(x_i - \mu) $$

where $\mu$ is the mean of the total population.

- If using a sample of the population  (Bessel's correction): 
  
$$\text{var}(x) = s^2 = \frac{1}{N-1} \; \displaystyle{\sum_{i=1}}(x_i - \bar{x}) $$

where $\bar{x}$ is the mean of the sample.


### Standard Deviation

$$\sigma = \sqrt{\rm var}$$


## Percentiles

A percentile is a value below which a given percentage of the observations in a dataset falls.

**Example:**

- 80th percentile: the value below which 80% of the data lie.

### Quartiles

Quartiles are specific percentiles that divide the data into four equal parts:

- **Q1 (First quartile):** 25th percentile (25% of the data lie below it).
- **Q2 (Second quartile):** 50th percentile, also known as the **median**.
- **Q3 (Third quartile):** 75th percentile (75% of the data lie below it).