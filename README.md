# Understanding Loss Functions in Machine Learning

Loss functions are a critical component of machine learning models. They quantify how well a model's predictions match the actual target values, serving as the basis for model training and optimization. In this guide, we'll explore various loss functions commonly used in machine learning, their mathematical definitions, their suitability for different tasks, and when to use them. 

**Note on Terminology:**
In the realm of Machine Learning, the terms "loss" and "cost" are often used interchangeably. However, adhering to the distinction outlined by Andrew Ng, it's valuable to clarify their roles: 

- We compute the "loss" for an individual example (i denoted by the subscript 'i'), representing how far off the model's prediction ($$a_i$$) is from the true target ($$y_i$$) for that single instance.
- The "cost" is computed for the entire training dataset, usually by taking the mean (average) of all the individual losses.

The methods provided in this repository are designed to calculate the "loss" for each training example. To determine the "cost" for your model, you can easily obtain it by calculating the mean of all these individual losses. This clarification ensures consistency in terminology and its usage throughout the library.


### 1. Squared Error Loss

The Squared Error loss function is defined as:

$$ \text{squared-error}(y_i, a_i) = (y_i - a_i)^2 $$

- **Suitable for:** Regression tasks
- **Advantage:** Sensitive to the magnitude of errors.

### 2. Absolute Error Loss

The Absolute Error loss function is defined as:

$$ \text{absolute-error}(y_i, a_i) = |y_i - a_i| $$

- **Suitable for:** Regression tasks
- **Advantage:** Robust to outliers.

### 3. Huber Loss

The Huber loss function is a combination of MSE and MAE, defined as:

$$ \text{huber}(y_i, a_i) = \begin{cases} 
\frac{1}{2}(y_i - a_i)^2 & \text{if } |y_i - a_i| \leq \delta \\
\delta(|y_i - a_i| - \frac{1}{2}\delta) & \text{otherwise}
\end{cases} $$

- **Suitable for:** Regression tasks, robust to outliers
- **Advantage:** Smooth transition between MAE and MSE.

### 4. Binary Cross-Entropy (BCE or log) Loss

The Binary Cross-Entropy (BCE) loss function is used for binary classification:

$$ \text{log-loss}(y_i, a_i) = -\frac{1}{n}\sum_{i=1}^{n}(y_i \log(a_i) + (1-y_i)\log(1-a_i)) $$

- **Suitable for:** Binary classification tasks
- **Advantage:** Measures the dissimilarity between predicted probabilities and true labels.

### 5. Categorical Cross-Entropy (CCE) Loss

The Categorical Cross-Entropy (CCE) loss function is used for multi-class classification:

$$ \text{categorical-cross-entropy}(y_i, a_i) = -\frac{1}{n}\sum_{i=1}^{n}\sum_{j=1}^{K}y_{ij} \log(a_{ij}) $$

- **Suitable for:** Multi-class classification tasks
- **Advantage:** Measures the dissimilarity between predicted class probabilities and true class labels.

### 6. Sparse Categorical Cross-Entropy (SCCE) Loss

The Sparse Categorical Cross-Entropy (SCCE) loss function is used for multi-class classification with integer target labels (not one-hot encoded):

$$ \text{sparse-categorical-loss}(y_{\text{index}}, a_i) = -\log(a_{i_{\text{index}}}) $$

- **Suitable for:** Multi-class classification tasks with integer target labels.
- **Advantage:** Measures the dissimilarity between predicted class probabilities and true class labels without the need for one-hot encoding.

---

## Usage

You can use these loss functions in your machine learning projects to quantify the performance of your models and guide their training. Each loss function is designed for specific tasks and scenarios, so choose the one that best suits your problem.

```python
from loss_functions import RegressionLoss, ClassificationLoss

# Calculate MSE loss
mse_loss = np.mean(RegressionLoss.squared_error(true_values, predicted_values))

# Calculate BCE loss
bce_loss = np.mean(ClassificationLoss.binary_cross_entropy(true_labels, predicted_probs))

---

Remove the hat and write a_i instead
