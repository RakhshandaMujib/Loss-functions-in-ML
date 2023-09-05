# Understanding Loss Functions in Machine Learning

Loss functions are a critical component of machine learning models. They quantify how well a model's predictions match the actual target values, serving as the basis for model training and optimization. In this guide, we'll explore various loss functions commonly used in machine learning, their mathematical definitions, their suitability for different tasks, and when to use them.

### 1. Mean Squared Error (MSE) Loss

The Mean Squared Error (MSE) loss function is defined as:

$$ \text{mean-squared-error}(y, \hat{y}) = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y_i})^2 $$

- **Suitable for:** Regression tasks
- **Advantage:** Sensitive to the magnitude of errors.

### 2. Mean Absolute Error (MAE) Loss

The Mean Absolute Error (MAE) loss function is defined as:

$$ \text{mean-absolute-error}(y, \hat{y}) = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y_i}| $$

- **Suitable for:** Regression tasks
- **Advantage:** Robust to outliers.

### 3. Huber Loss

The Huber loss function is a combination of MSE and MAE, defined as:

$$ \text{huber}(y, \hat{y}) = \begin{cases} 
\frac{1}{2}(y - \hat{y})^2 & \text{if } |y - \hat{y}| \leq \delta \\
\delta(|y - \hat{y}| - \frac{1}{2}\delta) & \text{otherwise}
\end{cases} $$

- **Suitable for:** Regression tasks, robust to outliers
- **Advantage:** Smooth transition between MAE and MSE.

### 4. Binary Cross-Entropy (BCE or log) Loss

The Binary Cross-Entropy (BCE) loss function is used for binary classification:

$$ \text{log-loss}(y, \hat{y}) = -\frac{1}{n}\sum_{i=1}^{n}(y_i \log(\hat{y_i}) + (1-y_i)\log(1-\hat{y_i})) $$

- **Suitable for:** Binary classification tasks
- **Advantage:** Measures the dissimilarity between predicted probabilities and true labels.

### 5. Categorical Cross-Entropy (CCE) Loss

The Categorical Cross-Entropy (CCE) loss function is used for multi-class classification:

$$ \text{categorical-cross-entropy}(y, \hat{y}) = -\frac{1}{n}\sum_{i=1}^{n}\sum_{j=1}^{K}y_{ij} \log(\hat{y_{ij}}) $$

- **Suitable for:** Multi-class classification tasks
- **Advantage:** Measures the dissimilarity between predicted class probabilities and true class labels.

### 6. Sparse Categorical Cross-Entropy (SCCE) Loss

The Sparse Categorical Cross-Entropy (SCCE) loss function is used for multi-class classification with integer target labels (not one-hot encoded):

$$ \text{sparse-categorical-loss}(y_{\text{index}}, \hat{y}) = -\log(\hat{y}_{\text{index}}) $$

- **Suitable for:** Multi-class classification tasks with integer target labels.
- **Advantage:** Measures the dissimilarity between predicted class probabilities and true class labels without the need for one-hot encoding.

---

## Usage

You can use these loss functions in your machine learning projects to quantify the performance of your models and guide their training. Each loss function is designed for specific tasks and scenarios, so choose the one that best suits your problem.

```python
from loss_functions import RegressionLoss, ClassificationLoss

# Calculate MSE loss
mse_loss = RegressionLoss.mean_squared_error(true_values, predicted_values)

# Calculate BCE loss
bce_loss = ClassificationLoss.binary_cross_entropy(true_labels, predicted_probs)

