# Understanding Non-Linear Activation Functions in Neural Networks

Activation functions are a crucial component of artificial neural networks. They introduce non-linearity into the models, allowing them to capture complex relationships in data. In this article, we'll explore various non-linear activation functions commonly used in neural networks, their properties, and when to use them.

## 1. Logistic or Sigmoid Activation

The sigmoid activation function is defined as:

$$
\text{sigmoid}(z) = \frac{1}{1 + e^{-z}}
$$

- **Range:** (0, 1)
- **Common Use:** Often used in the last layer for binary classification, where it maps the linear activation to a probability range.

## 2. Hyperbolic Tangent (tanh) Activation

The hyperbolic tangent activation function is defined as:

\[
\text{softmax}(z)_i = \frac{e^{z_i}}{\sum_{j=1}^{C}e^{z_j}}
\]

- **Range:** (-1, 1)
- **Common Use:** Suitable for hidden layers, it performs better than sigmoid but still suffers from the vanishing gradient problem.

## 3. Rectified Linear Unit (ReLU) Activation

The ReLU activation function is defined as:

$$
\text{relu}(z) = \max(0, z)
$$

- **Range:** [0, ∞)
- **Common Use:** Highly popular for hidden layers due to simplicity and efficiency. However, it can suffer from the "dying ReLU" problem.

## 4. Leaky ReLU Activation

The leaky ReLU activation function is defined as:

$$
\text{leaky-relu}(z) = \max(Lz, z)
$$

- **Range:** (-∞, ∞)
- **Common Use:** Helps solve the dying ReLU problem by introducing a small slope (\(L\)) (aka "leak") for negative inputs.

## 5. Parameterized ReLU (PReLU) Activation

The parameterized ReLU activation function is similar to leaky ReLU but with the slope (\(L\)) learned during training.

$$
\text{prelu}(z) = \max(Lz, z)
$$

- **Range:** (-∞, ∞)

## 6. Exponential Linear Unit (ELU) Activation

The ELU activation function is defined as:

$$
\text{elu}(z) = \max(L(e^z - 1), z)
$$

- **Range:** (-∞, ∞)
- **Advantage:** Smoother than ReLU for negative values.

## 7. Scaled Exponential Linear Unit (SELU) Activation

The SELU activation function is similar to ELU but includes a scaling factor (\(S\)).

$$
\text{selu}(z) = S \cdot \max(L(e^z - 1), z)
$$

- **Range:** (-∞, ∞)
- **Advantage:** Addresses vanishing-exploding gradient problems, but requires specific conditions on network architecture.

## 8. Softmax Activation

The softmax activation function is used in the output layer for multi-class classification:

$$
\text{softmax}(x_{i}) = \frac{\exp(x_{i})}{\sum_{j} \exp(x_{j})}
$$

- **Range:** [0, 1]
- **Common Use:** Converts raw scores (logits) into a probability distribution over multiple classes.
