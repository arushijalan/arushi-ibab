# Implement batch normalization from scratch. and layer normalization for training deep networks.

import numpy as np

# Activation & Loss Utilities
def relu(x):
    return np.maximum(0, x)

def relu_grad(x):
    return (x > 0).astype(x.dtype)

def mse_loss(pred, target):
    loss = np.mean((pred - target) ** 2)
    grad = 2 * (pred - target) / target.size
    return loss, grad

# Batch Normalization 
def batchnorm_init(num_features, eps=1e-5, momentum=0.9):
    params = {
        "gamma": np.ones((num_features,)),
        "beta": np.zeros((num_features,)),
        "running_mean": np.zeros((num_features,)),
        "running_var": np.ones((num_features,)),
        "eps": eps,
        "momentum": momentum
    }
    return params


def batchnorm_forward(x, params, training=True, cache=None):
    gamma, beta = params["gamma"], params["beta"]
    eps, momentum = params["eps"], params["momentum"]

    if training:
        mean = np.mean(x, axis=0)
        var = np.var(x, axis=0)
        std = np.sqrt(var + eps)
        x_norm = (x - mean) / std
        out = gamma * x_norm + beta

        # Update running stats
        params["running_mean"] = momentum * params["running_mean"] + (1 - momentum) * mean
        params["running_var"] = momentum * params["running_var"] + (1 - momentum) * var

        cache = (x, x_norm, mean, var, std, gamma)
    else:
        # Use running stats for inference
        x_norm = (x - params["running_mean"]) / np.sqrt(params["running_var"] + eps)
        out = gamma * x_norm + beta
        cache = None

    return out, cache


def batchnorm_backward(dout, cache):
    """Backward pass for batch normalization."""
    x, x_norm, mean, var, std, gamma = cache
    N, D = x.shape

    dgamma = np.sum(dout * x_norm, axis=0)
    dbeta = np.sum(dout, axis=0)

    dx_norm = dout * gamma
    sum_dxnorm = np.sum(dx_norm, axis=0)
    sum_dxnorm_xnorm = np.sum(dx_norm * x_norm, axis=0)

    dx = (1.0 / N) * (1.0 / std) * (N * dx_norm - sum_dxnorm - x_norm * sum_dxnorm_xnorm)
    return dx, dgamma, dbeta

# Layer Normalization 
def layernorm_init(num_features, eps=1e-5):
    """Initialize parameters for layer normalization."""
    params = {
        "gamma": np.ones((num_features,)),
        "beta": np.zeros((num_features,)),
        "eps": eps
    }
    return params


def layernorm_forward(x, params, cache=None):
    """Forward pass for layer normalization."""
    gamma, beta, eps = params["gamma"], params["beta"], params["eps"]
    mean = np.mean(x, axis=1, keepdims=True)
    var = np.var(x, axis=1, keepdims=True)
    std = np.sqrt(var + eps)
    x_norm = (x - mean) / std
    out = gamma * x_norm + beta
    cache = (x, x_norm, mean, var, std, gamma)
    return out, cache


def layernorm_backward(dout, cache):
    """Backward pass for layer normalization."""
    x, x_norm, mean, var, std, gamma = cache
    N, D = x.shape

    dgamma = np.sum(dout * x_norm, axis=0)
    dbeta = np.sum(dout, axis=0)

    dx_norm = dout * gamma
    sum_dxnorm = np.sum(dx_norm, axis=1, keepdims=True)
    sum_dxnorm_xnorm = np.sum(dx_norm * x_norm, axis=1, keepdims=True)

    dx = (1.0 / D) * (1.0 / std) * (D * dx_norm - sum_dxnorm - x_norm * sum_dxnorm_xnorm)
    return dx, dgamma, dbeta

def main():
    np.set_printoptions(precision=4, suppress=True)

    # Example input: 4 samples, 3 features
    X = np.array([
        [1.0, 2.0, 3.0],
        [2.0, 3.0, 4.0],
        [3.0, 4.0, 5.0],
        [4.0, 5.0, 6.0]
    ])
    # Example gradient (from next layer)
    grad_out = np.array([
        [0.1, 0.2, 0.3],
        [0.2, 0.1, 0.4],
        [0.3, 0.4, 0.1],
        [0.5, 0.2, 0.2]
    ])
    lr = 0.01

    print("Batch Normalization Demo")
    bn_params = batchnorm_init(num_features=3)
    bn_out, bn_cache = batchnorm_forward(X, bn_params, training=True)
    print("Forward Output:\n", bn_out)

    dx_bn, dgamma_bn, dbeta_bn = batchnorm_backward(grad_out, bn_cache)
    print("\nBackward Gradients:")
    print("dx =", dx_bn)
    print("dgamma =", dgamma_bn)
    print("dbeta  =", dbeta_bn)

    # Update params (simple SGD)
    bn_params["gamma"] -= lr * dgamma_bn
    bn_params["beta"] -= lr * dbeta_bn
    print("\nUpdated gamma:", bn_params["gamma"])
    print("Updated beta:", bn_params["beta"])

    print("\nLayer Normalization Demo")
    ln_params = layernorm_init(num_features=3)
    ln_out, ln_cache = layernorm_forward(X, ln_params)
    print("Forward Output:\n", ln_out)

    dx_ln, dgamma_ln, dbeta_ln = layernorm_backward(grad_out, ln_cache)
    print("\nBackward Gradients:")
    print("dx =", dx_ln)
    print("dgamma =", dgamma_ln)
    print("dbeta  =", dbeta_ln)

    # Update params
    ln_params["gamma"] -= lr * dgamma_ln
    ln_params["beta"] -= lr * dbeta_ln
    print("\nUpdated gamma:", ln_params["gamma"])
    print("Updated beta:", ln_params["beta"])


if __name__ == "__main__":
    main()
