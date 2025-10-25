# Implement various update rules used to optimize the neural network. 
# You can use PyTorch for the following implementations.
# SGD, Momentum, AdaGrad, etc.

import torch

def sgd_update(parameters, learning_rate):
    for param in parameters:
        if param.grad is not None:
            param.data -= learning_rate * param.grad

def sgd_momentum_update(parameters, velocities, learning_rate, momentum):
    for i, param in enumerate(parameters):
        if param.grad is not None:
            velocities[i] = momentum * velocities[i] - learning_rate * param.grad
            param.data += velocities[i]


def adagrad_update(parameters, grad_accum, learning_rate, epsilon=1e-8):
    for i, param in enumerate(parameters):
        if param.grad is not None:
            grad_accum[i] += param.grad ** 2
            adjusted_lr = learning_rate / (torch.sqrt(grad_accum[i]) + epsilon)
            param.data -= adjusted_lr * param.grad


def rmsprop_update(parameters, avg_sq_grad, learning_rate, beta=0.9, epsilon=1e-8):
    for i, param in enumerate(parameters):
        if param.grad is not None:
            avg_sq_grad[i] = beta * avg_sq_grad[i] + (1 - beta) * (param.grad ** 2)
            param.data -= learning_rate * param.grad / (torch.sqrt(avg_sq_grad[i]) + epsilon)


def adam_update(parameters, m_vals, v_vals, step, learning_rate, betas=(0.9, 0.999), epsilon=1e-8):
    beta1, beta2 = betas
    step += 1
    for i, param in enumerate(parameters):
        if param.grad is not None:
            m_vals[i] = beta1 * m_vals[i] + (1 - beta1) * param.grad
            v_vals[i] = beta2 * v_vals[i] + (1 - beta2) * (param.grad ** 2)

            m_hat = m_vals[i] / (1 - beta1 ** step)
            v_hat = v_vals[i] / (1 - beta2 ** step)

            param.data -= learning_rate * m_hat / (torch.sqrt(v_hat) + epsilon)
    return step


def zero_gradients(parameters):
    for param in parameters:
        if param.grad is not None:
            param.grad.zero_()

def main():
    # Dummy dataset and simple linear model
    inputs = torch.randn(30, 10)
    targets = torch.randn(30, 1)
    model = torch.nn.Linear(10, 1)
    loss_fn = torch.nn.MSELoss()

    # Hyperparameters
    lr = 0.01
    momentum = 0.9
    beta = 0.9
    betas = (0.9, 0.999)
    eps = 1e-8
    num_epochs = 5

    params = list(model.parameters())

    # Initialize optimizer-specific states
    velocities = [torch.zeros_like(p.data) for p in params]
    grad_accum = [torch.zeros_like(p.data) for p in params]
    avg_sq_grad = [torch.zeros_like(p.data) for p in params]
    m_vals = [torch.zeros_like(p.data) for p in params]
    v_vals = [torch.zeros_like(p.data) for p in params]
    step_count = 0

    # Training loop using chosen optimizer
    for epoch in range(num_epochs):
        preds = model(inputs)
        loss = loss_fn(preds, targets)

        zero_gradients(params)
        loss.backward()

        step_count = adam_update(params, m_vals, v_vals, step_count, lr, betas, eps)

        print(f"Epoch {epoch + 1}: Loss = {loss.item():.4f}")


if __name__ == "__main__":
    main()
