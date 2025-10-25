# Implement dropout to regularize neural networks from scratch.

import numpy as np

def dropout_forward(inputs, drop_rate=0.3, training=True):

    if training:
        batch_size, num_features = inputs.shape
        dropout_mask = (np.random.rand(batch_size, num_features) > drop_rate).astype(np.float32)
        output = inputs * dropout_mask / (1 - drop_rate)
    else:
        dropout_mask = np.ones_like(inputs)
        output = inputs
    return output, dropout_mask


def dropout_backward(grad_output, dropout_mask, drop_rate=0.3):
    grad_input = grad_output * dropout_mask / (1 - drop_rate)
    return grad_input

def main():
    # Example input (2 samples, 3 neurons)
    activations = np.array([
        [8.0, 5.0, 3.0],
        [4.0, 1.0, 5.0]
    ])

    grad_from_next = np.ones_like(activations)
    drop_rate = 0.3

    # Forward pass (training)
    out_train, mask_train = dropout_forward(activations, drop_rate, training=True)
    print("Dropout Forward (Training Mode)")
    print(out_train)

    # Forward pass (inference)
    out_eval, _ = dropout_forward(activations, drop_rate, training=False)
    print("\nDropout Forward (Inference Mode)")
    print(out_eval)

    # Backward pass
    grad_input = dropout_backward(grad_from_next, mask_train, drop_rate)
    print("\nDropout Backward Gradient")
    print(grad_input)


if __name__ == "__main__":
    main()
