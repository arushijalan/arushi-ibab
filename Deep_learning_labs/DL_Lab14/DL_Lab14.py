# Implement vanilla RNN from scratch  
# https://medium.com/@thisislong/building-a-recurrent-neural-network-from-scratch-ba9b27a42856 



import numpy as np
import matplotlib.pyplot as plt


def tanh_activation(z):
    return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))


def tanh_derivative(z):
    return 1 - np.power(tanh_activation(z), 2)


def visualize_tanh():
    z_values = np.linspace(-8, 8, 200)
    tanh_vals = tanh_activation(z_values)
    derivative_vals = tanh_derivative(z_values)

    plt.plot(z_values, tanh_vals, label="tanh(z)")
    plt.plot(z_values, derivative_vals, label="tanh'(z)")
    plt.legend()
    plt.title("Tanh Activation Function and Its Derivative")
    plt.xlabel("z")
    plt.ylabel("Value")
    plt.grid(True)
    plt.show()

    avg = np.mean(tanh_vals)
    print(f"Average tanh value: {avg:.4f}")
    if np.isclose(avg, 0, atol=1e-2):
        print("The tanh function is approximately zero-centered.")
    else:
        print("The tanh function is not zero-centered.")


def rnn_forward_pass(inputs, hidden_states, weight_input_hidden, weight_hidden_hidden, weight_hidden_output):
    sequence_length = len(inputs)
    outputs = []

    for t in range(sequence_length):
        current_input = np.array(inputs[t]).reshape(-1, 1)
        previous_hidden = np.array(hidden_states[-1]).reshape(-1, 1)

        # Compute hidden state
        hidden_linear = np.matmul(weight_hidden_hidden, previous_hidden) + np.matmul(weight_input_hidden, current_input)
        current_hidden = tanh_activation(hidden_linear).flatten()
        hidden_states.append(current_hidden)

        # Compute output
        current_output = np.matmul(weight_hidden_output, current_hidden)
        outputs.append(current_output)

        print(f"Time step {t + 1}:")
        print(f"  Input: {inputs[t]}")
        print(f"  Hidden state (h{t + 1}): {np.round(current_hidden, 3)}")
        print(f"  Output (y{t + 1}): {np.round(current_output, 3)}\n")

    return outputs, hidden_states


def main():
    print("Visualizing tanh and its derivative")
    visualize_tanh()

    print("\nRunning Vanilla RNN Forward Pass")
    # Define inputs and weights
    input_sequence = [[1, 2], [-1, 1], [2, 3]]
    initial_hidden_state = [[0, 0, 0]]

    weight_input_hidden = np.array([[0.5, -0.3],
                                    [0.8, 0.2],
                                    [0.1, 0.4]])

    weight_hidden_hidden = np.array([[0.1, 0.4, 0.0],
                                     [-0.2, 0.3, 0.2],
                                     [0.05, -0.1, 0.2]])

    weight_hidden_output = np.array([[1, -1, 0.5],
                                     [0.5, 0.5, -0.5]])

    # Run forward pass
    outputs, hidden_states = rnn_forward_pass(
        inputs=input_sequence,
        hidden_states=initial_hidden_state,
        weight_input_hidden=weight_input_hidden,
        weight_hidden_hidden=weight_hidden_hidden,
        weight_hidden_output=weight_hidden_output
    )

    print("Final Outputs:\n", np.round(outputs, 3))
    print("Final Hidden States:\n", np.round(hidden_states[-1], 3))


if __name__ == "__main__":
    main()
