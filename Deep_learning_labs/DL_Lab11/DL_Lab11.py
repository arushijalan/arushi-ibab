# Implement convolution operations from scratch. 
# Assume a 3x3 kernel and apply it on an input image of 32x32.
# Implement maxpool operation from scratch. 

import numpy as np


def add_zero_padding(matrix, pad_size=1):

    height, width = matrix.shape
    left_pad = np.zeros((height, pad_size))
    right_pad = np.zeros((height, pad_size))
    matrix = np.hstack([left_pad, matrix, right_pad])

    new_width = matrix.shape[1]
    top_pad = np.zeros((pad_size, new_width))
    bottom_pad = np.zeros((pad_size, new_width))

    padded_matrix = np.vstack([top_pad, matrix, bottom_pad])
    return padded_matrix


def perform_convolution(input_tensor, kernel_bank, padding=1, stride=1):

    channels, height, width = input_tensor.shape
    num_filters, _, kernel_size, _ = kernel_bank.shape

    # Apply padding per channel
    padded_tensor = np.array([add_zero_padding(input_tensor[c], padding) for c in range(channels)])
    padded_height, padded_width = padded_tensor.shape[1], padded_tensor.shape[2]

    # Compute output dimensions
    out_height = (padded_height - kernel_size) // stride + 1
    out_width = (padded_width - kernel_size) // stride + 1

    conv_output = np.zeros((num_filters, out_height, out_width))

    # Perform convolution
    for f in range(num_filters):
        for i in range(0, out_height * stride, stride):
            for j in range(0, out_width * stride, stride):
                region = padded_tensor[:, i:i+kernel_size, j:j+kernel_size]
                conv_output[f, i//stride, j//stride] = np.sum(region * kernel_bank[f])

    return conv_output


def perform_max_pooling(feature_maps, pool_size=2, stride=2):

    channels, height, width = feature_maps.shape
    out_height = (height - pool_size) // stride + 1
    out_width = (width - pool_size) // stride + 1

    pooled_output = np.zeros((channels, out_height, out_width))

    for c in range(channels):
        for i in range(0, out_height * stride, stride):
            for j in range(0, out_width * stride, stride):
                region = feature_maps[c, i:i+pool_size, j:j+pool_size]
                pooled_output[c, i//stride, j//stride] = np.max(region)

    return pooled_output


def main():

    np.random.seed(0)

    # Create a random RGB image (3 channels, 32x32)
    input_image = np.random.randint(0, 255, size=(3, 32, 32))

    # Define two 3x3 filters for 3 input channels
    kernel_bank = np.array([
        [[[1, 0, -1],
          [1, 0, -1],
          [1, 0, -1]],

         [[1, 0, -1],
          [1, 0, -1],
          [1, 0, -1]],

         [[1, 0, -1],
          [1, 0, -1],
          [1, 0, -1]]],

        [[[-1, -1, -1],
          [0, 0, 0],
          [1, 1, 1]],

         [[-1, -1, -1],
          [0, 0, 0],
          [1, 1, 1]],

         [[-1, -1, -1],
          [0, 0, 0],
          [1, 1, 1]]]
    ])

    # Perform convolution
    conv_result = perform_convolution(input_image, kernel_bank, padding=1, stride=1)
    print("Convolution output shape:", conv_result.shape)

    # Perform max pooling
    pooled_result = perform_max_pooling(conv_result, pool_size=2, stride=2)
    print("Max pooling output shape:", pooled_result.shape)


if __name__ == "__main__":
    main()
