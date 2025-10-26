# quantile_normalization.py
import numpy as np

def quantile_normalize(matrix):
    
    sorted_indices = np.argsort(matrix, axis=0)
    sorted_matrix = np.sort(matrix, axis=0)
    row_means = np.mean(sorted_matrix, axis=1)
    normalized_matrix = np.zeros_like(matrix)

    for col_index in range(matrix.shape[1]):
        normalized_matrix[sorted_indices[:, col_index], col_index] = row_means

    return normalized_matrix


def main():
    # Load input matrix
    input_data = np.load("1000G_float64.npy")
    print(f"Original data shape: {input_data.shape}")

    # Split into landmark and target gene sections
    landmark_genes = input_data[:943, :]
    target_genes = input_data[943:, :]

    # Apply quantile normalization
    normalized_landmark = quantile_normalize(landmark_genes)
    normalized_target = quantile_normalize(target_genes)

    # Merge normalized data
    normalized_full_matrix = np.vstack((normalized_landmark, normalized_target))
    print(f"Final normalized matrix shape: {normalized_full_matrix.shape}")

    # Save the normalized data
    np.save("1000G_reqnorm_float64.npy", normalized_full_matrix)
    print("Saved normalized data as 1000G_reqnorm_float64.npy")


if __name__ == "__main__":
    main()
