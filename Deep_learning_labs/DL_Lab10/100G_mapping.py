# Gene Mapping and Data Extraction
import numpy as np

INPUT_FILE = "sample.csv"
LANDMARK_FILE = "map_lm.txt"
TARGET_FILE = "map_tg.txt"
OUTPUT_NPY = "1000G_float64.npy"


def trim_gene_version(gene):
    return gene.split('.')[0]


def read_gene_ids(filepath):
    gene_ids = []
    with open(filepath) as file:
        for line in file:
            if line.strip():
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    gene_ids.append(trim_gene_version(parts[1]))
    return gene_ids


def locate_first_numeric_index(values):
    for i, val in enumerate(values):
        try:
            float(val)
            return i
        except ValueError:
            continue
    return None


def convert_to_float(value):
    try:
        return float(value)
    except ValueError:
        return 0.0


def main():
    gene_names = []
    gene_expression = []

    with open(INPUT_FILE) as file:
        header_line = file.readline().strip().split(",")
        data_start_index = None

        for line in file:
            entries = line.strip().split(",")
            if data_start_index is None:
                data_start_index = locate_first_numeric_index(entries)
                if data_start_index is None:
                    raise ValueError("No numeric columns detected in the input file.")

            gene_identifier = trim_gene_version(entries[0])
            numeric_data = [convert_to_float(v) for v in entries[data_start_index:]]

            gene_names.append(gene_identifier)
            gene_expression.append(numeric_data)

    gene_expression = np.array(gene_expression, dtype=np.float32)

    # Load mapped gene IDs
    landmark_genes = read_gene_ids(LANDMARK_FILE)
    target_genes = read_gene_ids(TARGET_FILE)

    # Build index mapping for gene lookup
    gene_index_map = {gene: idx for idx, gene in enumerate(gene_names)}
    landmark_indices = [gene_index_map[g] for g in landmark_genes if g in gene_index_map]
    target_indices = [gene_index_map[g] for g in target_genes if g in gene_index_map]

    print(f"Total genes in dataset: {len(gene_names)}")
    print(f"Landmark genes found: {len(landmark_indices)}/{len(landmark_genes)}")
    print(f"Target genes found: {len(target_indices)}/{len(target_genes)}")

    if len(landmark_indices) < len(landmark_genes):
        print(f"Missing {len(landmark_genes) - len(landmark_indices)} landmark genes")
    if len(target_indices) < len(target_genes):
        print(f"Missing {len(target_genes) - len(target_indices)} target genes")

    if not landmark_indices and not target_indices:
        print("\nExample CSV Gene IDs:", gene_names[:5])
        print("Example Landmark IDs:", landmark_genes[:5])
        print("Example Target IDs:", target_genes[:5])
        return

    selected_indices = landmark_indices + target_indices
    filtered_data = gene_expression[selected_indices, :].astype(np.float64)

    print(f"Final data matrix shape: {filtered_data.shape}")
    print(f"Example matched genes: {[gene_names[i] for i in selected_indices[:5]]}")

    np.save(OUTPUT_NPY, filtered_data)
    print(f"Saved matrix of shape {filtered_data.shape} to {OUTPUT_NPY}")


if __name__ == "__main__":
    main()
