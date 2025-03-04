import pandas as pd
file_path = "/home/ibab/simulated_data_multiple_linear_regression_for_ML.csv"
df = pd.read_csv(file_path)
thresholds = [80, 78, 82]
partitioned_datasets = {}
for t in thresholds:
    below_t = df[df["BP"] < t]
    above_t = df[df["BP"] >= t]
    partitioned_datasets[f"below_{t}"] = below_t
    partitioned_datasets[f"above_{t}"] = above_t
for key, value in partitioned_datasets.items():
    print(f"{key}: {len(value)} entries")
