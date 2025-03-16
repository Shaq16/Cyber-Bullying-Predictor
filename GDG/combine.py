import pandas as pd

# Load the datasets
aggressive_df = pd.read_csv("Aggressive_All.csv")
non_aggressive_df = pd.read_csv("Non_Aggressive_All.csv")

# Add labels: 1 for aggressive, 0 for non-aggressive
aggressive_df["label"] = 1
non_aggressive_df["label"] = 0

# Combine both datasets
combined_df = pd.concat([aggressive_df, non_aggressive_df], ignore_index=True)

# Shuffle the dataset (to mix aggressive and non-aggressive examples)
combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save to a new CSV file
combined_df.to_csv("cyberbullying_combined.csv", index=False)

print("Dataset combined and saved as 'cyberbullying_combined.csv'")
