import pandas as pd

# 1. Load your DVC-tracked raw dataset
raw_df = pd.read_csv("data/raw/swiggy.csv")

# 2. Extract a random 100-row sample
sample_df = raw_df.sample(n=100, random_state=42)

# 3. Save it as a new file in your data folder
sample_df.to_csv("data/sample_raw_test.csv", index=False)