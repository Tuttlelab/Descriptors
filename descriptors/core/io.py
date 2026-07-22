
import pandas as pd
import os

def save_features_parquet(feature_rows, out_path):
	df = pd.DataFrame(feature_rows)
	df.to_parquet(out_path, index=False)

def save_features_csv(feature_rows, out_path):
	df = pd.DataFrame(feature_rows)
	df.to_csv(out_path, index=False)

def ensure_dir(path):
	os.makedirs(path, exist_ok=True)
