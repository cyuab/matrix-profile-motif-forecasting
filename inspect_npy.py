import numpy as np
import os

file_path = 'data/traffic.npy'

if os.path.exists(file_path):
    try:
        data = np.load(file_path)
        print(f"File: {file_path}")
        print(f"Shape: {data.shape}")
        print(f"Data Type: {data.dtype}")
        print("Content (first 5 rows):")
        print(data[:5])
        
        if len(data.shape) > 1:
             print("\nContent (first 5 rows, first 5 columns):")
             print(data[:5, :5])

    except Exception as e:
        print(f"Error loading {file_path}: {e}")
else:
    print(f"File not found: {file_path}")
