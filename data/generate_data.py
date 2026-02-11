import numpy as np
import pandas as pd

np.random.seed(42)

n_samples = 5000
n_features = 8

# Generate random sensor readings
X = np.random.normal(0, 1, (n_samples, n_features))

# Simulate degradation pattern
weights = np.random.uniform(0.5, 2.0, n_features)
risk_score = np.dot(X, weights)

# Convert to failure probability
prob = 1 / (1 + np.exp(-risk_score))

# Introduce imbalance (only ~15% failures)
threshold = np.percentile(prob, 85)
y = (prob > threshold).astype(int)

columns = [f"sensor_{i}" for i in range(n_features)]
df = pd.DataFrame(X, columns=columns)
df["failure"] = y

df.to_csv("sensor_data.csv", index=False)

print("Dataset generated!")