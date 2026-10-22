import joblib
import numpy as np
from custom_model import LogisticRegression

model = joblib.load("cancer_pipeline.joblib")

print("Model loaded successfully!")
print("Model type:", type(model))
print("Weights:", model.weights.shape)
print("Bias:", model.bias)