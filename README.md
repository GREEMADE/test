# test
test
import os
import warnings

# غیرفعال کردن پیام‌های مربوط به oneDNN در TensorFlow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# غیرفعال کردن هشدارهای FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)

# ادامه کد شما...
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
import torch
import cv2
from PIL import Image
import spacy
import nltk
import xgboost as xgb
import lightgbm as lgb
import plotly.express as px
import joblib

# Numpy
arr = np.array([1, 2, 3])
print(f"Numpy array: {arr}")

# Pandas
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
print(f"Pandas DataFrame:\n{df}")

# Matplotlib
plt.plot([1, 2, 3], [4, 5, 6])
plt.title("Matplotlib Test Plot")
plt.show()

# Seaborn
sns.set_theme(style="darkgrid")
tips = sns.load_dataset("tips")
sns.boxplot(x="day", y="total_bill", data=tips)
plt.show()

# Scikit-learn
clf = RandomForestClassifier()
print(f"RandomForestClassifier: {clf}")

# TensorFlow
print(f"TensorFlow version: {tf.__version__}")

# PyTorch
x = torch.rand(5, 3)
print(f"PyTorch Tensor:\n{x}")

# OpenCV
print(f"OpenCV version: {cv2.__version__}")

# Pillow (PIL)
img = Image.new('RGB', (60, 30), color=(73, 109, 137))
img.show()
print(f"Pillow image created: {img}")

# Spacy
nlp = spacy.blank("en")
print(f"Spacy Language model: {nlp}")

# NLTK
nltk.download('punkt', quiet=True)
print("NLTK punkt downloaded")

# XGBoost
print(f"XGBoost version: {xgb.__version__}")

# LightGBM
print(f"LightGBM version: {lgb.__version__}")

# Plotly
fig = px.scatter(x=[1, 2, 3], y=[4, 5, 6], title="Plotly Test Scatter")
fig.show()

# Joblib
print(f"Joblib version: {joblib.__version__}")
