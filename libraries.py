import os
import re
import json
import joblib
import unicodedata
import pandas as pd
import numpy as np
import xgboost as xgb
import mysql.connector
import duckdb # MỚI: Dùng cho chat.py
import subprocess # MỚI: Dùng để gọi script con
import sys
import io
from collections import Counter
from datetime import datetime, timedelta # MỚI: Xử lý thời gian dự báo
from tabulate import tabulate # MỚI: Kẻ bảng đẹp

# Từ Scikit-learn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder

# --- CẤU HÌNH THƯ MỤC ---
MODEL_DIR = 'model'
DATA_HISTORY_DIR = 'data_history'
DATASET_DIR = 'dataset'

for folder in [MODEL_DIR, DATA_HISTORY_DIR, DATASET_DIR]:
    if not os.path.exists(folder):
        os.makedirs(folder)

print(">>> [LORA AI]: Hệ thống thư viện đã sẵn sàng.")