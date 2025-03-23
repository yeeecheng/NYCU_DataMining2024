import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

class DataLoader:
    def __init__(self, file_path, mode="train"):
        self.df = pd.read_csv(file_path)
        self.mode = mode
  
    def __train_processing(self):
        
        X, y = [], []
        self.df = self.df[self.features + ["Date"]]

        for month in range(1, 13):
            month_data = self.df[self.df["Date"].astype(str).str.match(rf"{month}/")]
            for i in range(len(month_data) - self.window_size):
           
                # last column in training X can not has PM2.5, which is predicted value.
                X.append(month_data.iloc[i : i + self.window_size, month_data.columns != 'Date'].values.flatten())
                y.append(month_data.iloc[i + self.window_size]["PM2.5"])

        return X, y
    
    def __test_processing(self):
        
        X = []
        self.df = self.df[self.features + ["index"]]
        
        for idx in range(244):
            day_pred_data = self.df[self.df["index"] == f"index_{idx}"]
            X.append(day_pred_data.iloc[-1 - self.window_size + 1: , day_pred_data.columns != 'index'].values.flatten())
               
        return X, None

    def process(self, features, window_size=1):

        self.features = features
        self.window_size = window_size
        
        X, y = self.__train_processing() if self.mode == "train" else self.__test_processing()

        X = np.array(X, dtype=np.float64)
        y = np.array(y, dtype=np.float64)
        
        return X, y

def z_score_standardize(X, mean=None, std=None, clip_range=3):
    if mean is None:
        mean = np.mean(X, axis=0, keepdims=True)
    if std is None:
        std = np.std(X, axis=0, keepdims=True)

    X_std = (X - mean) / (std + 1e-8)

    X_std = np.clip(X_std, -clip_range, clip_range)

    return X_std, mean, std

def robust_normalize(X, median=None, iqr=None):
    if median is None:
        median = np.median(X, axis=0, keepdims=True)
    if iqr is None:
        q1 = np.percentile(X, 25, axis=0, keepdims=True)
        q3 = np.percentile(X, 75, axis=0, keepdims=True)
        iqr = q3 - q1
    
    X_norm = (X - median) / (iqr + 1e-8)
    return X_norm, median, iqr

def min_max_normalize(X, min_val=None, max_val=None):
    if min_val is None:
        min_val = np.min(X, axis=0, keepdims=True)
    if max_val is None:
        max_val = np.max(X, axis=0, keepdims=True)
    
    X_norm = (X - min_val) / (max_val - min_val + 1e-8) 
    return X_norm, min_val, max_val


def create_output_csv(y, save_filepath="./submission.csv"):
    
    out = [[f"index_{i}", a] for i, a in enumerate(y)]
    df = pd.DataFrame(out, columns=["index", "answer"])
    df.to_csv(save_filepath, index=False, encoding='utf-8')
    print(f"Save at {save_filepath}")

if __name__ == "__main__":
  
    pass


