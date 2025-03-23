import argparse
import numpy as np
import pandas as pd
from utils import DataLoader, create_output_csv, z_score_standardize, robust_normalize, min_max_normalize
from model import LinearRegression
import matplotlib.pyplot as plt

def main():
    
    train_losses = {}
    val_losses = {}

    train_filepath = rf"./data/processed_train_with_valid_val_method2.csv"
    test_filepath = rf"./data/processed_test_with_valid_val_method2.csv"
    """
    features = ["AMB_TEMP", "CH4", "CO", "NMHC", "NO", "NO2",
                "NOx", "O3", "PM10", "RAINFALL", "RH","SO2", 
                "THC", "WD_HR", "WIND_DIREC", "WIND_SPEED", "WS_HR", "PM2.5"]
    """
    features = ['CO', 'NMHC', 'NO2', 'PM10', 'THC', "PM2.5"]
    # features = ["AMB_TEMP", "CH4", "CO", "NMHC", "NO", "NO2",
    #              "NOx", "PM10", "THC", "PM2.5"]
    # maximum is 9 because of limitation of test data
    
    # [1 ~ 9], best one is 8
    window_size = 8
    # [zscore, robust, maxmin], best one is robust
    normalize_method = "robust"
    # [normal, l1, l2, elastic_net(l1 + l2)], best one is l2
    regularization = "l2"
    train_dataloader = DataLoader(file_path=train_filepath, mode="train")
    test_dataloader = DataLoader(file_path=test_filepath, mode="test")
    
    # select_features_by_correlation(dataloader.df)
    train_X, train_y = train_dataloader.process(features=features,window_size=window_size)
    print(f"[Train] X's Shape: {train_X.shape} Type: {train_X.dtype}, y's Shape: {train_y.shape} Type: {train_y.dtype}")
    test_X, _ = test_dataloader.process(features=features,window_size=window_size)
    print(f"[Test] X's Shape: {test_X.shape} Type: {test_X.dtype}")


    # normalize to avoid gradient explosion
    if normalize_method == "zscore":
        train_X, mean_train_X, std_train_X = z_score_standardize(train_X)
        test_X, mean_test_X, std_test_X = z_score_standardize(test_X, mean=mean_train_X, std=std_train_X)
    elif normalize_method == "robust":
        train_X, median_train_X, iqr_train_X = robust_normalize(train_X)
        test_X, _, _ = robust_normalize(test_X, median=median_train_X, iqr=iqr_train_X)
    elif normalize_method == "maxmin":
        train_X, min_train_X, max_train_X = min_max_normalize(train_X)
        test_X, _, _ = min_max_normalize(test_X, min_val=min_train_X, max_val=max_train_X)
 
    # import matplotlib.pyplot as plt
    # plt.hist(train_X.flatten(), bins=50, alpha=0.5, label="train")
    # plt.hist(test_X.flatten(), bins=50, alpha=0.5, label="test")
    # plt.legend()
    # plt.savefig("histogram.png", dpi=300, bbox_inches='tight')
    # plt.close()


    """ experiment for problem 2 """
    # ratios = np.arange(0.9, 0.0, -0.1)
    # final_val_losses = []

    # for ratio in ratios:

    #     train_len = int(len(train_X) * ratio)
    #     train_X, val_X, train_y, val_y = train_X[:train_len], train_X[train_len:], train_y[:train_len], train_y[train_len]
    
    #     model = LinearRegression(learning_rate=0.01, epochs=20000)
    #     model.fit(train_X, train_y, val_X, val_y)
    #     final_val_losses.append(model.best_val_loss)
    #     plt.figure(figsize=(10, 5))
    #     plt.plot(model.train_losses, label=f"Train Loss (ratio={ratio:.1f})")
    #     plt.plot(model.val_losses, label=f"Val Loss (ratio={ratio:.1f})")
    #     plt.xlabel("Epochs")
    #     plt.ylabel("RMSE")
    #     plt.title(f"Training vs Validation Loss (Train Ratio: {ratio:.1f})")
    #     plt.legend()
    #     plt.grid(True)
    #     plt.tight_layout()
    #     plt.savefig(f"loss_curve_ratio_{int(ratio*100)}.png")
    #     plt.close()
    
    # plt.figure(figsize=(10, 6))
    # plt.bar([f"{int(r*100)}%" for r in ratios], final_val_losses)
    # plt.xlabel("Training Data Ratio")
    # plt.ylabel("Best Validation RMSE")
    # plt.title("Validation Loss vs. Training Data Ratio")
    # plt.grid(axis='y')
    # plt.tight_layout()
    # plt.savefig("val_loss_vs_ratio.png")
    # plt.show()

    """ experiment for problem 3 """

    # regularization_types = ["normal", "l1", "l2", "elastic_net"]
    # train_losses = {}
    # val_losses = {}
    # final_val_losses = {}

    # for reg in regularization_types:
    #     train_len = int(len(train_X) * 0.9)  # Using 80% of data for training
    #     train_X, val_X, train_y, val_y = train_X[:train_len], train_X[train_len:], train_y[:train_len], train_y[train_len]

    #     model = LinearRegression(learning_rate=0.01, epochs=20000, regularization=reg)
    #     model.fit(train_X, train_y, val_X, val_y)

    #     train_losses[reg] = model.train_losses
    #     val_losses[reg] = model.val_losses
    #     final_val_losses[reg] = model.best_val_loss

    #     plt.figure(figsize=(10, 5))
    #     plt.plot(model.train_losses, label=f"Train Loss ({reg})")
    #     plt.plot(model.val_losses, label=f"Val Loss ({reg})")
    #     plt.xlabel("Epochs")
    #     plt.ylabel("RMSE")
    #     plt.title(f"Training vs Validation Loss - {reg.capitalize()} Regularization")
    #     plt.legend()
    #     plt.grid(True)
    #     plt.tight_layout()
    #     plt.savefig(f"loss_curve_{reg}.png")
    #     plt.close()
    # plt.figure(figsize=(10, 6))
    # plt.bar(regularization_types, [final_val_losses[reg] for reg in regularization_types])
    # plt.xlabel("Regularization Type")
    # plt.ylabel("Best Validation RMSE")
    # plt.title("Validation RMSE vs. Regularization Type")
    # plt.grid(axis='y')
    # plt.tight_layout()
    # plt.savefig("val_loss_vs_regularization.png")
    # plt.show()


    """ main part """

    model = LinearRegression(learning_rate=0.01, epochs=20000, regularization=regularization)
    model.fit(train_X, train_y, train_X[:1000], train_y[:1000])

    # train_losses[f"{nor_method}"] = model.train_losses
    # val_losses[f"{nor_method}"] = model.val_losses
    # plt.figure(figsize=(12, 6))
    # for method in train_losses:
    #     plt.plot(train_losses[method], label=f"Train Loss ({method})", linestyle='-')
    #     plt.plot(val_losses[method], label=f"Val Loss ({method})", linestyle='--')
    
    # plt.xlabel("Epochs")
    # plt.ylabel("RMSE")
    # plt.title("Training vs Validation Loss: different normalize")
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.savefig("loss_comparison_different_normalize_methods.png")
    # plt.show()


    model.save_weights("linear_regression_weights.npz")

    model = LinearRegression()
    model.load_weights("linear_regression_weights.npz")
    y_pred = model.predict(test_X)
    create_output_csv(y_pred, save_filepath="./submission.csv")

    

if __name__ == "__main__":
   
    main()
