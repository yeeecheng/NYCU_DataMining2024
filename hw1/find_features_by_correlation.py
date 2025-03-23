import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    
    train_filepath = "./data/processed_train_with_valid_val.csv"
    test_filepath = "./data/processed_test_with_valid_val.csv"
    target = "PM2.5"
    threshold=0.5

    df = pd.read_csv(train_filepath)

    """
    features = ["AMB_TEMP", "CH4", "CO", "NMHC", "NO", "NO2",
                 "NOx", "O3", "PM10", "RAINFALL", "RH","SO2", 
                 "THC", "WD_HR", "WIND_DIREC", "WIND_SPEED", "WS_HR", "PM2.5"]
    """

    features = [col for col in df.columns if col not in ["Date", "index", target]]
    corr_matrix = df[features + [target]].corr()
    target_corr = corr_matrix[target].drop(target)
    selected_features = target_corr[abs(target_corr) >= threshold].index.tolist()
    print(f"selected features: {selected_features}")

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
    plt.title("Feature Correlation Heatmap")
    
    plt.savefig("./correlation_heatmap.png")
    plt.close()

if __name__ == "__main__":
    main()
