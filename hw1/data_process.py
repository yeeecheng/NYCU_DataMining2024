import pandas as pd
import numpy as np

def replace_invalid_val(df):
    
    for col in df.columns:
        if col == "Date" or col == "index":
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")
        # mean_value = df[col].mean()
        # df[col] = df[col].fillna(mean_value)
        df[col] = df[col].fillna(method="ffill") 
        df[col] = df[col].fillna(method="bfill") 
    return df

def preprocess_train_data(file_path):

    df = pd.read_csv(file_path)

    data_item = ['Date']
    data_item += [x.strip() for x in df['ItemName'].unique() if x.strip() != 'PM2.5']
    data_item.append('PM2.5')

    dates = [f"{m}/{d} {h:02d}:00" for m in range(1, 13) for d in range(1, 21) for h in range(0, 24)]
    datas = []
    for day_idx in range(12 * 20):

        for hours in range(0, 24):
            data = [dates[24 * day_idx + hours]]
            # remove space part
            data += [x.strip() if isinstance(x, str) else x for x in df[str(hours)][0 + 18 * day_idx : 9 + 18 * day_idx]]
            data += [x.strip() if isinstance(x, str) else x for x in df[str(hours)][10 + 18 * day_idx : 18 + 18 * day_idx]]
            data += [x.strip() if isinstance(x, str) else x for x in df[str(hours)][9 + 18 * day_idx : 10 + 18 * day_idx]]

            datas.append(data)

    df = pd.DataFrame(datas, columns=data_item)
    df.to_csv("./data/processed_train_with_invalid_val.csv", index=False)
    print("Save processed_train_with_invalid_val.csv")
    
    valid_df = replace_invalid_val(df)
    valid_df.to_csv("./data/processed_train_with_valid_val.csv", index=False)
    print("Save processed_train_with_valid_val.csv")

def preprocess_test_data(file_path):

    df = pd.read_csv(file_path, header=None)
    
    data_item = ["index", "AMB_TEMP", "CH4", "CO", "NMHC", "NO", "NO2", "NOx", 
                "O3", "PM10", "RAINFALL", "RH","SO2", "THC", "WD_HR", 
                "WIND_DIREC", "WIND_SPEED", "WS_HR", "PM2.5"]
    datas = []
    day_num = len(df) // 18
    
    for day_idx in range(day_num):
        
        daliy_data = df[df[0] == f"index_{day_idx}"]
      
        for hours in range(2, 11):
            # remove space part
            data = [f"index_{day_idx}"]
            data += [x.strip() if isinstance(x, str) else x for x in daliy_data[hours][0: 9]]
            data += [x.strip() if isinstance(x, str) else x for x in daliy_data[hours][10: 18]]
            data += [x.strip() if isinstance(x, str) else x for x in daliy_data[hours][9: 10]]
            datas.append(data)

    df = pd.DataFrame(datas, columns=data_item)
    df.to_csv("./data/processed_test_with_invalid_val.csv", index=False)
    print("Save processed_test_with_invalid_val.csv")

    valid_df = replace_invalid_val(df)
    valid_df.to_csv("./data/processed_test_with_valid_val.csv", index=False)
    print("Save processed_test_with_valid_val.csv")

if __name__ == "__main__":
  
    train_filepath = "./data/train.csv"
    preprocess_train_data(train_filepath)
    test_filepath = "./data/test.csv"
    preprocess_test_data(test_filepath)