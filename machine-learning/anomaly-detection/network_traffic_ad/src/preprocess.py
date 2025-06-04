# src/preprocess.py
import argparse
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from load_data import load_dataframe

def preprocess(input_csv, out_csv):
    print(input_csv)
    df = load_dataframe(input_csv)
    print(f"Raw data shape: {df.shape}")

    # 1) Keep only a subset of features (for brevity).
    keep_cols = [
        "duration","protocol_type","service","flag","src_bytes","dst_bytes",
        "logged_in","count","srv_count","dst_host_count","dst_host_srv_count","label"
    ]
    df = df[keep_cols]

    # 2) Encode categorical: protocol_type, service, flag
    cats = ["protocol_type","service","flag"]
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    onehot = encoder.fit_transform(df[cats])
    onehot_df = pd.DataFrame(onehot, columns=encoder.get_feature_names_out(cats))

    # 3) Drop original categorical, concat one-hot
    df_num = df.drop(columns=cats)
    df_proc = pd.concat([df_num.reset_index(drop=True), onehot_df.reset_index(drop=True)], axis=1)

    # 4) Convert label into binary: “normal.” → 0, others → 1
    df_proc["is_attack"] = df_proc["label"].apply(lambda x: 0 if x == "normal." else 1)
    df_proc = df_proc.drop(columns=["label"])

    print(f"Processed data shape: {df_proc.shape}")
    df_proc.to_csv(out_csv, index=False)
    print(f"Saved processed data to {out_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  required=True, help="Path to raw CSV")
    parser.add_argument("--outdata", required=True, help="Path to write processed CSV")
    args = parser.parse_args()
    preprocess(args.input, args.outdata)
