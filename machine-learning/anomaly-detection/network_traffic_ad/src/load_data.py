# src/load_data.py
import os
import urllib.request
import gzip
import shutil
import pandas as pd

# __file__ is ".../anomaly-detection/src/load_data.py"
# We want BASE_DIR = ".../anomaly-detection"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR   = os.path.dirname(SCRIPT_DIR)

DATA_DIR   = os.path.join(BASE_DIR, "data")
LOCAL_GZ   = os.path.join(DATA_DIR, "kddcup.data_10_percent.gz")
LOCAL_CSV  = os.path.join(DATA_DIR, "kddcup.data_10_percent.csv")
DATA_URL   = "http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data_10_percent.gz"

def download_and_extract():
    os.makedirs(DATA_DIR, exist_ok=True)

    if not os.path.exists(LOCAL_GZ):
        print("Downloading 10% KDD Cup '99 data…")
        urllib.request.urlretrieve(DATA_URL, LOCAL_GZ)
        print("Download complete.")
    else:
        print(f"{LOCAL_GZ} already exists, skipping download.")

    if not os.path.exists(LOCAL_CSV):
        print("Extracting .gz to .csv…")
        with gzip.open(LOCAL_GZ, 'rb') as f_in, open(LOCAL_CSV, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
        print("Extraction complete.")
    else:
        print(f"{LOCAL_CSV} already exists, skipping extraction.")
    return(LOCAL_CSV)

def load_dataframe(csv_path):
    col_names = [
        "duration","protocol_type","service","flag","src_bytes","dst_bytes",
        "land","wrong_fragment","urgent","hot","num_failed_logins","logged_in",
        "num_compromised","root_shell","su_attempted","num_root","num_file_creations",
        "num_shells","num_access_files","num_outbound_cmds","is_host_login",
        "is_guest_login","count","srv_count","serror_rate","srv_serror_rate",
        "rerror_rate","srv_rerror_rate","same_srv_rate","diff_srv_rate",
        "srv_diff_host_rate","dst_host_count","dst_host_srv_count",
        "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
        "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
        "dst_host_rerror_rate","dst_host_srv_rerror_rate","label"
    ]
    print("Loading CSV into DataFrame (this may take ~30 s)…")
    df = pd.read_csv(csv_path, names=col_names)
    print(f"Loaded {len(df)} rows.")
    return df

if __name__ == "__main__":
    csv_path = download_and_extract()
    df = load_dataframe(csv_path)

    # Save a quick sample under “…/anomaly-detection/data/”
    sample_path = os.path.join(DATA_DIR, "sample_head.csv")
    df.head(5).to_csv(sample_path, index=False)
    print(f"Sample head saved to {sample_path}.")
