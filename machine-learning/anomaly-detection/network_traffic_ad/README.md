# Anomaly Detection for Network Traffic

Detect anomalous network connections using an Isolation Forest on a small subset of the KDD Cup '99 dataset. This notebook-driven project demonstrates:  
- How to load and preprocess raw network-flow features  
- How to train an Isolation Forest model to flag outliers  
- How to evaluate and visualize detection performance  


## Dataset

This is the data set used for The Third International Knowledge Discovery and Data Mining Tools Competition, which was held in conjunction with KDD-99 The Fifth International Conference on Knowledge Discovery and Data Mining. The competition task was to build a network intrusion detector, a predictive model capable of distinguishing between bad connections, called intrusions or attacks, and good/normal connections. This database contains a standard set of data to be audited, which includes a wide variety of intrusions simulated in a military network environment.

We use the "10% KDD Cup '99" sample (≈494,021 rows, 42 features). In this project, we consider a subset of numerical + one-hot encoded categorical features. Ground-truth labels ("normal" vs. various attack types) allow us to estimate precision/recall.  

Original source (10% sample):  
http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data_10_percent.gz  


## Requirements
- Python 3.8+  
- pandas  
- scikit-learn  
- matplotlib  
- seaborn (only for quick EDA; purely optional)  
- joblib (for saving models)  

Install via:  
```bash
pip install pandas scikit-learn matplotlib seaborn joblib
```

## Structure
anomaly-detection/
├── data/
│   └── kddcup.data_10_percent.gz
├── notebooks/
│   └── anomaly_detection.ipynb
├── src/
│   ├── load_data.py
│   ├── preprocess.py
│   ├── train_model.py
│   └── evaluate.py
└── reports/
    └── anomaly_report.pdf


## How to run

0. Navigate to project directory:
```bash
cd anomaly-detection/
```

1. Install Requirements:
```bash
pip install -r src/requirements.txt 
```

2. Download the data:
```bash
python src/load_data.py
```

3. Preprocess and split:
```bash
python src/preprocess.py --input data/kddcup.data_10_percent.csv \
                        --outdata data/processed.csv
```

4. Train model:
```bash
python src/train_model.py --input data/processed.csv \
                         --model_out models/iso_forest.joblib
```

5. Evaluate & plot:
```bash
python src/evaluate.py --input data/processed.csv \
                      --model models/iso_forest.joblib \
                      --outdir reports/
```

6. Open notebooks/anomaly_detection.ipynb to see a combined, exploratory workflow.


## Results Summary
You will see metrics like:
- Precision/Recall on “known attacks”
- A histogram of anomaly scores
- Summary of false positives/negatives

