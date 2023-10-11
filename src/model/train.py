import argparse
import glob
import os
import pandas as pd
from sklearn.linear_model import LogisticRegression

def get_csvs_df(path):
    if not os.path.exists(path):
        raise RuntimeError(f"Cannot use non-existent path provided: {path}")
    csv_files = glob.glob(f"{path}/*.csv")
    if not csv_files:
        raise RuntimeError(f"No CSV files found in provided data path: {path}")
    return pd.concat((pd.read_csv(f) for f in csv_files), sort=False)

def split_data(df):
    X = df[['Pregnancies','PlasmaGlucose','DiastolicBloodPressure','TricepsThickness','SerumInsulin','BMI','DiabetesPedigree','Age']].values
    y = df['Diabetic'].values
    return X, y

def train_model(reg_rate, X_train, y_train):
    LogisticRegression(C=1/reg_rate, solver="liblinear").fit(X_train, y_train)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_data", dest='training_data', type=str)
    parser.add_argument("--reg_rate", dest='reg_rate', type=float, default=0.01)
    args = parser.parse_args()
    return args

def main(args):
    df = get_csvs_df(args.training_data)
    X_train, y_train = split_data(df)
    train_model(args.reg_rate, X_train, y_train)

if __name__ == "__main__":
    args = parse_args()
    main(args)
