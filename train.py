import os

import config

import joblib
import pandas as pd
from sklearn import metrics
from sklearn import tree

import argparse

def run(fold):
    df = pd.read_csv(config.TRAINING_FILE)
    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    x_train = df_train.drop('target', axis=1).values
    y_train = df_train.target.values

    x_valid = df_valid.drop('target', axis=1).values
    y_valid = df_valid.target.values

    clf = tree.DecisionTreeClassifier()

    clf.fit(x_train, y_train)

    preds = clf.predict(x_valid)
    score = metrics.roc_auc_score(y_valid, preds)
    print(f"fold = {fold}, ROC_AUC Score={score}")

    joblib.dump(clf, os.path.join(config.MODEL_OUTPUT, f"dt_{fold}.bin"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--fold", 
        type=int
    )

    args = parser.parse_args()

    run(fold=args.fold)
          