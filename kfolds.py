import pandas as pd
from sklearn import model_selection
from sklearn import preprocessing


if __name__ == "__main__":
    # Reading the csv file
    df = pd.read_csv("/home/devansh/workspace/Liver_prediction/input/indian_liver_patient.csv")
    
    # Replacing Target values
    df['Dataset'].replace({1:1, 2:0},inplace=True)

    # One hot Encoding Gender column
    encoder = preprocessing.LabelEncoder()
    df['Gender'] = encoder.fit_transform(df['Gender'])

    # Renaming Target columns
    df.rename(columns={'Dataset':'target'},inplace=True)
    
    # Filling the missing values
    df['Albumin_and_Globulin_Ratio'].fillna(df['Albumin_and_Globulin_Ratio'].mean(), inplace=True)
    
    # Making a new column kfold
    df['kfold'] = -1
    
    # Shuffle the dataset
    df = df.sample(frac=1).reset_index(drop=True)
    
    # Set target variable
    y = df.target.values
    
    # Doing stratified KFold
    kf = model_selection.StratifiedKFold(n_splits=5)

    # filling the value in the column KFold
    for f, (t_, v_) in enumerate(kf.split(X = df, y=y)):
        df.loc[v_, 'kfold'] = f

    # saving the data to new csv 
    df.to_csv('train_folds.csv', index=False)     