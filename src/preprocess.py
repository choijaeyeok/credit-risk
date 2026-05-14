import pandas as pd
import numpy as np

DATA_PATH = "data/GiveMeSomeCredit/cs-training.csv"

def load_data():
    df = pd.read_csv(DATA_PATH, index_col=0)
    return df

def clean_data(df):
    # 이상치 제거
    df = df[df['age'] > 0]
    df = df[df['age'] <= 100]

    # 결측치 처리
    df['MonthlyIncome'] = df['MonthlyIncome'].fillna(df['MonthlyIncome'].median())
    df['NumberOfDependents'] = df['NumberOfDependents'].fillna(0)

    # 이상치 클리핑
    df['DebtRatio'] = df['DebtRatio'].clip(upper=10)
    df['RevolvingUtilizationOfUnsecuredLines'] = df['RevolvingUtilizationOfUnsecuredLines'].clip(upper=1)

    return df

def feature_engineering(df):
    # 총 연체 횟수
    df['TotalPastDue'] = (
        df['NumberOfTime30-59DaysPastDueNotWorse'] +
        df['NumberOfTime60-89DaysPastDueNotWorse'] +
        df['NumberOfTimes90DaysLate']
    )

    # 소득 대비 부채 (월소득이 0이면 0으로 처리)
    df['MonthlyDebt'] = df['DebtRatio'] * df['MonthlyIncome']

    # 나이대 구간화
    df['AgeGroup'] = pd.cut(df['age'], bins=[0, 30, 40, 50, 60, 100], labels=[0, 1, 2, 3, 4])
    df['AgeGroup'] = df['AgeGroup'].astype(int)

    return df

def get_features_and_target(df):
    feature_cols = [
        'RevolvingUtilizationOfUnsecuredLines',
        'age',
        'NumberOfTime30-59DaysPastDueNotWorse',
        'DebtRatio',
        'MonthlyIncome',
        'NumberOfOpenCreditLinesAndLoans',
        'NumberOfTimes90DaysLate',
        'NumberRealEstateLoansOrLines',
        'NumberOfTime60-89DaysPastDueNotWorse',
        'NumberOfDependents',
        'TotalPastDue',
        'MonthlyDebt',
        'AgeGroup'
    ]
    X = df[feature_cols]
    y = df['SeriousDlqin2yrs']
    return X, y

if __name__ == "__main__":
    df = load_data()
    print(f"원본 데이터: {df.shape}")

    df = clean_data(df)
    print(f"정제 후: {df.shape}")

    df = feature_engineering(df)
    print(f"피처 엔지니어링 후: {df.shape}")

    X, y = get_features_and_target(df)
    print(f"부도 비율: {y.mean():.2%}")
    print("\n피처 목록:")
    print(X.columns.tolist())
