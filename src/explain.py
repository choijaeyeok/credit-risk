import pickle
import shap
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['axes.unicode_minus'] = False

FEATURE_NAME_MAP = {
    'RevolvingUtilizationOfUnsecuredLines': '신용카드 사용률',
    'age': '나이',
    'NumberOfTime30-59DaysPastDueNotWorse': '30-59일 연체 횟수',
    'DebtRatio': '부채비율',
    'MonthlyIncome': '월소득',
    'NumberOfOpenCreditLinesAndLoans': '대출 건수',
    'NumberOfTimes90DaysLate': '90일 이상 연체 횟수',
    'NumberRealEstateLoansOrLines': '부동산 대출 건수',
    'NumberOfTime60-89DaysPastDueNotWorse': '60-89일 연체 횟수',
    'NumberOfDependents': '부양가족 수',
    'TotalPastDue': '총 연체 횟수',
    'MonthlyDebt': '월 부채액',
    'AgeGroup': '나이대'
}

def load_model():
    with open("models/xgboost_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

def get_shap_values(model, X):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X)
    return explainer, shap_values

def explain_individual(model, X_input, save_path=None):
    """단일 신청자에 대한 SHAP 설명"""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_input)

    # 한글 컬럼명으로 변환
    shap_values.feature_names = [
        FEATURE_NAME_MAP.get(f, f) for f in X_input.columns.tolist()
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    shap.plots.waterfall(shap_values[0], max_display=10, show=False)
    plt.title("대출 심사 결과 상세 분석", fontsize=14, pad=15)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig, shap_values

def get_reason_text(shap_values, X_input):
    """거절/승인 주요 이유를 텍스트로 반환"""
    values = shap_values[0].values
    feature_names = [FEATURE_NAME_MAP.get(f, f) for f in X_input.columns.tolist()]

    reasons = sorted(zip(feature_names, values), key=lambda x: x[1])

    negative = [(name, val) for name, val in reasons if val < 0][:3]
    positive = [(name, val) for name, val in reasons if val > 0][-3:]

    result = {}
    if negative:
        result['부정적 요인'] = [(name, round(val, 3)) for name, val in negative]
    if positive:
        result['긍정적 요인'] = [(name, round(val, 3)) for name, val in reversed(positive)]

    return result

def summary_plot(model, X_sample, save_path=None):
    """전체 데이터 기준 피처 중요도 시각화"""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(
        shap_values, X_sample,
        feature_names=[FEATURE_NAME_MAP.get(f, f) for f in X_sample.columns],
        show=False
    )
    plt.title("전체 피처 영향도", fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig

if __name__ == "__main__":
    import sys
    sys.path.append("src")
    from preprocess import load_data, clean_data, feature_engineering, get_features_and_target

    df = load_data()
    df = clean_data(df)
    df = feature_engineering(df)
    X, y = get_features_and_target(df)

    model = load_model()

    # 첫 번째 신청자 분석
    X_sample = X.iloc[[0]]
    prob = model.predict_proba(X_sample)[0][1]
    print(f"부도 확률: {prob:.2%}")

    fig, shap_values = explain_individual(model, X_sample, save_path="models/shap_waterfall.png")
    reasons = get_reason_text(shap_values, X_sample)

    print("\n=== 심사 분석 ===")
    for category, items in reasons.items():
        print(f"\n{category}:")
        for name, val in items:
            print(f"  {name}: {val}")

    print("\nSHAP 차트 저장 완료: models/shap_waterfall.png")
