import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
from xgboost import XGBClassifier
from preprocess import load_data, clean_data, feature_engineering, get_features_and_target

def train():
    # 데이터 준비
    print("데이터 로딩 중...")
    df = load_data()
    df = clean_data(df)
    df = feature_engineering(df)
    X, y = get_features_and_target(df)

    # 학습/테스트 분리 (80% / 20%)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"학습 데이터: {X_train.shape[0]}명")
    print(f"테스트 데이터: {X_test.shape[0]}명")

    # 클래스 불균형 처리 (부도 6% vs 정상 94%)
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale = neg / pos
    print(f"\n클래스 불균형 비율: {scale:.1f} (정상/부도)")

    # XGBoost 모델 학습
    print("\n모델 학습 중...")
    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        scale_pos_weight=scale,  # 클래스 불균형 보정
        eval_metric='auc',
        random_state=42,
        n_jobs=-1
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=50
    )

    # 성능 평가
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)

    auc = roc_auc_score(y_test, y_pred_proba)
    print(f"\n=== 모델 성능 ===")
    print(f"AUC-ROC: {auc:.4f}")
    print("\n분류 리포트:")
    print(classification_report(y_test, y_pred, target_names=["정상", "부도"]))

    # 피처 중요도 Top 5
    print("=== 피처 중요도 Top 5 ===")
    importances = model.feature_importances_
    feature_names = X.columns.tolist()
    sorted_idx = np.argsort(importances)[::-1]
    for i in range(5):
        print(f"  {feature_names[sorted_idx[i]]}: {importances[sorted_idx[i]]:.4f}")

    # 모델 저장
    with open("models/xgboost_model.pkl", "wb") as f:
        pickle.dump(model, f)
    print("\n모델 저장 완료: models/xgboost_model.pkl")

    return model, X_test, y_test

if __name__ == "__main__":
    train()
