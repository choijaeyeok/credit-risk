import sys
sys.path.append("src")

import pickle
import pandas as pd
import streamlit as st

from explain import get_shap_values
from score import prob_to_score, get_grade_and_rate, MIN_INCOME, DSR_LIMIT, KRW_TO_USD

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


def val_to_label(val):
    abs_val = abs(val)
    if val < 0:
        if abs_val > 0.3:  return "🔴 매우 불리"
        if abs_val > 0.1:  return "🟠 불리"
        return "🟡 약간 불리"
    else:
        if abs_val > 0.3:  return "🟢 매우 유리"
        if abs_val > 0.1:  return "🔵 유리"
        return "🟡 약간 유리"

def val_to_bar(val):
    abs_val = abs(val)
    if abs_val > 0.3:  return 1.0
    if abs_val > 0.1:  return 0.6
    return 0.3

@st.cache_resource
def load_model():
    with open("models/xgboost_model.pkl", "rb") as f:
        return pickle.load(f)

def build_input(age, income, debt_ratio, util_rate,
                past30, past60, past90, open_loans,
                real_estate, dependents):
    total_past_due = past30 + past60 + past90
    monthly_debt = debt_ratio * income
    if age <= 30:   age_group = 0
    elif age <= 40: age_group = 1
    elif age <= 50: age_group = 2
    elif age <= 60: age_group = 3
    else:           age_group = 4
    data = {
        'RevolvingUtilizationOfUnsecuredLines': util_rate,
        'age': age,
        'NumberOfTime30-59DaysPastDueNotWorse': past30,
        'DebtRatio': debt_ratio,
        'MonthlyIncome': income,
        'NumberOfOpenCreditLinesAndLoans': open_loans,
        'NumberOfTimes90DaysLate': past90,
        'NumberRealEstateLoansOrLines': real_estate,
        'NumberOfTime60-89DaysPastDueNotWorse': past60,
        'NumberOfDependents': dependents,
        'TotalPastDue': total_past_due,
        'MonthlyDebt': monthly_debt,
        'AgeGroup': age_group
    }
    return pd.DataFrame([data])

# ── UI ───────────────────────────────────────────────
st.set_page_config(page_title="대출 신용 리스크 평가", page_icon="🏦", layout="wide")

st.title("🏦 대출 신용 리스크 평가 시스템")
st.markdown("신청자 정보를 입력하면 AI가 대출 가능 여부와 적용 금리를 분석합니다.")
st.divider()

col1, col2 = st.columns(2)
with col1:
    st.subheader("📋 기본 정보")
    age = st.number_input("나이", min_value=18, max_value=100, value=35)
    income = st.number_input("월소득 (원)", min_value=0, max_value=100000000,
                              value=3000000, step=100000, format="%d")
    dependents = st.number_input("부양가족 수", min_value=0, max_value=20, value=0)

with col2:
    st.subheader("💳 신용 정보")
    pay_col1, pay_col2 = st.columns(2)
    with pay_col1:
        loan_payment = st.number_input("월 대출 납입액 (원)", min_value=0, max_value=100000000,
                                        value=300000, step=100000, format="%d",
                                        help="매달 은행에 갚는 대출 금액")
    with pay_col2:
        card_payment = st.number_input("월 카드 납입액 (원)", min_value=0, max_value=100000000,
                                        value=200000, step=100000, format="%d",
                                        help="매달 카드값으로 나가는 금액")
    monthly_payment = loan_payment + card_payment

    card_col1, card_col2 = st.columns(2)
    with card_col1:
        card_limit = st.number_input("신용카드 한도 (원)", min_value=0, max_value=100000000,
                                      value=3000000, step=100000, format="%d",
                                      help="신용카드가 없으면 0 입력")
    with card_col2:
        card_used = st.number_input("이번달 신용카드 사용액 (원)", min_value=0, max_value=100000000,
                                     value=900000, step=100000, format="%d",
                                     help="신용카드가 없으면 0 입력")

    lc1, lc2 = st.columns(2)
    with lc1:
        open_loan_count = st.number_input("보유 대출 수 (신용대출, 자동차 할부 등)", 0, 50, 3)
    with lc2:
        card_count = st.number_input("신용카드 수", 0, 50, 2)
    open_loans = open_loan_count + card_count

    st.markdown("#### 🏠 주택담보대출이 있나요?")
    has_mortgage = st.checkbox("예, 있습니다")
    if has_mortgage:
        real_estate = st.number_input("주택담보대출 개수", 1, 20, 1)
    else:
        real_estate = 0

    debt_ratio = monthly_payment / income if income > 0 else 0.0
    util_rate  = min(card_used / card_limit if card_limit > 0 else 0.0, 1.0)

st.subheader("⚠️ 연체 이력")
c1, c2, c3 = st.columns(3)
with c1:
    past30 = st.number_input("30~59일 연체 횟수", 0, 20, 0)
with c2:
    past60 = st.number_input("60~89일 연체 횟수", 0, 20, 0)
with c3:
    past90 = st.number_input("90일 이상 연체 횟수", 0, 20, 0)

st.divider()

if st.button("🔍 대출 심사 시작", type="primary", use_container_width=True):
    if income < MIN_INCOME:
        st.error(f"월소득이 {MIN_INCOME:,}원 미만이면 대출 심사가 불가합니다.")
        st.stop()

    if debt_ratio > DSR_LIMIT:
        st.error(f"DSR(총부채상환비율)이 {debt_ratio:.0%}로 기준({DSR_LIMIT:.0%})을 초과하여 대출이 불가합니다.")
        st.stop()

    model = load_model()

    # 모델이 미국 달러 기준 데이터로 학습되어 있어 원화 → 달러 변환 후 입력
    X_input = build_input(age, income / KRW_TO_USD, debt_ratio, util_rate,
                          past30, past60, past90, open_loans,
                          real_estate, dependents)

    prob        = model.predict_proba(X_input)[0][1]
    total_score = prob_to_score(prob)
    grade, rate = get_grade_and_rate(total_score)
    _, shap_values = get_shap_values(model, X_input)

    st.session_state['result'] = {
        'grade': grade,
        'rate': rate,
        'total_score': total_score,
        'shap_values': shap_values,
        'X_input': X_input,
    }

# 결과가 있으면 항상 표시
if 'result' in st.session_state:
    r = st.session_state['result']
    grade       = r['grade']
    rate        = r['rate']
    total_score = r['total_score']
    shap_values = r['shap_values']
    X_input     = r['X_input']

    st.divider()
    tab1, tab2 = st.tabs(["📊 심사 결과", "🔎 상세 분석"])

    # ── 탭1: 심사 결과 ──────────────────────────────────
    with tab1:
        r1, r2, r3, r4 = st.columns(4)
        with r1:
            if rate is not None:
                st.success("✅ 승인")
            else:
                st.error("❌ 거절")
        with r2:
            st.metric("신용점수", f"{total_score:.1f} / 100")
        with r3:
            st.metric("신용등급", grade if rate is not None else "대출 불가")
        with r4:
            st.metric("적용 금리", f"연 {rate}%" if rate else "대출 불가")

    # ── 탭2: 상세 분석 ──────────────────────────────────
    with tab2:
        st.caption("AI가 이 결과를 낸 이유를 항목별로 분석한 결과입니다.")
        values = shap_values[0].values
        names  = [FEATURE_NAME_MAP.get(c, c) for c in X_input.columns]
        pairs  = sorted(zip(names, values), key=lambda x: abs(x[1]), reverse=True)[:8]
        bad  = [(n, v) for n, v in pairs if v < 0]
        good = [(n, v) for n, v in pairs if v >= 0]

        col_bad, col_good = st.columns(2)
        with col_bad:
            st.markdown("#### ❌ 불리한 요인")
            if bad:
                for name, val in bad:
                    st.markdown(f"**{name}**")
                    st.markdown(val_to_label(val))
                    st.progress(float(val_to_bar(val)))
            else:
                st.info("불리한 요인이 없습니다.")
        with col_good:
            st.markdown("#### ✅ 유리한 요인")
            if good:
                for name, val in good:
                    st.markdown(f"**{name}**")
                    st.markdown(val_to_label(val))
                    st.progress(float(val_to_bar(val)))
            else:
                st.info("유리한 요인이 없습니다.")

