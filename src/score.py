SCORE_GRADE_TABLE = [
    (94, "1등급",  4.0),
    (89, "2등급",  5.2),
    (83, "3등급",  7.0),
    (77, "4등급",  9.0),
    (70, "5등급", 11.5),
    (63, "6등급", 14.5),
    (53, "7등급", 18.0),
    (45, "8등급", 22.0),
    (34, "9등급",  None),   # 대출 불가
]

MIN_INCOME = 1_500_000   # 월소득 150만원 미만 자동 거절
DSR_LIMIT  = 0.60        # DSR 60% 초과 자동 거절


def get_grade_and_rate(score):
    for threshold, grade, rate in SCORE_GRADE_TABLE:
        if score >= threshold:
            return grade, rate
    return "10등급", None


def calculate_score(input_data):
    """
    1000점 만점으로 계산 후 10으로 나눠 100점 만점으로 반환.
    카테고리 구조는 FICO 5요소 기반, 임계값은 KCB 공개 정보 근사치.
    """
    row = input_data.iloc[0]

    # ── 1. 연체 이력 (350점) ────────────────────────────────
    total_late = row['TotalPastDue']
    late_90    = row['NumberOfTimes90DaysLate']

    if total_late == 0:
        pay_score = 350
    elif total_late == 1:
        pay_score = 220
    elif total_late == 2:
        pay_score = 120
    else:
        pay_score = 30

    if late_90 >= 1:
        pay_score = max(pay_score - 80, 0)

    # ── 2. 부채 수준 (300점) ────────────────────────────────
    debt_ratio = row['DebtRatio']   # app에서 월납입액/월소득 = DSR
    util_rate  = row['RevolvingUtilizationOfUnsecuredLines']
    income     = row['MonthlyIncome']

    # DSR (150점)
    if debt_ratio < 0.20:
        dsr_score = 150
    elif debt_ratio < 0.30:
        dsr_score = 120
    elif debt_ratio < 0.40:
        dsr_score = 90
    elif debt_ratio < 0.50:
        dsr_score = 50
    else:
        dsr_score = 20

    # 월소득 (90점)
    if income >= 8_000_000:
        income_score = 90
    elif income >= 5_000_000:
        income_score = 75
    elif income >= 3_000_000:
        income_score = 60
    elif income >= 2_000_000:
        income_score = 40
    elif income >= 1_500_000:
        income_score = 20
    else:
        income_score = 0

    # 카드 사용률 (60점)
    if util_rate < 0.30:
        util_score = 60
    elif util_rate < 0.60:
        util_score = 40
    elif util_rate < 0.90:
        util_score = 20
    else:
        util_score = 5

    debt_total = dsr_score + income_score + util_score

    # ── 3. 신용 기간 (150점) — 나이로 대체 ─────────────────
    age = row['age']

    if age >= 50:
        age_score = 150
    elif age >= 40:
        age_score = 120
    elif age >= 30:
        age_score = 90
    elif age >= 25:
        age_score = 60
    else:
        age_score = 40

    # ── 4. 신용 종류 (100점) ────────────────────────────────
    open_loans   = row['NumberOfOpenCreditLinesAndLoans']
    real_estate  = row['NumberRealEstateLoansOrLines']
    total_credit = open_loans + real_estate

    if 2 <= total_credit <= 7:
        mix_score = 100
    elif total_credit == 0 or total_credit > 12:
        mix_score = 40
    else:
        mix_score = 70

    # ── 5. 신규 신용 (100점) — 카드 사용률로 추정 ───────────
    if util_rate < 0.30:
        new_score = 100
    elif util_rate < 0.60:
        new_score = 70
    else:
        new_score = 40

    internal = pay_score + debt_total + age_score + mix_score + new_score  # /1000

    return {
        'total': round(internal / 10, 1),
        'detail': {
            '연체 이력': {'score': round(pay_score  / 10, 1), 'max': 35},
            '부채 수준': {'score': round(debt_total / 10, 1), 'max': 30},
            '신용 기간': {'score': round(age_score  / 10, 1), 'max': 15},
            '신용 종류': {'score': round(mix_score  / 10, 1), 'max': 10},
            '신규 신용': {'score': round(new_score  / 10, 1), 'max': 10},
        }
    }
