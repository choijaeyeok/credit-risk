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

MIN_INCOME  = 1_500_000
DSR_LIMIT   = 0.60
KRW_TO_USD  = 1300


def get_grade_and_rate(score):
    for threshold, grade, rate in SCORE_GRADE_TABLE:
        if score >= threshold:
            return grade, rate
    return "10등급", None


def prob_to_score(prob):
    """XGBoost 부도 확률(0~1)을 신용점수(0~100)로 변환 (구간별 선형 보간)."""
    knots = [
        (0.00, 100),
        (0.02,  94),
        (0.05,  89),
        (0.10,  83),
        (0.15,  77),
        (0.25,  70),
        (0.35,  63),
        (0.50,  53),
        (0.65,  45),
        (0.80,  34),
        (1.00,   0),
    ]
    for i in range(len(knots) - 1):
        p0, s0 = knots[i]
        p1, s1 = knots[i + 1]
        if prob <= p1:
            t = (prob - p0) / (p1 - p0)
            return round(s0 + t * (s1 - s0), 1)
    return 0.0
