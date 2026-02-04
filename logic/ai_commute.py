import math

def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))

def success_prob(congestion_0_1: float, late_count_7days: int, depart_delay_min: float) -> dict:
    # 설명가능 가중합 -> 확률
    risk = (
        0.4 * float(congestion_0_1) +
        0.3 * min(late_count_7days / 7.0, 1.0) +
        0.3 * min(max(depart_delay_min, 0.0) / 15.0, 1.0)
    )
    score = 2.0 * (1.0 - risk) - 1.0
    p = sigmoid(2.2 * score)

    factors = []
    if congestion_0_1 >= 0.6: factors.append("교통 혼잡")
    if late_count_7days >= 2: factors.append("최근 지각 빈도")
    if depart_delay_min > 0: factors.append("늦은 출발")

    return {"risk": round(risk, 3), "p": round(p, 3), "factors": factors}