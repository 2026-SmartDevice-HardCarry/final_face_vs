def laplace_prob(missed: int, total: int) -> float:
    return (missed + 1) / (total + 2)

def risk_level(p: float) -> str:
    if p >= 0.66: return "high"
    if p >= 0.33: return "mid"
    return "low"