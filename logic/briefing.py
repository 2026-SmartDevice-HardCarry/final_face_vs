def make_briefing(payload: dict) -> dict:
    p_now = payload["success_prob_now"]
    p_early = payload["success_prob_early"]
    depart_in = payload["recommend_depart_in_min"]
    precip = payload.get("precip_prob", 0.0)

    points = []
    points.append(f"{depart_in}분 후 출발 추천 (성공확률 {int(p_now*100)}% → {int(p_early*100)}%)")

    if precip >= 0.5:
        points.append(f"강수확률 {int(precip*100)}%: 우산 우선 확인")

    if payload.get("eta_min") is not None:
        points.append(f"버스 ETA 약 {payload['eta_min']}분")

    if payload.get("checklist_risks"):
        points.append("체크: " + "/".join(payload["checklist_risks"]))

    return {"summary": "오늘의 외출 리스크를 요약할게요.", "action_points": points[:3]}