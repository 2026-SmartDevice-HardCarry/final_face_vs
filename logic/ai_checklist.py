def order_checklist(base_items: list[str], miss_freq: dict[str,int], context: dict) -> list[str]:
    w = {it: 1.0 for it in base_items}

    if context.get("rain"):
        if "우산" in w:
            w["우산"] += 1.5
    if context.get("weekend"):
        pass

    scored = []
    for it in base_items:
        score = miss_freq.get(it, 0) * 1.0 + w.get(it, 1.0)
        scored.append((score, it))
    scored.sort(reverse=True)
    return [it for _, it in scored]