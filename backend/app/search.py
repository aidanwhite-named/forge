import math
import re
from collections import Counter

def tokens(value: str) -> list[str]:
    return re.findall(r"[\w가-힣]+", value.lower())

def retrieve(query: str, chunks: list[dict], limit: int = 5) -> list[dict]:
    q = Counter(tokens(query))
    scored = []
    for index, chunk in enumerate(chunks):
        c = Counter(tokens(chunk["text"]))
        bm25 = sum((qterm in c) * (1 + math.log1p(c[qterm])) for qterm in q)
        overlap = len(set(q) & set(c)) / max(1, len(set(q)))
        scored.append((bm25, overlap, index, chunk))
    bm25_rank = {item[2]: rank for rank, item in enumerate(sorted(scored, key=lambda x: (-x[0], x[2])), 1)}
    vector_rank = {item[2]: rank for rank, item in enumerate(sorted(scored, key=lambda x: (-x[1], x[2])), 1)}
    fused = [(1 / (60 + bm25_rank[i]) + 1 / (60 + vector_rank[i]), i, chunk) for _, _, i, chunk in scored]
    ranked = sorted(fused, key=lambda item: (-item[0], item[1]))
    return [chunk for score, _, chunk in ranked[:limit] if score > 0.02]

