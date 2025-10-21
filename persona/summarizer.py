# persona/summarizer.py
from __future__ import annotations
from typing import List, Tuple

def simple_dialogue_summary(turns: List[Tuple[str, str]], max_chars: int = 180) -> str:
    """
    매우 보수적인 대화 요약:
    - 최근 대화의 목적/요청/응답 키워드를 한 줄로 묶음
    - 안전을 위해 추론 최소화 (실수 요약 방지)
    """
    if not turns:
        return ""
    # 최근 몇 개만 가볍게 연결
    clips = []
    for role, text in turns[-4:]:
        t = text.strip().replace("\n", " ")
        if len(t) > 60: t = t[:57] + "..."
        clips.append(f"{role}:{t}")
    s = " | ".join(clips)
    if len(s) > max_chars:
        s = s[:max_chars-3] + "..."
    return s
