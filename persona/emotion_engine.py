# persona/emotion_engine.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple
import math
import time
import re

@dataclass
class EmotionState:
    mood: str = "neutral"
    intensity: float = 0.3   # 0~1
    valence: float = 0.0     # -1~1
    arousal: float = 0.3     # 0~1
    last_update_ts: float = 0.0

def clamp(v, lo, hi): return max(lo, min(hi, v))

class EmotionEngine:
    """
    휴리스틱 기반 v1:
    state_{t+1} = decay(state_t) + sentiment(user) + llm_signal + transition(trigger)
    - decay: 시간 경과로 강도·각성 감소 (히스테리시스 완화)
    - sentiment(user): 키워드/이모지 기반 점수 → 감정별 가중 반영
    - llm_signal: LLM JSON의 mood_delta/affinity_delta를 보조 신호로 사용
    - transition: rei.json에 정의된 전이 테이블(트리거→목표감정) 우선 적용
    """
    def __init__(self, config: Dict):
        self.cfg = config
        self.state = EmotionState()
        self.cooldown_sec = self.cfg.get("engine", {}).get("cooldown_sec", 3.0)
        self.decay = self.cfg.get("engine", {}).get("decay", {"intensity":0.08,"arousal":0.05})
        self.keyword_map = self._compile_keywords(self.cfg.get("sentiment_keywords", {}))
        self.transition_rules = self.cfg.get("transitions", {})
        self.bias_by_affinity = self.cfg.get("engine", {}).get("affinity_bias", 0.002)

    def _compile_keywords(self, kw_cfg: Dict[str, List[str]]):
        compiled = {}
        for k, words in kw_cfg.items():
            compiled[k] = [re.compile(re.escape(w)) for w in words]
        return compiled

    def _score_sentiment(self, text: str) -> Tuple[float, float]:
        """
        text → (valence_delta, arousal_delta)
        간단 규칙:
        - 긍정 키워드 = +valence, 약간 +arousal
        - 부정 키워드 = -valence, +arousal (분노/공포는 각성↑)
        - 위로/공감 키워드 = +valence, -arousal(진정)
        """
        v, a = 0.0, 0.0
        def hit(key): 
            return any(p.search(text) for p in self.keyword_map.get(key, []))
        if hit("praise"): v += 0.25; a += 0.05
        if hit("insult"): v -= 0.35; a += 0.15
        if hit("comfort"): v += 0.15; a -= 0.05
        if hit("hype"):    v += 0.10; a += 0.20
        if hit("sadness"): v -= 0.20; a -= 0.05
        return v, a

    def _apply_decay(self, st: EmotionState, dt: float):
        st.intensity = clamp(st.intensity - self.decay["intensity"]*dt, 0.0, 1.0)
        st.arousal   = clamp(st.arousal   - self.decay["arousal"]  *dt, 0.0, 1.0)
        # valence는 천천히 0으로 회귀
        st.valence   = clamp(st.valence - 0.03*dt*math.copysign(1, st.valence), -1.0, 1.0) if abs(st.valence)>0.01 else 0.0

    def _pick_mood_from_pad(self, v: float, a: float, hint: str|None=None) -> str:
        """
        간단 매핑:
        - v↑, a↑ → happy/excited
        - v↑, a↓ → calm/happy
        - v↓, a↑ → angry/anxious
        - v↓, a↓ → sad/tired
        - hint가 있으면 우선 반영
        """
        if hint: 
            return hint
        if v >= 0.15 and a >= 0.55: return "excited"
        if v >= 0.10 and a <  0.55: return "happy"
        if v <= -0.15 and a >= 0.55: return "angry"
        if v <= -0.10 and a <  0.55: return "sad"
        if a < 0.25: return "tired"
        return "neutral"

    def _apply_transition_triggers(self, text: str) -> str|None:
        # rei.json의 transitions: {"insult->angry":[...키워드...] , ...}
        for rule, words in self.transition_rules.items():
            src, to = rule.split("->")
            for w in words:
                if w in text:
                    return to
        return None

    def update(self, user_text: str, llm_signal: Dict, affinity: int) -> EmotionState:
        now = time.time()
        st = self.state
        dt = max(0.0, now - (st.last_update_ts or now))
        st.last_update_ts = now

        # 1) decay
        self._apply_decay(st, dt)

        # 2) 사용자 발화 기반 감성
        dv, da = self._score_sentiment(user_text.lower())
        st.valence = clamp(st.valence + dv, -1.0, 1.0)
        st.arousal = clamp(st.arousal + da,  0.0, 1.0)
        st.intensity = clamp(st.intensity + (abs(dv)+abs(da))*0.4, 0.0, 1.0)

        # 3) LLM 신호 보조 (mood_delta: -2~+2 가정)
        md = float(llm_signal.get("mood_delta", 0.0))
        st.intensity = clamp(st.intensity + 0.15*md, 0.0, 1.0)
        st.valence   = clamp(st.valence   + 0.10*md, -1.0, 1.0)

        # 4) Affinity 편향 (친밀 높으면 긍정 바이어스)
        st.valence = clamp(st.valence + self.bias_by_affinity*affinity, -1.0, 1.0)

        # 5) 전이 트리거(강제 힌트)
        force_to = self._apply_transition_triggers(user_text.lower())
        new_mood = self._pick_mood_from_pad(st.valence, st.arousal, hint=force_to)

        # 6) 쿨다운: 너무 잦은 변화 방지 (같은 계열이면 교체 허용)
        if new_mood != st.mood and dt < self.cooldown_sec:
            same_family = {
                "happy":{"excited"}, "excited":{"happy"},
                "sad":{"tired"}, "tired":{"sad"}
            }
            if new_mood not in same_family.get(st.mood, set()):
                new_mood = st.mood

        st.mood = new_mood
        return st