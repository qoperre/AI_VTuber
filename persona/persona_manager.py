# persona/persona_manager.py
from __future__ import annotations
import os, sys, re, json
from typing import Dict, Any, Optional, Tuple, List

# ── 이중 임포트 전략: (A) 패키지 컨텍스트 → (B) 폴더 컨텍스트 ──
try:
    # A) persona가 패키지로 인식되는 경우 (권장)
    from persona.emotion_engine import EmotionEngine, EmotionState
    from persona.memory_store import MemoryStore, make_key_discord, MemoryKey
    from persona.summarizer import simple_dialogue_summary
except Exception:
    # B) 스크립트가 persona 폴더를 sys.path에 추가한 경우
    CUR = os.path.dirname(os.path.abspath(__file__))
    if CUR not in sys.path:
        sys.path.append(CUR)
    from emotion_engine import EmotionEngine, EmotionState            # type: ignore
    from memory_store import MemoryStore, make_key_discord, MemoryKey # type: ignore
    from summarizer import simple_dialogue_summary                    # type: ignore


# ---- 명시적 사실(이름) 추출 룰 ----
NAME_PATTERNS = [
    r'내\s*이름은\s*["“”]?([A-Za-z가-힣0-9_]+)["“”]?\s*야',
    r'제\s*이름은\s*["“”]?([A-Za-z가-힣0-9_]+)["“”]?',
    r'(?:my|내)\s*name\s*is\s*["“”]?([A-Za-z0-9_가-힣]+)["“”]?',
    r'나는\s*["“”]?([A-Za-z가-힣0-9_]+)["“”]?\s*라고\s*해'
]
name_regexes = [re.compile(p, re.IGNORECASE) for p in NAME_PATTERNS]

class PersonaManager:
    """
    - EmotionEngine: v/a/i → mood 결정
    - MemoryStore: Profile/STM/Summary/Affinity
    - 프롬프트 구성 시: Known facts 요약 + 최근 STM N턴(혹은 summary) 삽입
    - step()은 Discord (user_id, guild_id) 스코프로 동작 (섞임 방지)
    """
    def __init__(self, rei_config_path: str, memory_store: MemoryStore, base_dir: Optional[str] = None):
        with open(rei_config_path, "r", encoding="utf-8") as f:
            self.cfg = json.load(f)
        self.engine = EmotionEngine(self.cfg)
        self.memory = memory_store
        self.base_dir = base_dir or os.getcwd()
        # Optional: comma-separated user IDs to log affinity changes for
        ids = os.getenv("AFFINITY_LOG_USER_IDS") or os.getenv("AFFINITY_LOG_USER_ID") or ""
        self._affinity_log_ids = {s.strip() for s in ids.split(",") if s.strip()}

    # ---------- 내부: 이름 학습 ----------
    def _maybe_learn_name(self, k: MemoryKey, user_text: str):
        t = (user_text or "").strip()
        for rx in name_regexes:
            m = rx.search(t)
            if m:
                name = m.group(1).strip()
                self.memory.remember_name(k, name, source="user_statement")
                break

    # ---------- 보이스/VTS ----------
    def select_voice(self, mood: str) -> Dict[str, Any]:
        vm = self.cfg["voice_map"].get(mood) or self.cfg["voice_map"]["neutral"]
        return dict(vm)

    def vts_params(self, mood: str) -> dict:
        return self.cfg["vts_parameters"].get(mood, {})

    # ---------- 프롬프트 ----------
    def _known_facts_line(self, k: MemoryKey) -> str:
        name = self.memory.get_name(k)
        facts = []
        if name: facts.append(f"name={name}")
        return ", ".join(facts) if facts else "none"

    def _recent_context_block(self, k: MemoryKey) -> str:
        # STM 우선, 너무 길면 간단 요약으로 대체
        turns = self.memory.get_history(k, n=6)
        if not turns:
            return ""
        # 대화체 그대로 넣되 길이 과하면 summarizer 사용
        joined = "\n".join(f"{r.upper()}: {t}" for (r, t) in turns)
        if len(joined) <= 800:
            return joined
        return "SUMMARY: " + simple_dialogue_summary(turns, max_chars=200)

    def build_prompt(self, k: MemoryKey, user_text: str) -> str:
        st = self.engine.state
        style_hint = self.cfg["prompt_prefs"]["style_hints"].get(st.mood, "")
        facts = self._known_facts_line(k)
        ctx = self._recent_context_block(k)

        base = f"""You are Rei, a friendly Japanese VTuber.
    - Tone: {style_hint}
    - Current mood: {st.mood} (intensity {st.intensity:.2f})
    - Known user facts: {facts}
    - Register: casual Japanese
    If the user asks for their name, answer using Known user facts.
    Reply in Japanese and keep the tone natural.

    You must respond ONLY in the following JSON format:
    {{
    "reply": "your message to the user (in Japanese)",
    "mood_delta": (float between -2 and +2),
    "affinity_delta": (integer between -3 and +3)
    }}

    Guidelines:
    - Increase affinity_delta when the user is kind, warm, funny, or caring.
    - Decrease affinity_delta when the user is rude, cold, distant, or hostile.
    - Keep 0 if neutral.
    - Positive mood_delta for cheerful tone, negative for sad or tired tone.
    - Do NOT add markdown, code fences, or text outside the JSON object.
    """


        if ctx:
            base += "\nRecent turns:\n" + ctx + "\n"
        return base + f"\nUSER: {user_text}\n"

    # ---------- 퍼소나 메인 스텝 ----------
    def step(self, user_id: str, guild_id: Optional[int], user_text: str, llm_call):
        k = make_key_discord(user_id=user_id, guild_id=guild_id)

        # (0) 사용자 발화 기록 + 이름 학습 시도
        self.memory.add_history(k, "user", user_text)
        self._maybe_learn_name(k, user_text)

        # (1) LLM 호출(JSON)
        prompt = self.build_prompt(k, user_text)
        llm_json = llm_call(prompt)  # {reply, mood_delta, affinity_delta}
        reply = str(llm_json.get("reply", ""))
        mood_delta = float(llm_json.get("mood_delta", 0))
        affinity_delta = int(llm_json.get("affinity_delta", 0))
        affinity_source = "llm"

        # Heuristic fallback: derive affinity change from sentiment if model left it neutral
        if affinity_delta == 0:
            try:
                dv, _ = self.engine._score_sentiment(user_text.lower())
            except Exception:
                dv = 0.0
            if dv >= 0.25:
                affinity_delta = 2
                affinity_source = "heuristic"
            elif dv >= 0.12:
                affinity_delta = 1
                affinity_source = "heuristic"
            elif dv <= -0.25:
                affinity_delta = -2
                affinity_source = "heuristic"
            elif dv <= -0.12:
                affinity_delta = -1
                affinity_source = "heuristic"

        # (2) Affinity 갱신 (+선택적 감쇠)
        prev_affinity = self.memory.get_affinity(k)
        affinity = self.memory.add_affinity(k, affinity_delta)
        if self._affinity_log_ids and (k.user_id in self._affinity_log_ids or "ALL" in self._affinity_log_ids):
            if affinity != prev_affinity:
                sign = "+" if affinity_delta >= 0 else ""
                print(f"[Affinity] user={k.user_id} scope={k.scope} delta={sign}{affinity_delta} ({affinity_source}) prev={prev_affinity} -> new={affinity}")
        # self.memory.decay_affinity(k)  # 필요시 활성화

        # (3) Emotion 업데이트
        new_state: EmotionState = self.engine.update(user_text, {"mood_delta": mood_delta}, affinity)

        # (4) 어시스턴트 응답 기록, STM/요약 관리
        self.memory.add_history(k, "assistant", reply)
        # 필요시 요약 갱신 로직을 여기에 붙일 수 있음(턴 카운트 기준)

        # (5) 출력(보이스/VTS)
        voice_cfg = self.select_voice(new_state.mood)
        vts = self.vts_params(new_state.mood)
        return {
            "reply": reply,
            "mood": new_state.mood,
            "intensity": new_state.intensity,
            "voice": voice_cfg,
            "voice_id": voice_cfg.get("voice_id") if isinstance(voice_cfg, dict) else voice_cfg,
            "vts_params": vts
        }
