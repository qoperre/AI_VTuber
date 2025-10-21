# persona/memory_store.py
from __future__ import annotations
import os, io, json, time, threading, re
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
from typing import Deque, Dict, List, Tuple, Optional, Any

# ==== 키(네임스페이스) ====
@dataclass(frozen=True)
class MemoryKey:
    platform: str           # "discord"
    scope: str              # "<guild_id>" or "DM"
    user_id: str            # discord user id as str

def make_key_discord(user_id: str, guild_id: Optional[int]) -> MemoryKey:
    scope = str(guild_id) if guild_id is not None else "DM"
    return MemoryKey(platform="discord", scope=scope, user_id=str(user_id))

# ==== 유틸 ====
def _safe_mkdir(p: str): 
    os.makedirs(p, exist_ok=True)

def _atomic_write_json(path: str, data: Any):
    tmp = path + ".tmp"
    with io.open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

def _load_json(path: str, default: Any):
    try:
        with io.open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

# ==== 메모리 저장소 ====
class MemoryStore:
    """
    - 유저별/길드별 완전 분리: 키=MemoryKey(platform, scope, user_id)
    - 타입 분리:
        Profile   : 사실형 (name 등), 각 필드에 confidence/source/ts/ttl
        STM       : 최근 N턴 대화 (deque)
        Summary   : STM 압축 1~2문장과 커버범위
        Affinity  : [-100,100] 점수, ts
    - 디스크 영속화: base_dir/mem/<platform>/<scope>/<user_id>/{profile,stm,summary,affinity}.json
    - 스레드 세이프: 파일 단위 락
    """
    def __init__(self, base_dir: str, stm_maxlen: int = 10):
        self.base_dir = base_dir
        self.stm_maxlen = stm_maxlen
        self._locks: Dict[str, threading.Lock] = defaultdict(threading.Lock)
        _safe_mkdir(self.base_dir)
        _safe_mkdir(os.path.join(self.base_dir, "mem"))

    # --------- 경로 & 락 ---------
    def _root(self, k: MemoryKey) -> str:
        p = os.path.join(self.base_dir, "mem", k.platform, k.scope, k.user_id)
        _safe_mkdir(p)
        return p
    def _lock(self, k: MemoryKey) -> threading.Lock:
        return self._locks[self._root(k)]

    # --------- 프로필(Profile) ---------
    def get_profile(self, k: MemoryKey) -> Dict[str, Any]:
        path = os.path.join(self._root(k), "profile.json")
        with self._lock(k):
            return _load_json(path, {})

    def set_profile_field(self, k: MemoryKey, field: str, value: Any, source: str = "system",
                          confidence: float = 0.95, ttl_sec: Optional[float] = None):
        path = os.path.join(self._root(k), "profile.json")
        with self._lock(k):
            prof = _load_json(path, {})
            now = time.time()
            prof[field] = {
                "value": value,
                "confidence": max(0.0, min(1.0, float(confidence))),
                "source": source,
                "ts": now,
                "ttl": ttl_sec
            }
            _atomic_write_json(path, prof)

    def get_profile_value(self, k: MemoryKey, field: str) -> Optional[Any]:
        prof = self.get_profile(k)
        entry = prof.get(field)
        if not entry:
            return None
        # ttl 검사
        ttl = entry.get("ttl")
        if ttl is not None and (time.time() - float(entry.get("ts", 0))) > ttl:
            return None
        return entry.get("value")

    # 편의: 이름 저장/가져오기
    def remember_name(self, k: MemoryKey, name: str, source="user_statement"):
        if not name: return
        name = name.strip()
        if len(name) > 64: return
        self.set_profile_field(k, "name", name, source=source, confidence=0.98)

    def get_name(self, k: MemoryKey) -> Optional[str]:
        v = self.get_profile_value(k, "name")
        return str(v) if v is not None else None

    # --------- STM (recent turns) ---------
    def _load_stm(self, k: MemoryKey) -> List[Tuple[str, str, float]]:
        path = os.path.join(self._root(k), "stm.json")
        return _load_json(path, [])

    def _save_stm(self, k: MemoryKey, turns: List[Tuple[str, str, float]]):
        path = os.path.join(self._root(k), "stm.json")
        _atomic_write_json(path, turns)

    def add_history(self, k: MemoryKey, role: str, text: str):
        path = os.path.join(self._root(k), "stm.json")
        with self._lock(k):
            data = _load_json(path, [])
            data.append([role, str(text), time.time()])
            # maxlen 유지
            if len(data) > self.stm_maxlen:
                data = data[-self.stm_maxlen:]
            _atomic_write_json(path, data)

    def get_history(self, k: MemoryKey, n: int = 6) -> List[Tuple[str, str]]:
        data = self._load_stm(k)
        data = data[-int(n):]
        return [(r, t) for (r, t, _) in data]

    # --------- Summary ---------
    def get_summary(self, k: MemoryKey) -> Dict[str, Any]:
        path = os.path.join(self._root(k), "summary.json")
        with self._lock(k):
            return _load_json(path, {})

    def set_summary(self, k: MemoryKey, text: str, covered_turns: int):
        path = os.path.join(self._root(k), "summary.json")
        with self._lock(k):
            _atomic_write_json(path, {
                "text": text,
                "covered_turns": int(covered_turns),
                "ts": time.time()
            })

    # --------- Affinity ---------
    def get_affinity(self, k: MemoryKey) -> int:
        path = os.path.join(self._root(k), "affinity.json")
        with self._lock(k):
            obj = _load_json(path, {"score": 0, "ts": 0})
            return int(obj.get("score", 0))

    def add_affinity(self, k: MemoryKey, delta: int) -> int:
        path = os.path.join(self._root(k), "affinity.json")
        with self._lock(k):
            obj = _load_json(path, {"score": 0, "ts": 0})
            score = int(obj.get("score", 0)) + int(delta)
            score = max(-100, min(100, score))
            obj["score"] = score
            obj["ts"] = time.time()
            _atomic_write_json(path, obj)
            return score

    def decay_affinity(self, k: MemoryKey, toward: float = 0.0, rate_per_sec: float = 0.001):
        """시간에 따라 affinity를 0에 살짝 수렴(선택)."""
        path = os.path.join(self._root(k), "affinity.json")
        with self._lock(k):
            obj = _load_json(path, {"score": 0, "ts": time.time()})
            now = time.time()
            last = float(obj.get("ts", now))
            dt = max(0.0, now - last)
            cur = float(obj.get("score", 0))
            # 선형 근사
            cur = cur + (toward - cur) * min(1.0, rate_per_sec * dt)
            obj["score"] = int(round(cur))
            obj["ts"] = now
            _atomic_write_json(path, obj)

    # --------- 관리/삭제 ---------
    def clear_user(self, k: MemoryKey):
        """해당 유저 네임스페이스 삭제(요청 시)."""
        root = self._root(k)
        with self._lock(k):
            for fn in ("profile.json", "stm.json", "summary.json", "affinity.json"):
                p = os.path.join(root, fn)
                try: os.remove(p)
                except OSError: pass
