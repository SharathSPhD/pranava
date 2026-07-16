"""Client for the Saṃsādhanī morphological analyzer (Amba Kulkarni's SCL toolkit).

Talks to the local container's ``morph.cgi`` (JSON mode). Devanāgarī input is
transliterated to WX locally before the query. Only the morphological analyzer
is used here — the sentence segmenter/parser in this image depends on a Heritage
backend that is not installed (see research/01-samsaadhanii-integration.md).
"""
from __future__ import annotations

import json
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from functools import lru_cache

from indic_transliteration import sanscript

BASE = "http://localhost:8090"
MORPH_CGI = "/cgi-bin/scl/MT/prog/morph/morph.cgi"


def service_available(timeout: float = 2.0) -> bool:
    try:
        with urllib.request.urlopen(BASE + "/", timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


@dataclass(frozen=True, slots=True)
class MorphAnalysis:
    app: str  # "noun" | "verb" | "avyaya" | ...
    root: str  # RT field, in outencoding (IAST)
    root_wx: str  # rt field, in WX
    features: str  # ANS field, e.g. "{liṅgam:puṃ}{vibhaktiḥ:1}{vacanam:eka}"
    raw: str  # the raw JSON object, verbatim, for provenance

    @property
    def is_valid(self) -> bool:
        return bool(self.app and (self.root or self.root_wx))


@dataclass(slots=True)
class SamsaadhaniiClient:
    base: str = BASE
    outencoding: str = "IAST"
    timeout: float = 20.0
    _cache: dict[str, list[MorphAnalysis]] = field(default_factory=dict)

    @staticmethod
    def to_wx(devanagari: str) -> str:
        return sanscript.transliterate(devanagari.strip(), sanscript.DEVANAGARI, sanscript.WX)

    def analyze(self, word_devanagari: str) -> list[MorphAnalysis]:
        wx = self.to_wx(word_devanagari)
        if wx in self._cache:
            return self._cache[wx]
        result = self._analyze_wx(wx)
        self._cache[wx] = result
        return result

    def _analyze_wx(self, wx: str) -> list[MorphAnalysis]:
        if not wx:
            return []
        params = urllib.parse.urlencode(
            {"morfword": wx, "encoding": "WX", "outencoding": self.outencoding, "mode": "json"}
        )
        url = f"{self.base}{MORPH_CGI}?{params}"
        try:
            with urllib.request.urlopen(url, timeout=self.timeout) as r:
                body = r.read().decode("utf-8", errors="replace").strip()
        except Exception:
            return []
        if not body:
            return []
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            return []
        out: list[MorphAnalysis] = []
        for obj in data if isinstance(data, list) else [data]:
            if not isinstance(obj, dict):
                continue
            out.append(
                MorphAnalysis(
                    app=str(obj.get("APP", "")).strip(),
                    root=str(obj.get("RT", "")).strip(),
                    root_wx=str(obj.get("rt", "")).strip(),
                    features=str(obj.get("ANS", "")).strip(),
                    raw=json.dumps(obj, ensure_ascii=False),
                )
            )
        return out


@lru_cache(maxsize=1)
def _shared_client() -> SamsaadhaniiClient:
    return SamsaadhaniiClient()
