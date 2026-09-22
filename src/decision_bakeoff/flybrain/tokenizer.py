"""Word-level tokenizer. Saved with checkpoints; hash must match on load."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

_TOKEN_RE = re.compile(r"[A-Za-z0-9']+|[^\sA-Za-z0-9]")

PAD = "<pad>"
UNK = "<unk>"
BOS = "<bos>"
EOS = "<eos>"
SEP = "<sep>"


@dataclass(frozen=True)
class Tokenizer:
    """Deterministic word/punct split. Specials first, then vocab in sorted order."""

    token_to_id: dict[str, int]
    id_to_token: tuple[str, ...]

    @property
    def size(self) -> int:
        return len(self.id_to_token)

    @property
    def fingerprint(self) -> str:
        blob = "\n".join(self.id_to_token).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()[:16]

    def encode(self, text: str, *, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids: list[int] = []
        if add_bos:
            ids.append(self.token_to_id[BOS])
        for piece in _TOKEN_RE.findall(text.lower()):
            ids.append(self.token_to_id.get(piece, self.token_to_id[UNK]))
        if add_eos:
            ids.append(self.token_to_id[EOS])
        return ids

    def decode(self, ids: list[int] | tuple[int, ...]) -> str:
        parts: list[str] = []
        for i in ids:
            if i < 0 or i >= len(self.id_to_token):
                parts.append(UNK)
                continue
            tok = self.id_to_token[i]
            if tok in {PAD, BOS, EOS, SEP}:
                continue
            if tok == UNK:
                parts.append("?")
            else:
                parts.append(tok)
        out: list[str] = []
        for p in parts:
            if out and p in {".", ",", "!", "?", ";", ":"}:
                out[-1] = out[-1] + p
            else:
                out.append(p)
        return " ".join(out)

    def save(self, path: Path) -> None:
        path.write_text(
            json.dumps({"tokens": list(self.id_to_token), "fingerprint": self.fingerprint}, indent=2)
            + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path, *, expect_fingerprint: str | None = None) -> Tokenizer:
        data = json.loads(path.read_text(encoding="utf-8"))
        tokens = tuple(data["tokens"])
        tok = cls.from_tokens(tokens)
        if expect_fingerprint is not None and tok.fingerprint != expect_fingerprint:
            raise ValueError(
                f"tokenizer fingerprint mismatch: got {tok.fingerprint}, want {expect_fingerprint}"
            )
        stored = data.get("fingerprint")
        if stored and stored != tok.fingerprint:
            raise ValueError("tokenizer file fingerprint does not match token list")
        return tok

    @classmethod
    def from_tokens(cls, tokens: tuple[str, ...] | list[str]) -> Tokenizer:
        token_to_id = {t: i for i, t in enumerate(tokens)}
        for required in (PAD, UNK, BOS, EOS, SEP):
            if required not in token_to_id:
                raise ValueError(f"missing special token {required}")
        return cls(token_to_id=token_to_id, id_to_token=tuple(tokens))

    @classmethod
    def build(cls, texts: list[str], *, max_vocab: int = 8000) -> Tokenizer:
        counts: dict[str, int] = {}
        for text in texts:
            for piece in _TOKEN_RE.findall(text.lower()):
                counts[piece] = counts.get(piece, 0) + 1
        specials = [PAD, UNK, BOS, EOS, SEP]
        ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        room = max(0, max_vocab - len(specials))
        words = [w for w, _ in ranked[:room] if w not in specials]
        return cls.from_tokens(tuple(specials + words))
