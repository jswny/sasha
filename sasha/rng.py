from __future__ import annotations

import hashlib
from typing import Any


def stable_seed(*parts: Any) -> int:
    h = hashlib.blake2b(digest_size=8)
    for part in parts:
        h.update(str(part).encode("utf-8"))
        h.update(b"|")
    return int.from_bytes(h.digest(), "big")
