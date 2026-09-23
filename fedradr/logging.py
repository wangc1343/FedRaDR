from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping


class JsonlLogger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, record: Mapping[str, object]) -> None:
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(dict(record), sort_keys=True) + "\n")
