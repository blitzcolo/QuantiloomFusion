"""Minimal deterministic TOML writer with a round-trip self-check."""

import json
import math
import tomllib


def scalar(value):
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (float, int)):
        if not math.isfinite(value):
            raise ValueError("nonfinite TOML value")
        return repr(value)
    if isinstance(value, list):
        return "[" + ", ".join(scalar(v) for v in value) + "]"
    raise ValueError(f"unsupported TOML value: {value!r}")


def toml_text(doc: dict) -> str:
    lines = []
    def emit(table, keys, array=False):
        if keys:
            path = ".".join(json.dumps(k) for k in keys)
            lines.append(("[[" if array else "[") + path + ("]]" if array else "]"))
        for k, v in table.items():
            if not isinstance(v, dict) and not (isinstance(v, list) and v and isinstance(v[0], dict)):
                lines.append(json.dumps(k) + " = " + scalar(v))
        lines.append("")
        for k, v in table.items():
            if isinstance(v, dict):
                emit(v, keys + [k])
            elif isinstance(v, list) and v and isinstance(v[0], dict):
                for entry in v:
                    emit(entry, keys + [k], True)
    emit(doc, [])
    text = "\n".join(lines)
    # A serializer mistake must fail before a rendering task is dispatched.
    if tomllib.loads(text) != doc:
        raise ValueError("TOML round-trip mismatch")
    return text
