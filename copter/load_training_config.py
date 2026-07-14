#!/usr/bin/env python3
"""Emit shell-sourceable KEY=VALUE assignments from training_config.yaml.

We avoid a hard PyYAML dependency: parse the small subset of YAML we use
(scalars + the single nested 'convergence' block).
"""
import os
import re
import sys


def _strip(s: str) -> str:
    s = s.split("#", 1)[0].strip()
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        s = s[1:-1]
    return s


def parse(path):
    flat = {}
    current_section = None
    with open(path) as f:
        for raw in f:
            line = raw.rstrip("\n")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            indent = len(line) - len(line.lstrip(" "))
            m = re.match(r"^([A-Za-z0-9_]+)\s*:\s*(.*)$", line.strip())
            if not m:
                continue
            key, val = m.group(1), _strip(m.group(2))
            if val == "" and indent == 0:
                current_section = key
                continue
            if indent == 0:
                current_section = None
                flat[key] = val
            else:
                if current_section is None:
                    continue
                flat[f"{current_section}_{key}"] = val
    return flat


def main():
    if len(sys.argv) < 2:
        sys.stderr.write("usage: load_training_config.py <config.yaml>\n")
        sys.exit(2)
    cfg = parse(sys.argv[1])
    # Emit KEY=VALUE lines for `eval` in bash.
    for k, v in cfg.items():
        # Quote the value safely for bash.
        qv = v.replace("'", "'\"'\"'")
        print(f"CFG_{k}='{qv}'")


if __name__ == "__main__":
    main()
