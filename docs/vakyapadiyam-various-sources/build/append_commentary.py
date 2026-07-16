#!/usr/bin/env python3
"""
Append/merge commentary entries into a kanda commentary file.

Rationale: the commentary JSONs are large and are rewritten by tooling, so
anchor-based text edits are brittle. This merges a payload dict into the
target file, preserving existing entries unless explicitly overwritten.

Usage: python3 append_commentary.py <kanda-number> <payload.json>
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    k, payload_path = sys.argv[1], sys.argv[2]
    target = os.path.join(HERE, "commentary", "kanda%s.json" % k)

    with open(target, encoding="utf-8") as f:
        cur = json.load(f)
    with open(payload_path, encoding="utf-8") as f:
        new = json.load(f)

    added = [key for key in new if key not in cur]
    updated = [key for key in new if key in cur]
    cur.update(new)

    with open(target, "w", encoding="utf-8") as f:
        json.dump(cur, f, ensure_ascii=False, indent=1)

    print("kanda %s: +%d added, %d updated, %d total"
          % (k, len(added), len(updated), len(cur)))


if __name__ == "__main__":
    main()
