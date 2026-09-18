#!/usr/bin/env python3
"""Cross-check citations in main.tex against references.bib.

Reports the bibliography size, any citation with no bib entry (would compile to
"?") and any bib entry that is never cited (dead weight in the .bbl).

Run:  python verify_references.py
"""
import os
import re
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
tex = open(os.path.join(HERE, "main.tex"), encoding="utf-8").read()
bib = open(os.path.join(HERE, "references.bib"), encoding="utf-8").read()

cited = set()
counter = Counter()
for m in re.finditer(r"cite[a-zA-Z]*\{([^}]*)\}", tex):
    for key in m.group(1).split(","):
        key = key.strip()
        if key:
            cited.add(key)
            counter[key] += 1

defined = set(re.findall(r"@\w+\{([^,]+),", bib))

print(f"bib entries        : {len(defined)}")
print(f"distinct keys cited: {len(cited)}")
print(f"total \\cite commands: {sum(counter.values())}")

missing = sorted(cited - defined)
unused = sorted(defined - cited)

print("\n[FATAL] cited but not defined:", missing if missing else "none")
print("[WARN ] defined but never cited:", unused if unused else "none")

print("\n--- citation counts (descending) ---")
for key, n in counter.most_common():
    print(f"  {n:2d}  {key}")

ok = not missing and not unused
print("\nRESULT:", "PASS" if ok else "CHECK ABOVE")
