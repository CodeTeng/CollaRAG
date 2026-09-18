#!/usr/bin/env python3
"""Cross-check the tool-permission matrix across the paper and the code.

Verifies three sources agree, so that a future edit to any one of them cannot
silently drift:

  1. Table 2 of ``main.tex``      -- the per-category summary matrix
  2. Appendix B of ``main.tex``   -- the per-tool "Agents" column
  3. ``agent_factory.TOOL_MATRIX``-- the code of record

Also checks that every LaTeX environment is balanced, and that the pool sizes
sum to 34.

Run:  python verify_tool_matrix.py
"""
import ast
import os
import re
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
TEX = os.path.join(HERE, "main.tex")
CODE = os.path.join(HERE, "..", "..", "..", "src", "chimera_rag",
                    "plugins", "colla_rag", "agent_factory.py")
AGENTS = ["PP", "SH", "MH", "SM", "OT"]

tex = open(TEX, encoding="utf-8").read()

# ---------- 1. Table 2: per-category counts ----------
t2_start = tex.index("Retrieval (6)")
t2_end = tex.index("\\textbf{Total}") + 120
rows = []
for line in tex[t2_start:t2_end].splitlines():
    m = re.match(r"\s*([A-Za-z][A-Za-z ]*)\((\d+)[^)]*\)\s*&(.+?)\\\\", line)
    if m:
        cat, pool, nums = m.group(1).strip(), int(m.group(2)), m.group(3)
        vals = [int(x) for x in re.findall(r"\d+", nums)]
        if len(vals) == 5:
            rows.append((cat, pool, vals))

m = re.search(r"\\textbf\{Total\}\s*&(.+?)\\\\", tex, re.S)
total_row = [int(x) for x in re.findall(r"\d+", m.group(1))]
table2 = dict(zip(AGENTS, total_row))

print("=== Table 2 (summary matrix) ===")
col = [0] * 5
for cat, pool, vals in rows:
    assert max(vals) <= pool, f"{cat}: an agent sees {max(vals)} > pool {pool}"
    for i in range(5):
        col[i] += vals[i]
    print(f"  {cat:<16} pool={pool:2d}  " + " ".join(f"{a}={v}" for a, v in zip(AGENTS, vals)))
pool_total = sum(p for _, p, _ in rows)
print(f"  pool total = {pool_total}   column sums = {dict(zip(AGENTS, col))}")
print("  PASS" if col == total_row and pool_total == 34 else "  FAIL")

# ---------- 2. Appendix B: per-tool agent lists ----------
appb = tex[tex.index("Full Tool Inventory"):]
tools, cur_cat = [], None
CAT_RE = re.compile(r"\\multirow\{\d+\}\{\*\}\{([^}]+)\}")
TOOL_RE = re.compile(r"\\tool\{([^}]+)\}")
for line in appb.splitlines():
    mc = CAT_RE.search(line)
    if mc:
        cur_cat = mc.group(1).strip()
        continue
    mt = TOOL_RE.search(line)
    if mt and line.rstrip().endswith("\\\\"):
        name = mt.group(1).replace("\\_", "_")
        agfield = line.split("&")[-1].rstrip("\\").strip()
        alist = AGENTS if agfield == "All" else [a.strip() for a in agfield.split(",")]
        tools.append((cur_cat, name, alist))

per_agent = Counter()
for _, _, alist in tools:
    per_agent.update(alist)

print("\n=== Appendix B (per-tool) ===")
print(f"  tool rows parsed: {len(tools)}")
print(f"  per-agent counts: {dict(per_agent)}")
print(f"  vs Table 2      : {table2}")
print("  PASS" if dict(per_agent) == table2 and len(tools) == 34 else "  FAIL")

# ---------- 3. code TOOL_MATRIX ----------
src = open(CODE, encoding="utf-8").read()
m = re.search(r"TOOL_MATRIX[^=]*=\s*(\{.*?\n\})", src, re.S)
matrix = ast.literal_eval(m.group(1))
slot2abbr = {"preprocessor": "PP", "single_hop": "SH", "multi_hop": "MH",
             "summarization": "SM", "other": "OT"}
code_counts = {slot2abbr[k]: len(v) for k, v in matrix.items()}

print("\n=== code TOOL_MATRIX ===")
for slot, v in matrix.items():
    print(f"  {slot:<15} {len(v)}")
print(f"  vs Table 2 : {table2}")
print("  PASS" if code_counts == table2 else "  FAIL")

# ---------- 4. LaTeX environment balance ----------
begins = Counter(re.findall(r"\\begin\{(\w+\*?)\}", tex))
ends = Counter(re.findall(r"\\end\{(\w+\*?)\}", tex))
bad = {k: (begins[k], ends[k]) for k in set(begins) | set(ends) if begins[k] != ends[k]}
print("\n=== LaTeX env balance ===")
print("  unbalanced:", bad if bad else "none")
