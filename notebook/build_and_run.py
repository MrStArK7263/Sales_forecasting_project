import re, sys, time, nbformat
from nbclient import NotebookClient
parts = ["nb_part1.py", "nb_part2.py", "nb_part3.py", "nb_part4.py"]
src = "\n".join(open(p).read() for p in parts)
cells, cur, kind = [], [], None
def flush():
    global cur, kind
    if kind is None: return
    body = "\n".join(cur).strip("\n")
    if kind == "md":
        body = "\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in body.split("\n"))
        cells.append(nbformat.v4.new_markdown_cell(body))
    elif body.strip():
        cells.append(nbformat.v4.new_code_cell(body))
    cur = []
for line in src.split("\n"):
    if line.startswith("# %%"):
        flush(); kind = "md" if "[markdown]" in line else "code"
    else: cur.append(line)
flush()
nb = nbformat.v4.new_notebook(cells=cells); nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
out = "Sales_Forecasting.ipynb"
if len(sys.argv) > 1 and sys.argv[1] == "build": nbformat.write(nb, out); print(len(cells), "cells"); sys.exit()
t0 = time.time()
def done(cell, cell_index, execute_reply):
    print(f"cell {cell_index+1}/{len(cells)} done {time.time()-t0:.0f}s status={execute_reply['content']['status']}", flush=True)
    nbformat.write(nb, out)
c = NotebookClient(nb, timeout=None, kernel_name="python3", resources={"metadata": {"path": "."}}, allow_errors=False, on_cell_executed=done)
try:
    c.execute()
finally:
    nbformat.write(nb, out)
print("FINISHED", time.time()-t0, flush=True)
