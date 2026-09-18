"""Assemble the PNN report notebook from source, so the document is itself
reproducible.  Run `python build_notebook.py`, then execute the notebook.
"""

import pathlib

import nbformat as nbf

ROOT = pathlib.Path(__file__).parent
C = []


def md(src):
    C.append(nbf.v4.new_markdown_cell(src.strip("\n")))


def code(src):
    C.append(nbf.v4.new_code_cell(src.strip("\n")))


# The pre-analysis is embedded verbatim from the registered file rather than
# paraphrased, since a pre-registration rewritten after the results is not one.
PRE = (ROOT / "PRE_ANALYSIS.md").read_text(encoding="utf-8")
PRE_BODY = PRE.split("\n", 4)[4].strip()
PRE_QUOTED = "\n".join("> " + line if line else ">" for line in PRE_BODY.splitlines())

# =========================================================================== 0
md(r"""
# **Title of the project**
## **PNN — Behavioural Finance**

---

**Authors:** Alan Jesús Hernández Soto · Francisco Uriel Ledezma Chávez · Esteban Vega Campos · Santiago Villegas Baltazar  
**Programme:** Ingeniería Financiera  
**Institution:** ITESO — Universidad Jesuita de Guadalajara  
**Course:** Comportamiento en las Finanzas y Toma de Decisiones  
**Professor:** Luis Felipe Gómez Estrada  
**Date:** DD Month 2026

---
""")

md(r"""
<a id="s1"></a>

## 1. Pre-analysis statement
""")

md(PRE_QUOTED)

code(r"""
# A kernel left running keeps the version of the package it first imported, and
# "Run All" does not restart it, so an edit to src/ would otherwise go unseen.
%load_ext autoreload
%autoreload 2

import sys, pathlib

# Jupyter, VS Code and nbconvert all start the kernel in the notebook's folder.
HERE = pathlib.Path.cwd()
sys.path.insert(0, str(HERE / "src"))
(HERE / "results" / "figures").mkdir(parents=True, exist_ok=True)
""")

nb = nbf.v4.new_notebook(cells=C)
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.12.5"},
}
path = ROOT / "PNN_behavioral_finance.ipynb"
nbf.write(nb, str(path))
print(f"wrote {path.name} with {len(C)} cells")
