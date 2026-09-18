# PNN — Title of the project

One paragraph a reader outside the course can follow: the question, how the project
answers it, and the headline result once there is one.

---

## Contents

| File | What it is |
|---|---|
| [`PNN_behavioral_finance.ipynb`](PNN_behavioral_finance.ipynb) | The written report, with every number computed by its own code cells |
| [`PRE_ANALYSIS.md`](PRE_ANALYSIS.md) | What we expect, written before the first result, and never edited afterwards |
| [`build_notebook.py`](build_notebook.py) | Assembles the report notebook from source. **The notebook's source of truth** |
| [`src/studypkg/`](src/studypkg) | Code used only by this project |
| [`tests/`](tests) | The properties the report's conclusions rest on |
| [`results/`](results) | Tables as CSV and figures as PNG, regenerated on every run |

---

## Running it

From this folder:

```bash
python build_notebook.py
jupyter nbconvert --to notebook --execute --inplace PNN_behavioral_finance.ipynb
```
