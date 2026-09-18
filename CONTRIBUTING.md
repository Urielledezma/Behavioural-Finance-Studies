# Contributing

## Getting set up

```bash
git clone https://github.com/Urielledezma/Behavioural-Finance-Studies.git
cd Behavioural-Finance-Studies
pip install -r requirements.txt
python -m pytest -q tests
```

That is the whole setup. The results and executed notebooks are committed, so every
report can be read immediately; the tests confirm your environment reproduces the
properties the reports rest on.

## Branching

Work goes straight to `main`. There are no pull requests and no review gate: push a
piece when it is finished, and pull before you start the next one. With four people on
studies that touch mostly separate folders, the coordination cost of a branch per study
buys less than it charges.

Branch only when the work can break a report for everyone else: a change to `src/` that
moves numbers already reported, or an experiment you are not confident in. Name it
`fix/…`, `feat/…` or `exp/…`, merge it yourself once the tests pass and the notebook
executes clean, and delete it.

One rule survives from the branch-per-study model, because it is the one that actually
bites: **only one person commits an executed notebook at a time.** Two people committing
notebook outputs from divergent copies of `main` produces a conflict in thousands of
machine-written lines. Say in the group chat that you are executing, and pull
immediately afterwards.

## Commits

[Conventional Commits](https://www.conventionalcommits.org/): `feat:`, `fix:`, `docs:`,
`test:`, `chore:`, `refactor:`, `perf:`, `ci:`.

Keep them small and self-contained, one idea per commit. The body is where the reasoning
goes: not what changed, which the diff already says, but why this approach and not the
obvious alternative. A commit that changes a number in a report explains what moved it.

## Where code goes

| Kind of change | Where it belongs |
|---|---|
| A reusable calculation, estimator or simulation step | `src/`, with a test in `tests/` |
| A parameter or seed | `src/bfsim/config.py`, never inline |
| Prose, tables and figures for a study | that study's `build_notebook.py` |
| A property a conclusion depends on | `tests/test_<study>_*.py` |

## Working on a study

1. **Pull first.** You are committing to `main` alongside three other people.
2. **Edit the prose in `build_notebook.py`, never in the `.ipynb`.** The notebook is
   regenerated from that file, so an edit made in Jupyter is overwritten on the next
   build and lost.
3. **Rebuild and execute from a clean kernel**, from the study's folder:

   ```bash
   python build_notebook.py
   jupyter nbconvert --to notebook --execute --inplace <notebook>.ipynb
   ```

   In VS Code or Jupyter, the equivalent is **Restart**, then **Run All**. "Run All" on
   its own keeps whatever the kernel imported earlier. The notebooks load `autoreload`
   to guard against that, but a clean kernel is the only run whose numbers can be quoted.
4. **Check that every number quoted in the prose matches the output beside it.** A
   change to `src/` can move a figure without touching the sentence that cites it.

## Before you push

Nobody is going to catch these for you, so run them yourself.

- [ ] `python -m pytest -q tests` passes.
- [ ] The notebook executes from a clean kernel with no errors.
- [ ] No credential, token or `.env` file is staged. Check `git diff --cached`.
- [ ] Every reported number has an interpretation in the prose beside it.
- [ ] Every figure with more than one series is accompanied by its table.
- [ ] Sources are listed in the report's References section.

## Style

Python follows [PEP 8](https://peps.python.org/pep-0008/) with a 100-character line.
Module-level constants are `UPPER_CASE`; functions and variables `snake_case`. Code,
comments, documentation and commit messages are in English.

## Reviewing

There is no review gate, so the reading happens in two places instead: on your own diff
before you push, and on a teammate's commit when a number in it surprises you. Both use
the same questions.

A review checks the reasoning, not only the syntax. The questions worth asking are
whether a simulated agent uses information it could not have had at the time, whether
the sign convention holds, whether the number in the table matches the number in the
sentence, and whether a reader who disagrees with the conclusion could find the evidence
to argue with it.
