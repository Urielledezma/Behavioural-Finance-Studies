# Contributing

## Getting set up

```bash
git clone https://github.com/Urielledezma/Behavioural-Finance-Studies.git
cd Behavioural-Finance-Studies
pip install -r requirements.txt
python -m pytest -q
```

That is the whole setup. The results and executed notebooks are committed, so every
report can be read immediately; the tests confirm your environment reproduces the
properties the reports rest on.

## Branching

Work goes straight to `main`. There are no pull requests and no review gate: push a
piece when it is finished, and pull before you start the next one. With four people on
four projects that each live in their own folder, the coordination cost of a branch per
project buys less than it charges.

Branch only when the work can break a report already delivered: a change to a project's
`src/` that moves numbers already reported, or an experiment you are not confident in. Name it
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

Every project lives in `studies/pNN-<topic>/` and owns everything beneath it. Nothing in
one project's folder imports from another's.

| Kind of change | Where it belongs |
|---|---|
| A calculation, estimator or simulation step | the project's `src/<package>/`, with a test in its `tests/` |
| A parameter or seed | the project's config module, never inline in the report |
| Prose, tables and figures | the project's `build_notebook.py` |
| A property a conclusion depends on | the project's `tests/test_pNN_*.py` |
| A dependency | `requirements.txt` at the root, shared by all four projects |

Code is copied between projects rather than shared until a second project genuinely
needs the same helper. At that point it moves to one place with a test, in its own
commit, and both projects re-execute their reports.

## Starting a new project

1. **Copy the template** to a folder named after the project and its topic:

   ```bash
   cp -r studies/_template studies/p02-<topic>
   ```

2. **Rename the placeholders.** Replace `PNN` with the project number in the folder's
   README, `build_notebook.py` and the notebook name, and rename `src/studypkg/` to a
   package name no other project uses. Every project's `src/` goes on the path during
   testing, so two packages with the same name shadow each other.
3. **Write `PRE_ANALYSIS.md` and commit it before the first result exists.** The report
   quotes it verbatim, so the commit history is the proof it came first.
4. **Add the project to the table in the root README** in the same commit that creates
   the folder.

## Working on a project

1. **Pull first.** You are committing to `main` alongside three other people.
2. **Edit the prose in `build_notebook.py`, never in the `.ipynb`.** The notebook is
   regenerated from that file, so an edit made in Jupyter is overwritten on the next
   build and lost.
3. **Rebuild and execute from a clean kernel**, from the project's folder:

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

- [ ] `python -m pytest -q` passes, from the repository root.
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
