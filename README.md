# DFS Lineup Optimizer

Streamlit front end for [pydfs-lineup-optimizer](https://github.com/DimaKudosh/pydfs-lineup-optimizer),
built for DraftKings NFL — classic main slates and Showdown.

## Running locally

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Inputs

Both are uploaded in the app; nothing is committed to the repo.

| File | What it is |
|------|------------|
| **DK salaries CSV** | The DraftKings export. The classic slate needs the `Game Info` column for game-based rules; Showdown needs the `Roster Position` column (CPT/FLEX). |
| **Projections CSV** | Daily Fantasy Fuel cheatsheet format: `first_name`, `last_name`, `ppg_projection`, `injury_status`. Optional — without it lineups are built on the salary file's `AvgPointsPerGame`. |

When projections are supplied they replace `AvgPointsPerGame` as the optimization
target, and players are dropped if they have no projection, are listed `O`, or
project at or below 0.6.

## What it does

- **Player pool** — filter by position and team, lock/exclude/set exposure in bulk or per player
- **Opposing-team restrictions** — e.g. never roster a skill player against your own DST
- **Same-team restrictions** — e.g. never two RBs from one team
- **Per-team caps** — at most N players from a team unless you roster its QB
- **QB stacks** — QB + N teammates + M bring-back, mixable by share
- **Flex mix** — allocate the flex slot across positions by percentage
- **Contest settings** — field size drives projection randomness; global exposure cap; minimum unique players
- **Export** — DraftKings upload CSV with player IDs

## Solver

Ships an OR-Tools CP-SAT backend (`or_solver.py`) plugged into pydfs's public
`Solver` interface. Roughly 7x faster than the default PuLP/CBC path with
identical output; falls back to PuLP automatically if OR-Tools is unavailable or
a solve fails, since PuLP reports which constraints were violated.

## Deploying to Streamlit Cloud

1. Push this folder to GitHub.
2. On [share.streamlit.io](https://share.streamlit.io) create an app from the repo,
   main file `app.py`.
3. **Set Python to 3.12** under *Advanced settings*. OR-Tools has no Linux wheel
   for 3.13, and 3.14 is not offered — on either the app still runs, but it falls
   back to the PuLP solver and is roughly 7x slower.

`pydfs-lineup-optimizer` is published as a source distribution only; pip builds
it on install, which is fine as it is pure Python.

## Notes

- Share-based mixes (stack shapes, flex positions) run one optimizer pass per
  combination and concatenate — that batching is this app's, not the library's.
- Lineups are deterministic unless randomness is on, so re-running the same
  settings returns the same lineups.
