# New UI

A dark, custom front end for the DFS optimizer. It runs the real `app.py` headlessly
(Streamlit's AppTest) and shows its controls, rules, results and downloads in a new page,
so every default, rule and lineup comes from the same code as the Streamlit app.

- Start: `./start.sh` (serves http://localhost:8510 and opens it). Stop: `./stop.sh`.
- `server.py`: holds one headless session of `app.py`, serves its state as JSON, applies
  page actions, saves the setup to `saved/` after every change and restores it on start.
- `index.html`: the page.
- `check_same.py`: confirms a no-randomness build matches a plain headless app run.

Projection edits in the player pool are applied to the uploaded cheatsheet in memory;
the file on disk is never changed.
