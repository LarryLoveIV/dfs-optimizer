"""Lineup assistant: fixed checks on a finished build, and a chat model that
answers questions about it.

The checks are plain Python so they can't be talked out of a result — the model
gets them as facts and only explains them. The model itself runs through an
OpenAI-style chat endpoint, which both free options speak: Google's Gemini
(free API key, limited requests per day) and Ollama (a model on this machine).
"""
import csv
import io
import json

import pandas as pd
import requests

PROVIDERS = {
    "Gemini": {"url": "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
               "model": "gemini-3.8-flash", "needs_key": True},
    "Ollama": {"url": "http://localhost:11434/v1/chat/completions",
               "model": "qwen3:14b", "needs_key": False},
}

# DraftKings' injury tags. "None" is healthy; Q plays more often than not.
QUESTIONABLE = {"Q"}
NOT_PLAYING = {"O", "OUT", "IR", "D", "DOUBTFUL", "SUSP", "PUP", "NA", "NFI", "DND"}

SYSTEM = """You review DraftKings NFL lineups built by the user's optimizer and \
simulator. Answer only from the build data below; if it doesn't cover something \
(news, weather, a player not in the pool), say so instead of guessing.

The CHECKS section is computed by code and is authoritative: never say a lineup \
is clean if a check flags it, and never invent a problem the checks and data \
don't show. Live DraftKings status is the freshest injury source; the \
projections-file status may be older.

Terms: "Proj" is the number the optimizer maximized (often TwoGunGPT's Ceiling, \
not a median). "Sim avg/90th/99th" are simulated scores. "Top rate" is how often \
a lineup finished in the target top % of the simulated field. "Own" is projected \
ownership. A stack is a QB with his own WR/TE; a bring-back is an opposing WR/TE.

Be brief and concrete: name lineups by number and players by name, use short \
bullet lists, and lead with anything that needs fixing before lock."""


def opponents(csv_text):
    """{team: opponent} from the salary file's Game Info ("AWAY@HOME ...")."""
    opp = {}
    for row in csv.DictReader(io.StringIO(csv_text)):
        game = (row.get("Game Info") or "").split(" ")[0]
        if "@" in game:
            away, home = game.split("@", 1)
            opp[away], opp[home] = home, away
    return opp


def checks(lineups, merged, exposure, live, excluded, roster_size, budget):
    """Problems in a build, one row each: Lineup, Player, Issue, Severity.

    `live` is {draftable ID: DraftKings tag}, or None when the lookup failed.
    """
    picks = lineups[lineups["Position"] != "TOTAL"]
    by_name = merged.drop_duplicates("Name").set_index("Name")
    out = []

    def add(lineup, player, issue, severity):
        out.append({"Lineup": lineup, "Player": player, "Issue": issue,
                    "Severity": severity})

    for lid, grp in picks.groupby("Lineup", sort=True):
        if len(grp) != roster_size:
            add(lid, "", f"{len(grp)} players, roster needs {roster_size}", "Fix")
        salary = int(grp["Salary"].sum())
        if salary > budget:
            add(lid, "", f"salary ${salary:,} is over the ${budget:,} cap", "Fix")
        for name, n in grp["Name"].value_counts().items():
            if n > 1:
                add(lid, name, "same player in two slots", "Fix")
        if roster_size == 6 and grp["Team"].nunique() < 2:
            add(lid, "", "all six players from one team (DraftKings needs both)", "Fix")
        for _, p in grp.iterrows():
            name = p["Name"]
            tag = str((live or {}).get(str(p["ID"]), "None")).upper()
            if tag in NOT_PLAYING:
                add(lid, name, f"DraftKings lists him {tag} — not playing", "Fix")
            elif tag in QUESTIONABLE:
                add(lid, name, "DraftKings lists him Questionable", "Watch")
            info = by_name.loc[name] if name in by_name.index else None
            if info is None:
                continue
            file_tag = str(info.get("injury_status", "") or "").upper()
            if file_tag in {"O", "D"} and tag not in NOT_PLAYING:
                add(lid, name, f"projections file marks him {file_tag}", "Fix")
            proj = pd.to_numeric(info.get("ppg_projection"), errors="coerce")
            if "ppg_projection" in by_name.columns and (pd.isna(proj) or proj <= 0.6):
                add(lid, name, "no real projection (missing or near 0)", "Fix")
            if name in excluded:
                add(lid, name, "player you excluded is in this lineup", "Fix")

    sets = picks.groupby("Lineup")["ID"].apply(lambda s: frozenset(map(str, s)))
    seen = {}
    for lid, ids in sets.items():
        if ids in seen:
            add(lid, "", f"identical to lineup {seen[ids]}", "Fix")
        else:
            seen[ids] = lid

    if exposure is not None and "Max set %" in exposure.columns:
        over = exposure[pd.to_numeric(exposure["Max set %"], errors="coerce")
                        .lt(exposure["Exposure %"] - 0.5)]
        for _, r in over.iterrows():
            add("all", r["Name"], f"exposure {r['Exposure %']:.0f}% is above your "
                f"{r['Max set %']:.0f}% max", "Watch")

    df = pd.DataFrame(out, columns=["Lineup", "Player", "Issue", "Severity"])
    df["Lineup"] = df["Lineup"].astype(str)
    return df.sort_values("Severity", key=lambda s: s.ne("Fix"), kind="stable")


def stack_label(grp, pos_of, opp):
    """'BUF QB+2 (WR, TE) +1 bring-back' for a classic lineup."""
    qb = grp[grp["Name"].map(pos_of).eq("QB")]
    if qb.empty:
        return "no QB"
    team = qb.iloc[0]["Team"]
    catchers = grp[grp["Name"].map(pos_of).isin(["WR", "TE"])]
    mates = catchers[catchers["Team"] == team]
    back = catchers[catchers["Team"] == opp.get(team)]
    rbs = grp[(grp["Name"].map(pos_of) == "RB") & (grp["Team"] == team)]
    label = f"{team} QB+{len(mates)}"
    if len(mates):
        label += " (" + ", ".join(mates["Name"].map(pos_of)) + ")"
    label += f" +{len(back)} bring-back"
    if len(rbs):
        label += f", +{len(rbs)} own RB"
    return label


def context(results, merged, found, live_ok, csv_text, is_showdown, target):
    """The build as compact text for the model: summary, checks, every lineup."""
    lineups = results["lineups"]
    picks = lineups[lineups["Position"] != "TOTAL"]
    pos_of = dict(zip(merged["Name"], merged["Position"].astype(str).str.split("/").str[0]))
    exposure = results["exposure"]
    own = (dict(zip(exposure["Name"], exposure["Proj own %"]))
           if "Proj own %" in exposure.columns else {})
    opp = opponents(csv_text)
    sim_rows = ({int(r["Lineup"]): r for _, r in results["sim"].iterrows()}
                if results.get("sim") is not None else {})

    parts = [f"BUILD: {lineups['Lineup'].max()} {'showdown' if is_showdown else 'classic'} "
             f"lineups, optimized on {target}, built at {results.get('built_at')}."]
    note = results.get("sim_note")
    if note:
        parts.append(f"Simulation: picked from {note['candidates']} candidates over "
                     f"{note['runs']:,} sims. Chance at least one lineup finishes top "
                     f"{note['top']:.1%}: {note['covered']:.0%} (optimizer alone would be "
                     f"{note['baseline']:.0%}).")
    parts.append("\nCHECKS (" + ("live DraftKings status included" if live_ok
                                 else "live DraftKings status UNAVAILABLE") + "):")
    parts += ([f"- Lineup {r.Lineup}: {r.Player + ' — ' if r.Player else ''}{r.Issue} "
               f"[{r.Severity}]" for r in found.itertuples()] or ["- none found"])

    parts.append("\nLINEUPS:")
    totals = lineups[lineups["Position"] == "TOTAL"].set_index("Lineup")
    for lid, grp in picks.groupby("Lineup", sort=True):
        head = (f"L{lid}: salary ${int(totals.loc[lid, 'Salary']):,}, "
                f"proj {totals.loc[lid, 'AvgPoints']}")
        s = sim_rows.get(int(lid))
        if s is not None:
            head += "".join(f", {k.lower()} {s[k]}" for k in s.index if k != "Lineup")
        if not is_showdown:
            head += f", stack {stack_label(grp, pos_of, opp)}"
        players = [f"{p.Position} {p.Name} ({pos_of.get(p.Name, '?')} {p.Team}, "
                   f"${int(p.Salary):,}, proj {p.AvgPoints}"
                   + (f", own {own[p.Name]}%" if pd.notna(own.get(p.Name)) else "") + ")"
                   for p in grp.itertuples()]
        parts.append(head + "\n  " + "; ".join(players))

    for title, key in (("EXPOSURE", "exposure"), ("TEAM EXPOSURE", "teams"),
                       ("CAPTAINS", "captains"), ("STACK MIX", "mix"),
                       ("FLEX/CAPTAIN POSITION", "flex"), ("PLAYER COMBOS", "combos")):
        df = results.get(key)
        if df is not None and len(df):
            parts.append(f"\n{title}:\n" + df.to_csv(index=False).strip())
    return "\n".join(parts)


def stream(provider, model, key, messages):
    """Yield the model's reply as it arrives."""
    cfg = PROVIDERS[provider]
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    try:
        r = requests.post(cfg["url"], headers=headers, stream=True, timeout=180,
                          json={"model": model, "messages": messages, "stream": True})
    except requests.ConnectionError:
        raise RuntimeError(
            "Ollama isn't running on this computer. Install it from ollama.com, open "
            f"it, and run `ollama pull {model}` once." if provider == "Ollama"
            else f"couldn't reach {provider} — check the internet connection.") from None
    with r:
        if r.status_code == 404 and provider == "Ollama":
            raise RuntimeError(f"Ollama doesn't have {model} yet. Run "
                               f"`ollama pull {model}` in a terminal, then ask again.")
        if r.status_code != 200:
            raise RuntimeError(f"{provider} answered {r.status_code}: {r.text[:300]}")
        for line in r.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            delta = json.loads(data)["choices"][0].get("delta") or {}
            if delta.get("content"):
                yield delta["content"]
