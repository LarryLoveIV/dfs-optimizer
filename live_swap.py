"""Mid-slate contest view: reads a DraftKings contest standings export and
shows where each of your entries stands, which players it has left, and the
real ownership of the players still to play.

Information only. A replay of 44 Sundays (2024-26, 478 contests) found that
swap suggestions from the simulator lost money versus holding, so this module
makes none.
"""
import io
import json
import re
import zipfile

import pandas as pd

SLOT_RE = re.compile(r"(?:^|\s)(QB|RB|WR|TE|FLEX|DST)\s+")
CAP = 50_000


def norm(n):
    s = re.sub(r"[^a-z ]", "", str(n).lower().replace("-", " ").replace(".", ""))
    return " ".join(w for w in s.split() if w not in {"jr", "sr", "ii", "iii", "iv", "v"})


def parse_lineup(s):
    """'DST Jaguars  FLEX Tee Higgins QB Joe Burrow ...' -> [(slot, name), ...]"""
    s = str(s)
    if not s.strip() or s.strip() == "nan":
        return []
    parts = SLOT_RE.split(" " + s)
    return [(slot, name.strip()) for slot, name in zip(parts[1::2], parts[2::2])]


def open_standings(name, data):
    """(contest id or None, text) from an uploaded standings .csv or DraftKings .zip."""
    m = re.search(r"(\d{6,})", name)
    if name.lower().endswith(".zip"):
        z = zipfile.ZipFile(io.BytesIO(data))
        inner = z.namelist()[0]
        m = re.search(r"(\d{6,})", inner) or m
        data = z.read(inner)
    return (int(m.group(1)) if m else None), io.StringIO(data.decode("utf-8-sig"))


def read_standings(text):
    """(entries, players). entries: EntryId, user, Points, lineup [(slot, norm name)].
    players: norm name -> dict(name, drafted %, fpts so far)."""
    df = pd.read_csv(text, dtype=str)
    ent = df[df.EntryId.notna()].copy()
    ent["user"] = ent.EntryName.str.replace(r"\s*\(\d+/\d+\)\s*$", "", regex=True).str.strip()
    ent["Points"] = pd.to_numeric(ent.Points, errors="coerce").fillna(0.0)
    ent["lineup"] = [[(sl, norm(n)) for sl, n in parse_lineup(x)] for x in ent.Lineup]
    players = {}
    pl = df[df.Player.notna()]
    for name, drafted, fpts in zip(pl.Player, pl["%Drafted"], pl.FPTS):
        players[norm(name)] = dict(name=name.strip(),
                                   drafted=float(str(drafted).rstrip("%") or 0),
                                   fpts=float(fpts or 0))
    return ent.reset_index(drop=True), players


def contest_detail(contest_id):
    import requests
    r = requests.get(f"https://api.draftkings.com/contests/v1/contests/{contest_id}?format=json",
                     timeout=20)
    r.raise_for_status()
    return json.loads(r.text)["contestDetail"]


def paid_places(detail):
    """Number of paid places in a DraftKings contest."""
    return max((t["maxPosition"] for t in detail.get("payoutSummary", [])), default=0)


def summarize(ent, players, me, is_late, salary, info):
    """Everything the screen shows for one contest.

    is_late(norm name) -> True if his game hasn't started (swappable).
    salary: norm name -> salary. info: norm name -> dict(team, pos, proj, kickoff).
    """
    paid = info.get("_paid", 0)
    # Points so far = players whose game has started. Same as DraftKings' Points on a
    # live file; on a finished file it rebuilds the picture at the chosen time.
    ent = ent.assign(Points=[sum(players[n]["fpts"] for _, n in L if n in players and not is_late(n))
                             for L in ent.lineup])
    pts = ent.Points.to_numpy()
    order = sorted(pts, reverse=True)
    cash_line = order[min(paid, len(order)) - 1] if paid and order else None
    mine = ent[ent.user.str.lower() == me.lower()]
    rows = []
    for e in mine.itertuples():
        left = [n for _, n in e.lineup if is_late(n)]
        used = sum(salary.get(n, 0) for _, n in e.lineup)
        rank = int((pts > e.Points + 1e-9).sum()) + 1
        rows.append({
            "Entry": e.EntryId, "Rank": rank, "Points": round(e.Points, 2),
            "To cash line": round(e.Points - cash_line, 2) if cash_line is not None else None,
            "Players left": len(left),
            "Left to play": ", ".join(players[n]["name"] if n in players else n for n in left),
            "Salary left": int(CAP - used),
        })
    # Ownership of players still to play: whole field, entries in the money now,
    # the current top 10%, and you.
    top = ent.nlargest(max(1, len(ent) // 10), "Points")
    cashing = ent[ent.Points >= cash_line] if cash_line is not None else ent.iloc[:0]
    share = lambda d, n: (d.lineup.map(lambda L: any(m == n for _, m in L)).mean() * 100
                          if len(d) else None)
    own = []
    for n, p in players.items():
        if not is_late(n):
            continue
        i = info.get(n, {})
        own.append({"Player": p["name"], "Pos": i.get("pos", ""), "Team": i.get("team", ""),
                    "Kickoff": i.get("kickoff", ""), "Salary": salary.get(n),
                    "Proj": i.get("proj"), "Field %": p["drafted"],
                    "In the money now %": share(cashing, n), "Top 10% now %": share(top, n),
                    "Yours %": share(mine, n)})
    return dict(rows=pd.DataFrame(rows), own=pd.DataFrame(own), cash_line=cash_line,
                mine=len(mine))
