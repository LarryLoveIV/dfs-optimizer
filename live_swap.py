"""Mid-slate contest view: reads a DraftKings contest standings export and
shows where each of your entries stands, its full current lineup, which players
it has left, and the players still to play.

Mid-slate, DraftKings writes "LOCKED" in the standings for every player whose
game has NOT started yet (hidden, still swappable), in every entry including
yours, and leaves those players out of the ownership rows. Your own hidden
players are filled in from your DKEntries file when you upload it.

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
HIDDEN = "locked"          # norm("LOCKED"): DraftKings hides a player until his game starts
ID_RE = re.compile(r"^(.*?)\s*\((\d+)\)\s*$")


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
    # DraftKings lists a player once per roster slot he was drafted at (e.g. WR 34.25%
    # and FLEX 1.92%), so his ownership is the sum of his rows.
    pl = df[df.Player.notna()]
    for name, drafted, fpts in zip(pl.Player, pl["%Drafted"], pl.FPTS):
        p = players.setdefault(norm(name), dict(name=name.strip(), drafted=0.0,
                                                fpts=float(fpts or 0)))
        p["drafted"] += float(str(drafted).rstrip("%") or 0)
    return ent.reset_index(drop=True), players


def read_entries_full(data):
    """Everything in a DraftKings DKEntries.csv (the file DraftKings gives you for editing
    entries; it lists every player, including ones hidden in the standings).
    Returns dict(slots=[roster slots], entries={entry id: dict(contest, contest_id, fee,
    lineup=[(slot, norm name, display name, DraftKings ID)])}, ids={norm name: ID})."""
    import csv
    rows = list(csv.reader(io.StringIO(data.decode("utf-8-sig") if isinstance(data, bytes) else data)))
    if not rows or not rows[0] or rows[0][0].strip() != "Entry ID":
        raise ValueError("not a DraftKings entries file (first column should be Entry ID)")
    head = rows[0]
    end = next((i for i in range(4, len(head)) if not head[i].strip()), len(head))
    slots = [h.strip() for h in head[4:end]]
    entries, ids = {}, {}
    for r in rows[1:]:
        # The player list DraftKings appends on the right: Position, Name + ID, Name, ID, ...
        k = next((i for i in range(end, len(r) - 3) if ID_RE.match(r[i].strip())
                  and r[i + 2].strip().isdigit()), None)
        if k is not None:
            ids.setdefault(norm(r[k + 1]), r[k + 2].strip())
        if not r or not r[0].strip().isdigit():
            continue
        lineup = []
        for slot, cell in zip(slots, r[4:end]):
            m = ID_RE.match(cell.strip())
            name = (m.group(1) if m else cell).strip()
            if name:
                lineup.append((slot, norm(name), name, m.group(2) if m else ""))
        if lineup:
            entries[r[0].strip()] = dict(contest=r[1].strip(), contest_id=r[2].strip(),
                                         fee=r[3].strip(), lineup=lineup)
    for e in entries.values():
        for _, n, _, pid in e["lineup"]:
            if pid:
                ids.setdefault(n, pid)
    return dict(slots=slots, entries=entries, ids=ids)


def read_entries(data):
    """{entry id: [(slot, norm name, display name), ...]} from a DKEntries.csv."""
    full = read_entries_full(data)
    return {k: [(sl, n, nm) for sl, n, nm, _ in e["lineup"]] for k, e in full["entries"].items()}


SLOT_FITS = {"QB": {"QB"}, "RB": {"RB"}, "WR": {"WR"}, "TE": {"TE"}, "DST": {"DST"},
             "FLEX": {"RB", "WR", "TE"}}
OUT_TAGS = {"O", "OUT", "IR", "SUSP", "NA", "PUP", "D"}


def ruled_out(n, pool, live_tags=None):
    """Why a player can't score, or None. pool[n]: dict(status, proj); live_tags: norm -> DK tag."""
    p = pool.get(n)
    tag = str((live_tags or {}).get(n, "")).upper()
    if tag in OUT_TAGS:
        return f"{tag} on DraftKings"
    if p is None:
        return "not on the board"
    if str(p.get("status", "")).upper() == "O":
        return "Out on the board"
    if not p.get("target") or p["target"] <= 0.6:
        return "Out or no projection on the current board"
    return None


def swap_options(lineup, out_slots, pool, is_late, live_tags=None, per_slot=15, show=5):
    """Replacements for the ruled-out players in one lineup, keeping everyone else.

    lineup: [(slot, norm)]; out_slots: indexes into lineup to replace.
    pool[n]: dict(name, pos (set), salary, target, proj, own, kickoff, status).
    Returns dict(budget, best=[norm per out slot] or None, choices=[[norm...] per out slot])
    ranked by the build target (the board number the lineups are optimized on)."""
    import itertools
    keep = {n for i, (_, n) in enumerate(lineup) if i not in out_slots}
    budget = CAP - sum(pool.get(n, {}).get("salary", 0) for n in keep)
    lists = []
    for i in out_slots:
        fits = SLOT_FITS.get(lineup[i][0], set())
        c = [n for n, p in pool.items()
             if n not in keep and is_late(n) and p["pos"] & fits and p["salary"] <= budget
             and ruled_out(n, pool, live_tags) is None]
        c.sort(key=lambda n: -pool[n]["target"])
        lists.append(c[:per_slot])
    best, best_v = None, -1.0
    for combo in itertools.product(*lists):
        if len(set(combo)) < len(combo):
            continue
        if sum(pool[n]["salary"] for n in combo) > budget:
            continue
        v = sum(pool[n]["target"] for n in combo)
        if v > best_v:
            best, best_v = list(combo), v
    return dict(budget=budget, best=best, choices=[c[:show] for c in lists])


def swap_file(full, changes):
    """DraftKings upload file (same layout as DKEntries.csv) for the entries that change.
    changes: {entry id: {slot index: (norm name, display name)}}. Returns (csv text, problems)."""
    import csv
    out = io.StringIO()
    w = csv.writer(out)
    w.writerow(["Entry ID", "Contest Name", "Contest ID", "Entry Fee"] + full["slots"])
    problems = []
    for eid, ch in changes.items():
        e = full["entries"][eid]
        cells = []
        for i, (_, n, name, pid) in enumerate(e["lineup"]):
            if i in ch:
                new, new_name = ch[i]
                pid = full["ids"].get(new)
                if not pid:
                    problems.append(f"entry {eid}: no DraftKings ID for {new_name} in the "
                                    "entries file")
                    break
                cells.append(f"{new_name} ({pid})")
            else:
                cells.append(f"{name} ({pid})" if pid else name)
        else:
            w.writerow([eid, e["contest"], e["contest_id"], e["fee"]] + cells)
    return out.getvalue(), problems


def contest_detail(contest_id):
    import requests
    r = requests.get(f"https://api.draftkings.com/contests/v1/contests/{contest_id}?format=json",
                     timeout=20)
    r.raise_for_status()
    return json.loads(r.text)["contestDetail"]


def paid_places(detail):
    """Number of paid places in a DraftKings contest."""
    return max((t["maxPosition"] for t in detail.get("payoutSummary", [])), default=0)


def summarize(ent, players, me, is_late, salary, info, my_lineups=None, own_proj=None):
    """Everything the screen shows for one contest.

    is_late(norm name) -> True if his game hasn't started (swappable).
    salary: norm name -> salary. info: norm name -> dict(name, team, pos, proj, kickoff).
    my_lineups: {entry id: [(slot, norm, name)]} from your DKEntries file (optional).
    own_proj: norm name -> projected ownership % from the board (optional).
    """
    my_lineups = my_lineups or {}
    own_proj = own_proj or {}
    paid = info.get("_paid", 0)
    name_of = lambda n: (players[n]["name"] if n in players
                         else info.get(n, {}).get("name") or n)
    # Points so far = players whose game has started. Same as DraftKings' Points on a
    # live file; on a finished file it rebuilds the picture at the chosen time.
    ent = ent.assign(Points=[sum(players[n]["fpts"] for _, n in L if n in players and not is_late(n))
                             for L in ent.lineup])
    left_of = lambda L: sum(1 for _, n in L if n == HIDDEN or is_late(n))
    ent = ent.assign(left=[left_of(L) for L in ent.lineup])
    hidden_file = any(n == HIDDEN for L in ent.lineup for _, n in L)
    pts = ent.Points.to_numpy()
    order = sorted(pts, reverse=True)
    cash_line = order[min(paid, len(order)) - 1] if paid and order else None
    mine = ent[ent.user.str.lower() == me.lower()]
    top = ent.nlargest(max(1, len(ent) // 10), "Points")
    cashing = ent[ent.Points >= cash_line] if cash_line is not None else ent.iloc[:0]
    rows, lineups, my_full = [], {}, []
    for e in mine.itertuples():
        full = my_lineups.get(str(e.EntryId))
        L = [(sl, n) for sl, n, _ in full] if full else list(e.lineup)
        my_full.append(L)
        left = [n for _, n in L if n == HIDDEN or is_late(n)]
        hidden = sum(1 for n in left if n == HIDDEN)
        started_salary = sum(salary.get(n, 0) for _, n in L if n != HIDDEN and not is_late(n))
        proj_left = (None if hidden else
                     round(sum(info.get(n, {}).get("proj") or 0 for n in left), 1))
        rank = int((pts > e.Points + 1e-9).sum()) + 1
        rows.append({
            "Entry": e.EntryId, "Rank": rank, "Points": round(e.Points, 2),
            "To cash line": round(e.Points - cash_line, 2) if cash_line is not None else None,
            "Players left": len(left),
            "Proj. left": proj_left,
            "Proj. final": None if proj_left is None else round(e.Points + proj_left, 1),
            "Left to play": ", ".join("hidden" if n == HIDDEN else name_of(n) for n in left),
            "Salary left": None if hidden else int(CAP - sum(salary.get(n, 0) for _, n in L)),
            "Late-swap budget": int(CAP - started_salary),
            "Lineup from": "entries file" if full else "standings",
        })
        lines = []
        for sl, n in L:
            late = n == HIDDEN or is_late(n)
            i = info.get(n, {})
            lines.append({
                "Slot": sl, "Player": "hidden (game not started)" if n == HIDDEN else name_of(n),
                "Status": "To play" if late else "Started",
                "Kickoff": i.get("kickoff", ""),
                "Points so far": None if late else (players[n]["fpts"] if n in players else 0.0),
                "Salary": salary.get(n), "Proj": i.get("proj"),
                "Own %": (players[n]["drafted"] if n in players and players[n]["drafted"]
                          else own_proj.get(n)),
            })
        lineups[e.EntryId] = pd.DataFrame(lines)
    # How many players the field still has to play: where the leaders still have players
    # left, the cash line will keep moving.
    avg_left = lambda d: round(float(d.left.mean()), 1) if len(d) else None
    field_left = dict(field=avg_left(ent), cashing=avg_left(cashing), top=avg_left(top))
    # Players still to play: everyone on the board whose game hasn't started (the
    # standings leave them out mid-slate), plus anything named in a lineup.
    mine_sets = [set(n for _, n in L) for L in my_full]
    yours = lambda n: (sum(n in S for S in mine_sets) / len(mine_sets) * 100) if mine_sets else None
    share = lambda d, n: (d.lineup.map(lambda L: any(m == n for _, m in L)).mean() * 100
                          if len(d) else None)
    late_names = {n for n in info if n != "_paid" and is_late(n)}
    late_names |= {n for n in players if is_late(n)}
    own = []
    for n in late_names:
        i = info.get(n, {})
        field = players[n]["drafted"] if n in players and not hidden_file else None
        if not (i.get("proj") or field or own_proj.get(n) or yours(n)):
            continue
        row = {"Player": name_of(n), "Pos": i.get("pos", ""), "Team": i.get("team", ""),
               "Kickoff": i.get("kickoff", ""), "Salary": salary.get(n), "Proj": i.get("proj"),
               "Proj. own %": own_proj.get(n)}
        if not hidden_file:
            row.update({"Field %": field, "In the money now %": share(cashing, n),
                        "Top 10% now %": share(top, n)})
        row["Yours %"] = yours(n)
        own.append(row)
    return dict(rows=pd.DataFrame(rows), own=pd.DataFrame(own), cash_line=cash_line,
                mine=len(mine), lineups=lineups, hidden=hidden_file, field_left=field_left,
                filled=sum(1 for e in mine.itertuples() if str(e.EntryId) in my_lineups))
