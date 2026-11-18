"""Today's DraftKings NFL and NBA slates and player pools, via the draft_kings client.

The client reads DraftKings' public lobby endpoints without logging in. A
slate's pool is turned into the same CSV DraftKings' "Export to CSV" gives, so
the rest of the app can't tell the difference — except AvgPointsPerGame, which
the endpoint doesn't carry and is left at 0 (a projections file supplies the
numbers the optimizer actually uses).
"""
import csv
import io
import json
import os
import re
import time
from collections import Counter, defaultdict
from datetime import datetime
from zoneinfo import ZoneInfo

from draft_kings import Client
from draft_kings.data import Sport

CLASSIC, SHOWDOWN = 21, 96          # DraftKings contest_type_id (NFL)
# contest_type_id of the classic and showdown slates, by sport.
TYPES = {"NFL": (CLASSIC, SHOWDOWN), "NBA": (70, 81)}
SPORTS = {"NFL": Sport.NFL, "NBA": Sport.NBA}
ET = ZoneInfo("America/New_York")
ROSTER = {"QB": "QB", "RB": "RB/FLEX", "WR": "WR/FLEX", "TE": "TE/FLEX", "DST": "DST"}


def nba_roster(position):
    """'PG/SG' -> 'PG/SG/G/UTIL', the Roster Position DraftKings' NBA export uses."""
    ps = position.split("/")
    slots = list(ps)
    if {"PG", "SG"} & set(ps):
        slots.append("G")
    if {"SF", "PF"} & set(ps):
        slots.append("F")
    return "/".join(slots + ["UTIL"])
COLUMNS = ["Position", "Name + ID", "Name", "ID", "Roster Position", "Salary",
           "Game Info", "TeamAbbrev", "AvgPointsPerGame"]


def clock(t, prefix=""):
    """'Sun 1:00 PM' — the hour without a leading zero. strftime's %-I does
    that only on Mac/Linux, so the zero is stripped by hand for Windows."""
    return t.strftime(prefix) + t.strftime("%I:%M %p").lstrip("0")


def slates(sport="NFL"):
    """Open classic and showdown slates, soonest first.

    Each: {"id", "kind" ("classic"/"showdown"), "label", "games", "starts",
    "contests"}.
    """
    classic, showdown = TYPES[sport]
    details = Client().contests(sport=SPORTS[sport])
    by_group = defaultdict(list)
    for c in details.contests:
        by_group[c.draft_group_id].append(c)
    out = []
    for g in details.draft_groups:
        if g.contest_type_id not in (classic, showdown):
            continue
        contests = by_group[g.draft_group_id]
        # Contest names end in the slate's tag — "(Early Only)", "(BAL @ ATL)" —
        # except most of the main slate's, which carry none (a few say
        # "(Winner Takes All)" and the like, so untagged has to win the vote).
        tags = Counter((m.group(1) if (m := re.search(r"\(([^()]+)\)\s*$", c.name))
                        else "Main") for c in contests)
        tag = tags.most_common(1)[0][0] if tags else "Main"
        starts = g.starts_at.astimezone(ET)
        kind = "classic" if g.contest_type_id == classic else "showdown"
        out.append({
            "id": g.draft_group_id, "kind": kind, "games": g.games_count,
            "starts": starts, "contests": len(contests),
            "label": (f"{tag} — {g.games_count} games" if kind == "classic" else tag)
                     + f" · {clock(starts, '%a ')} ET · {len(contests)} contests",
        })
    return sorted(out, key=lambda s: (s["starts"], -s["games"]))


def contests(draft_group_id, sport="NFL"):
    """The slate's open contests, one per distinct name, fee and size, cheapest first.

    DraftKings runs many identical copies of a contest, and head-to-heads are left
    out. Each: {"id", "name", "fee", "field", "is_cash"}.
    """
    details = Client().contests(sport=SPORTS[sport])
    out = {}
    for c in details.contests:
        if c.draft_group_id != draft_group_id or c.is_head_to_head:
            continue
        e = c.entries_details
        out.setdefault((c.name, e.fee, e.maximum), {
            "id": c.contest_id, "name": c.name, "fee": e.fee, "field": e.maximum,
            "is_cash": bool(c.is_double_up or c.is_fifty_fifty)})
    return sorted(out.values(), key=lambda c: (c["fee"], c["field"], c["name"]))


SEEN_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "contests_seen.json")
SEEN_DAYS = 6          # a slate not opened for this long is dropped from the file


def contests_remembered(draft_group_id, sport="NFL", path=None):
    """(contests, ids of the ones no longer in the lobby) for a slate.

    DraftKings drops a contest from the lobby once it fills. Every contest seen
    for a slate is kept in a small file next to the app, so a filled one can
    still be picked. Slates not opened for SEEN_DAYS days are removed from the
    file, so last week's contests clear themselves out. If the file can't be
    read or written, this is just the live list.
    """
    live = contests(draft_group_id, sport)
    path = path or SEEN_FILE
    key = lambda c: f"{c['name']}|{c['fee']}|{c['field']}"
    try:
        try:
            with open(path, encoding="utf-8") as f:
                seen = json.load(f)
        except (FileNotFoundError, ValueError):
            seen = {}
        now = time.time()
        seen = {g: v for g, v in seen.items() if now - v.get("opened", 0) < SEEN_DAYS * 86400}
        group = seen.setdefault(str(draft_group_id), {"sport": sport, "contests": {}})
        group["opened"] = now
        for c in live:
            group["contests"][key(c)] = c
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(seen, f)
        os.replace(tmp, path)
        live_keys = {key(c) for c in live}
        gone = [c for k, c in group["contests"].items() if k not in live_keys]
    except Exception:
        return live, set()
    both = sorted(live + gone, key=lambda c: (c["fee"], c["field"], c["name"]))
    return both, {c["id"] for c in gone}


STARTED_HOURS = 10     # a slate that left the lobby stays pickable this long after its first game


def slates_remembered(sport="NFL", path=None):
    """slates(), plus slates that have started and left the lobby.

    DraftKings drops a slate from the lobby when its first game locks, but its
    player pool can still be read, which late swap needs. Every slate seen is
    kept in the same file as the contests; one that is gone from the lobby and
    started less than STARTED_HOURS ago is returned with "started": True.
    """
    live = slates(sport)
    path = path or SEEN_FILE
    try:
        try:
            with open(path, encoding="utf-8") as f:
                seen = json.load(f)
        except (FileNotFoundError, ValueError):
            seen = {}
        now = time.time()
        for s in live:
            group = seen.setdefault(str(s["id"]), {"sport": sport, "contests": {}})
            group["opened"] = now
            group["slate"] = {**s, "starts": s["starts"].isoformat()}
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(seen, f)
        os.replace(tmp, path)
        live_ids = {s["id"] for s in live}
        gone = []
        for g in seen.values():
            s = g.get("slate")
            if not s or g.get("sport") != sport or s["id"] in live_ids:
                continue
            starts = datetime.fromisoformat(s["starts"])
            if 0 <= now - starts.timestamp() < STARTED_HOURS * 3600:
                gone.append({**s, "starts": starts, "started": True,
                             "label": s["label"] + " · started"})
    except Exception:
        return live
    return sorted(live + gone, key=lambda s: (s["starts"], -s["games"]))


def contest_rules(contest_id):
    """What the lobby list doesn't carry: {"max_per_user", "winner_take_all"}.
    Winner-take-all means only first place is paid."""
    import requests
    url = f"https://api.draftkings.com/contests/v1/contests/{contest_id}?format=json"
    d = requests.get(url, timeout=15).json()["contestDetail"]
    paid = max(p["maxPosition"] for p in d["payoutSummary"])
    return {"max_per_user": int(d["maximumEntriesPerUser"]), "winner_take_all": paid == 1}


def salaries_csv(draft_group_id, kind, sport="NFL"):
    """The slate's player pool as DraftKings salary-export CSV text."""
    pool = Client().draftables(draft_group_id=draft_group_id).players
    pool = [p for p in pool if not p.is_disabled]          # game already locked
    copies = defaultdict(list)
    for p in pool:
        copies[p.player_id].append(p)
    cpt_ids = set()
    if kind == "showdown":
        # Each player is listed twice; the captain copy costs 1.5x.
        cpt_ids = {max(ps, key=lambda p: p.salary).draftable_id
                   for ps in copies.values() if len(ps) > 1}
    else:
        # Classic lists RB/WR/TE again under the FLEX slot with a second ID;
        # the export (and the upload) use the position slot's, the lower one.
        pool = [min(ps, key=lambda p: p.roster_slot_id) for ps in copies.values()]
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(COLUMNS)
    rows = []
    for p in pool:
        comp = p.competition_details
        game = (comp.name.replace(" @ ", "@").replace(" vs ", "@")
                + f" {comp.starts_at.astimezone(ET):%m/%d/%Y %I:%M%p} ET")
        name = p.name_details.display.strip()
        slot = (("CPT" if p.draftable_id in cpt_ids else "UTIL" if sport == "NBA" else "FLEX")
                if kind == "showdown"
                else nba_roster(p.position_name) if sport == "NBA"
                else ROSTER.get(p.position_name, p.position_name))
        rows.append([p.position_name, f"{name} ({p.draftable_id})", name, p.draftable_id,
                     slot, int(p.salary), game, p.team_details.abbreviation, 0])
    rows.sort(key=lambda r: (r[4] != "CPT", -r[5]))
    w.writerows(rows)
    return buf.getvalue()


def player_status(ids, kind, draft_group_id=None):
    """DraftKings' live injury tag for each draftable ID: {"123": "OUT", ...}.

    Tags are "None" (healthy), "Q", "D", "OUT", "IR" and the like. A salary file
    uploaded by hand has no draft group, so every open slate of the same kind is
    tried and the one sharing the most IDs wins. Read straight from the endpoint
    because the client drops the field.
    """
    import requests
    ids = {str(i) for i in ids}
    groups = ([draft_group_id] if draft_group_id
              else [s["id"] for s in slates() if s["kind"] == kind])
    best = {}
    for gid in groups:
        url = f"https://api.draftkings.com/draftgroups/v1/draftgroups/{gid}/draftables"
        pool = requests.get(url, timeout=15).json().get("draftables", [])
        found = {str(p["draftableId"]): p.get("status") or "None"
                 for p in pool if str(p["draftableId"]) in ids}
        if len(found) > len(best):
            best = found
        if len(best) == len(ids):
            break
    return best
