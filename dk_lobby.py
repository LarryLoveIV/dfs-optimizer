"""Today's DraftKings NFL slates and player pools, via the draft_kings client.

The client reads DraftKings' public lobby endpoints without logging in. A
slate's pool is turned into the same CSV DraftKings' "Export to CSV" gives, so
the rest of the app can't tell the difference — except AvgPointsPerGame, which
the endpoint doesn't carry and is left at 0 (a projections file supplies the
numbers the optimizer actually uses).
"""
import csv
import io
import re
from collections import Counter, defaultdict
from zoneinfo import ZoneInfo

from draft_kings import Client
from draft_kings.data import Sport

CLASSIC, SHOWDOWN = 21, 96          # DraftKings contest_type_id
ET = ZoneInfo("America/New_York")
ROSTER = {"QB": "QB", "RB": "RB/FLEX", "WR": "WR/FLEX", "TE": "TE/FLEX", "DST": "DST"}
COLUMNS = ["Position", "Name + ID", "Name", "ID", "Roster Position", "Salary",
           "Game Info", "TeamAbbrev", "AvgPointsPerGame"]


def slates():
    """Open classic and showdown slates, soonest first.

    Each: {"id", "kind" ("classic"/"showdown"), "label", "games", "starts",
    "contests"}.
    """
    details = Client().contests(sport=Sport.NFL)
    by_group = defaultdict(list)
    for c in details.contests:
        by_group[c.draft_group_id].append(c)
    out = []
    for g in details.draft_groups:
        if g.contest_type_id not in (CLASSIC, SHOWDOWN):
            continue
        contests = by_group[g.draft_group_id]
        # Contest names end in the slate's tag — "(Early Only)", "(BAL @ ATL)" —
        # except most of the main slate's, which carry none (a few say
        # "(Winner Takes All)" and the like, so untagged has to win the vote).
        tags = Counter((m.group(1) if (m := re.search(r"\(([^()]+)\)\s*$", c.name))
                        else "Main") for c in contests)
        tag = tags.most_common(1)[0][0] if tags else "Main"
        starts = g.starts_at.astimezone(ET)
        kind = "classic" if g.contest_type_id == CLASSIC else "showdown"
        out.append({
            "id": g.draft_group_id, "kind": kind, "games": g.games_count,
            "starts": starts, "contests": len(contests),
            "label": (f"{tag} — {g.games_count} games" if kind == "classic" else tag)
                     + f" · {starts:%a %-I:%M %p} ET · {len(contests)} contests",
        })
    return sorted(out, key=lambda s: (s["starts"], -s["games"]))


def salaries_csv(draft_group_id, kind):
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
        slot = (("CPT" if p.draftable_id in cpt_ids else "FLEX") if kind == "showdown"
                else ROSTER.get(p.position_name, p.position_name))
        rows.append([p.position_name, f"{name} ({p.draftable_id})", name, p.draftable_id,
                     slot, int(p.salary), game, p.team_details.abbreviation, 0])
    rows.sort(key=lambda r: (r[4] != "CPT", -r[5]))
    w.writerows(rows)
    return buf.getvalue()
