import streamlit as st
import pandas as pd
import tempfile
import os
import re
import io
from collections import Counter, defaultdict
from pydfs_lineup_optimizer import (get_optimizer, Site, Sport, LineupOptimizerException,
                                    TotalExposureStrategy, AfterEachExposureStrategy,
                                    PlayersGroup, Stack, RandomFantasyPointsStrategy)
# NestedPlayersGroup isn't re-exported from the package root, so it has to come
# from the module that defines it.
from pydfs_lineup_optimizer.stacks import NestedPlayersGroup
from pydfs_lineup_optimizer.rules import OptimizerRule
from pydfs_lineup_optimizer.solvers import SolverSign

try:
    # Same model, faster engine. Verified to produce identical lineups; if the
    # import fails the app just falls back to the library's default solver.
    from or_solver import ORToolsSat as FAST_SOLVER
except Exception:
    FAST_SOLVER = None

# Jitter applied to each projection, keyed off the contest's actual field size.
#
# Boundaries come from how the field-size numbers are actually used in practice
# rather than from any standard definition (there isn't one):
#   - Establish The Run's game-selection work treats ~1.1k-6k as the small-field
#     zone it prefers (a 1,136-entry $50 Red Zone and a 1,960-entry $15 are cited
#     as small-field; 2,450 is preferred over 9,803 and 19,607; 5,882 is called
#     "about as small as possible" for a GPP at that buy-in).
#   - DK's Flea Flicker, described as mid-tier, runs ~8k-24k entries.
#   - The Milly Maker is 150k+; a recent Football Millionaire drew 832k.
#
# One Week Season frames the same split by variance rather than headcount: in
# small fields and cash you manage variance and build tight rosters; in large
# top-heavy fields you accept variance to win outright. The ladder below follows
# that shape.
#
# The percentages themselves come from optimizer guidance (Footballguys,
# Stokastic) putting the usable band at 2-5% with 5% a ceiling. Neither ETR nor
# OWS prescribes an optimizer randomness number. Measured on a 100-lineup run
# with this pool: 2-5% costs ~0.1 projection points while cutting top-player
# exposure from 85% to 70%; 12% costs a full point and 25% costs 3.4.
FIELD_RANDOMNESS = [
    (500,          0.01, "very small — manage variance, build the tightest roster"),
    (6_000,        0.02, "small field — ETR's preferred zone"),
    (50_000,       0.03, "mid field — Flea Flicker territory"),
    (150_000,      0.04, "large field — accept variance to win outright"),
    (float("inf"), 0.05, "Milly Maker scale"),
]


def randomness_for_field(entries):
    for limit, dev, note in FIELD_RANDOMNESS:
        if entries <= limit:
            return dev, note
    return 0.05, ""


st.set_page_config(page_title="DFS Lineup Optimizer", layout="wide")
st.title("🏈 DFS Lineup Optimizer")

st.sidebar.header("Settings")
site = st.sidebar.selectbox("Site", ["DraftKings", "DraftKings Showdown"])
is_showdown = site == "DraftKings Showdown"
sport = st.sidebar.selectbox("Sport", ["NFL", "NBA", "MLB", "NHL"])
# Showdown is a 6-man roster from one game, so the usable pool is tiny compared
# with a main slate. Fewer lineups, min-unique on, and no global cap — with ~24
# players some are unavoidable, so a cap just fails silently.
num_lineups = st.sidebar.number_input(
    "Number of lineups", min_value=1, max_value=150,
    value=20 if is_showdown else 100, step=5 if is_showdown else 10)

st.sidebar.subheader("Exposure")
global_max_exp = st.sidebar.slider(
    "Global max exposure %", min_value=0, max_value=100, value=50, step=5,
    help=("Measured on a 24-player showdown pool: 50% cuts lineup overlap at no cost "
          "to projection, while tighter caps get expensive fast (35% costs ~3 points, "
          "25% costs ~9)."
          if is_showdown else
          "Default cap for every player. 100% means no cap. Per-player values override "
          "this. 50% costs nothing in projection here and stops one player carrying "
          "most of your entries."),
)
min_unique = st.sidebar.number_input(
    "Min unique players per lineup", min_value=0, max_value=5,
    value=2 if is_showdown else 0, step=1,
    help="0 is off. At 2, no two lineups may share more than roster-2 players — it "
         "removes near-twin lineups that differ by a single name. On showdown this is "
         "the main diversification lever, since exposure caps can't bind on a small "
         "pool.",
)

strategy_label = st.sidebar.selectbox(
    "Exposure strategy", ["Total", "After each lineup"],
    help="Total: share of all lineups. After each lineup: share of lineups generated so far.",
)
strategy_map = {"Total": TotalExposureStrategy, "After each lineup": AfterEachExposureStrategy}

st.sidebar.subheader("Contest")
contest_mode = st.sidebar.radio(
    "Type", ["Tournament (GPP)", "Cash / single entry", "Custom randomness"],
    help="Randomness is jitter added to each projection before every solve, so "
         "lineups stop repeating. Bigger fields justify more of it.",
)
if contest_mode == "Cash / single entry":
    deviation = 0.0
    st.sidebar.caption("No jitter — you want the single best lineup, and every run "
                       "returns the same one.")
elif contest_mode == "Custom randomness":
    deviation = st.sidebar.slider("Randomness %", 0, 25, 3, step=1) / 100
    if deviation > 0.05:
        st.sidebar.warning(
            "Above 5% is beyond what most DFS guidance recommends — it trades real "
            "projection for diversity. Exposure caps are usually the better lever."
        )
else:
    field_entries = st.sidebar.number_input(
        "Field size (entries)", min_value=2, max_value=2_000_000, value=5_000,
        step=500, help="The contest's entry count. Check the lobby if unsure.",
    )
    deviation, field_note = randomness_for_field(field_entries)
    st.sidebar.caption(
        f"{field_entries:,} entries — {field_note}. Applying **{deviation:.0%}** "
        "randomness: each projection is multiplied by 1 ± a random draw up to that, "
        "redrawn per lineup. Displayed projections stay the real ones."
    )

site_map = {"DraftKings": Site.DRAFTKINGS,
            "DraftKings Showdown": Site.DRAFTKINGS_CAPTAIN_MODE}
sport_map = {"NFL": Sport.FOOTBALL, "NBA": Sport.BASKETBALL, "MLB": Sport.BASEBALL, "NHL": Sport.HOCKEY}

st.subheader("1. Upload Files")
col1, col2 = st.columns(2)
with col1:
    dk_file = st.file_uploader("DK/FD Salaries CSV", type="csv", key="dk")
with col2:
    cheatsheet_file = st.file_uploader("Projections Cheatsheet CSV", type="csv", key="cheat")


# Conventional roster order per sport, so pills read the way a lineup does
# (team defense last for NFL). Anything not listed sorts alphabetically after.
POSITION_ORDER = {
    "NFL": ["QB", "RB", "WR", "TE", "K", "DST", "DEF", "D"],
    "NBA": ["PG", "SG", "SF", "PF", "C", "G", "F"],
    "NHL": ["C", "LW", "RW", "W", "D", "G"],
    "MLB": ["P", "SP", "RP", "C", "1B", "2B", "3B", "SS", "OF", "LF", "CF", "RF"],
}


def order_positions(positions, sport):
    preferred = POSITION_ORDER.get(sport, [])

    def key(position):
        if position in preferred:
            return (preferred.index(position), "")
        return (len(preferred), position)

    return sorted(positions, key=key)


class TeamCapUnlessQBRule(OptimizerRule):
    """Cap players per team, with a higher cap for the team whose QB is rostered.

    Built as a rule because no stack or restriction can express a conditional
    limit: the constraint needs a negative coefficient on the QB variable.

        sum(non-QB from team) - extra * sum(QB from team) <= cap_without_qb

    With cap_without_qb=2 and cap_with_qb=4 (so extra=1): no QB means at most 2
    from that team; rostering its QB allows 3 non-QB, i.e. 4 in total.
    """

    def apply(self, solver):
        cfg = getattr(self.optimizer, "team_cap_cfg", None)
        if not cfg:
            return
        cap, cap_with_qb = cfg
        extra = max((cap_with_qb - 1) - cap, 0)
        by_team = defaultdict(lambda: ([], []))
        for player, variable in self.players_dict.items():
            non_qb, qbs = by_team[player.team]
            (qbs if "QB" in player.positions else non_qb).append(variable)
        for non_qb, qbs in by_team.values():
            if len(non_qb) <= cap:
                continue          # can't breach the cap anyway
            if qbs and extra:
                solver.add_constraint(
                    non_qb + qbs,
                    [1] * len(non_qb) + [-extra] * len(qbs),
                    SolverSign.LTE, cap,
                )
            else:
                solver.add_constraint(non_qb, None, SolverSign.LTE, cap)


def build_qb_stack(pool, qb_side, bring_back, max_exposure=None):
    """A single Stack requiring: a QB, (qb_side-1) of his teammates, and
    `bring_back` players from his opponent.

    One stack rather than two means the QB can anchor the same-team part and the
    game part at once — separate stacks may not share a player, and with one QB
    slot that makes a QB-anchored stack plus a game stack impossible.
    """
    # filtered_players, not all_players: removed players are absent from the
    # solver's player map, and the library's exposure branch looks group members
    # up without guarding, so including them raises KeyError once a cap is set.
    by_team = defaultdict(list)
    for player in pool.filtered_players:
        by_team[player.team].append(player)
    groups = []
    for game in pool.games:
        for own, opp in ((game.home_team, game.away_team),
                         (game.away_team, game.home_team)):
            qbs = [p for p in by_team[own] if "QB" in p.positions]
            mates = [p for p in by_team[own] if "QB" not in p.positions]
            others = by_team[opp]
            if not qbs or len(mates) < qb_side - 1 or len(others) < bring_back:
                continue
            members = [PlayersGroup(qbs, min_from_group=1)]
            if qb_side > 1:
                members.append(PlayersGroup(mates, min_from_group=qb_side - 1))
            if bring_back:
                members.append(PlayersGroup(others, min_from_group=bring_back))
            groups.append(NestedPlayersGroup(groups=members, max_exposure=max_exposure))
    return Stack(groups=groups) if groups else None


NAME_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


def normalize_name(name):
    """Loose key for matching projection rows to salary rows.

    Drops punctuation, case and generational suffixes, so "T.J. Hockenson",
    "TJ Hockenson" and "Tj Hockenson Jr." all collapse to the same key.
    """
    cleaned = re.sub(r"[^a-z0-9\s]", " ", str(name).lower())
    parts = [w for w in cleaned.split() if w not in NAME_SUFFIXES]
    return " ".join(parts)


def load_optimizer(csv_text, solver=None):
    """Build a fresh optimizer from raw CSV text.

    Locks, removals and rules accumulate on an optimizer instance and are never
    cleared, so generation builds its own instance instead of reusing the one
    backing the player pool table.
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8") as tmp:
        tmp.write(csv_text)
        tmp_path = tmp.name
    try:
        optimizer = (get_optimizer(site_map[site], sport_map[sport], solver=solver)
                     if solver else get_optimizer(site_map[site], sport_map[sport]))
        optimizer.load_players_from_csv(tmp_path)
        return optimizer
    finally:
        os.unlink(tmp_path)


if dk_file is not None:
    content = dk_file.getvalue().decode("utf-8-sig")

    try:
        optimizer = load_optimizer(content)
        players = optimizer.player_pool.all_players
        roster_slots = optimizer.settings.get_total_players()
        slot_capacity = Counter()
        for _slot in optimizer.settings.positions:
            for _pos in _slot.positions:
                slot_capacity[_pos] += 1

        if is_showdown:
            # The captain-mode importer overwrites each player's position with
            # CPT/FLEX, so the real one has to come back from the salary file.
            # The pool also holds every player twice (once per slot) — the table
            # shows one row each, and rules apply to both copies.
            dk_meta = pd.read_csv(io.StringIO(content))
            real_by_id = dict(zip(dk_meta["ID"].astype(str),
                                  dk_meta["Position"].astype(str)))
            available_positions = order_positions(
                {p for p in real_by_id.values() if p}, sport)
            seen = set()
            rows = []
            for p in players:
                if "CPT" in p.positions:      # list each player once, on the FLEX row
                    continue
                if p.full_name in seen:
                    continue
                seen.add(p.full_name)
                rows.append({
                    "Name": p.full_name,
                    "Position": real_by_id.get(str(p.id), "?"),
                    "Team": p.team,
                    "Salary": p.salary,
                    "AvgPoints": p.fppg,
                    "_player_obj": p,
                })
            players_df = pd.DataFrame(rows)
        else:
            available_positions = order_positions(
                optimizer.player_pool.available_positions, sport)
            players_df = pd.DataFrame([{
                "Name": p.full_name,
                "Position": "/".join(p.positions),
                "Team": p.team,
                "Salary": p.salary,
                "AvgPoints": p.fppg,
                "_player_obj": p  # keep reference for lookups later
            } for p in players])

        # --- Merge in cheatsheet projections if provided ---
        EXTRA_COLS = ["injury_status", "ppg_projection", "value_projection",
                      "L5_fppg_avg", "L10_fppg_avg"]
        if cheatsheet_file is not None:
            cheat_df = pd.read_csv(cheatsheet_file)
            cheat_cols = list(cheat_df.columns)   # before Name/_key are added
            if {"first_name", "last_name"} <= set(cheat_df.columns):
                # fillna before astype(str): a blank last_name (common for team
                # defenses) would otherwise become the literal string "nan".
                cheat_df["Name"] = (cheat_df["first_name"].fillna("").astype(str).str.strip()
                                    + " "
                                    + cheat_df["last_name"].fillna("").astype(str).str.strip())
            elif "Name" not in cheat_df.columns:
                st.error(
                    "The projections file needs either **first_name** and **last_name** "
                    "columns or a single **Name** column. Found: "
                    + ", ".join(map(str, cheat_df.columns))
                )
                st.stop()
            cheat_df["_key"] = cheat_df["Name"].map(normalize_name)
            # 1:1 merge — a repeated name would otherwise multiply player rows.
            cheat_df = cheat_df.drop_duplicates(subset="_key")
            present = [c for c in EXTRA_COLS if c in cheat_df.columns]
            missing = [c for c in EXTRA_COLS if c not in cheat_df.columns]
            if missing:
                st.caption("Projection columns not in this file: " + ", ".join(missing))
            players_df["_key"] = players_df["Name"].map(normalize_name)
            merged_df = players_df.merge(cheat_df[["_key"] + present], on="_key", how="left")
            merged_df = merged_df.drop(columns=["_key"])
        else:
            merged_df = players_df.copy()

        merged_df = merged_df.drop(columns=["_player_obj"])

        # Selections live in session state so they survive reruns and can be
        # cleared. The editor's own state is an edit-diff keyed by row, so a
        # clear also needs a new widget key — hence the nonce.
        st.session_state.setdefault("locked_names", set())
        st.session_state.setdefault("excluded_names", set())
        st.session_state.setdefault("max_exp", {})
        st.session_state.setdefault("min_exp", {})
        st.session_state.setdefault("editor_nonce", 0)

        EDITABLE = ["Lock", "Exclude", "Min Exp %", "Max Exp %"]

        def apply_to_shown(names, field, on):
            """Bulk set Lock/Exclude for the rows the filters currently show."""
            key = "locked_names" if field == "lock" else "excluded_names"
            shown = set(names)
            st.session_state[key] = ((st.session_state[key] | shown) if on
                                     else (st.session_state[key] - shown))
            st.session_state.editor_nonce += 1

        def set_exp_on_shown(names, pct, field):
            """Write a Min/Max Exp % onto every row the filters currently show."""
            store = "max_exp" if field == "max" else "min_exp"
            updated = dict(st.session_state[store])
            for nm in names:
                if pct is None:
                    updated.pop(nm, None)
                else:
                    updated[nm] = float(pct)
            st.session_state[store] = updated
            st.session_state.editor_nonce += 1

        def clear_selections(lock=False, exclude=False, exposure=False):
            if lock:
                st.session_state.locked_names = set()
            if exclude:
                st.session_state.excluded_names = set()
            if exposure:
                st.session_state.max_exp = {}
                st.session_state.min_exp = {}
            st.session_state.editor_nonce += 1

        merged_df["Lock"] = merged_df["Name"].isin(st.session_state.locked_names)
        merged_df["Exclude"] = merged_df["Name"].isin(st.session_state.excluded_names)
        # Float64 (nullable) keeps "no limit" blank rather than showing 0.
        merged_df["Min Exp %"] = merged_df["Name"].map(st.session_state.min_exp).astype("Float64")
        merged_df["Max Exp %"] = merged_df["Name"].map(st.session_state.max_exp).astype("Float64")

        # --- Can the surviving pool still fill a roster? ---
        def pool_shortfalls(frame):
            """Roster slots that accept exactly one position and lack candidates."""
            have = Counter()
            for joined in frame["Position"]:
                for one in str(joined).split("/"):
                    have[one] += 1
            needed = Counter()
            if not is_showdown:
                # Showdown's slots are CPT/FLEX, which any player can fill — they
                # aren't real positions, so there is nothing per-position to check.
                for slot in optimizer.settings.positions:
                    if len(slot.positions) == 1:
                        needed[slot.positions[0]] += 1
            return [(pos, needed[pos], have.get(pos, 0))
                    for pos in sorted(needed) if have.get(pos, 0) < needed[pos]], have

        # --- Which numbers the optimizer maximises ---
        # A projections file means: optimize on ppg_projection and drop anyone it
        # doesn't cover, so nothing unprojected can reach a lineup.
        proj_map = {}
        unprojected_names = []
        if cheatsheet_file is not None:
            if "ppg_projection" not in merged_df.columns:
                st.error(
                    "The projections file has no **ppg_projection** column. Found: "
                    + ", ".join(str(c) for c in cheat_cols)
                )
                st.stop()
            # Three ways out of the pool: no projection at all, ruled out (O), or
            # projected at or below the floor. A player can hit more than one;
            # label by priority.
            MIN_PROJECTION = 0.6
            no_proj = merged_df["ppg_projection"].isna()
            if "injury_status" in merged_df.columns:
                is_out = (merged_df["injury_status"].astype(str).str.upper().str.strip()
                          .eq("O"))
            else:
                is_out = pd.Series(False, index=merged_df.index)
            zero_proj = (merged_df["ppg_projection"].fillna(0).le(MIN_PROJECTION)
                         & ~no_proj)

            reason = pd.Series("", index=merged_df.index, dtype="object")
            reason[zero_proj] = f"projected <= {MIN_PROJECTION:g}"
            reason[is_out] = "out (O)"
            reason[no_proj] = "no projection"
            drop_mask = no_proj | is_out | zero_proj

            dropped = merged_df.loc[drop_mask, ["Name", "Position", "Team"]].copy()
            dropped["Reason"] = reason[drop_mask]
            unprojected_names = dropped["Name"].tolist()

            keep = ~drop_mask
            proj_map = dict(zip(merged_df.loc[keep, "Name"],
                                merged_df.loc[keep, "ppg_projection"]))
            if not proj_map:
                st.error(
                    "Every player was dropped — no projection, ruled out, or projected 0. "
                    "Check that the names line up with the salary file."
                )
                st.stop()
            merged_df = merged_df[keep].reset_index(drop=True)

            counts = reason[drop_mask].value_counts().to_dict()
            detail = ", ".join(f"{v} {k}" for k, v in counts.items()) or "none"
            st.caption(
                f"Optimizing on **ppg_projection** — {len(proj_map)} players. "
                f"{len(unprojected_names)} removed ({detail})."
            )
            if unprojected_names:
                with st.expander(f"{len(unprojected_names)} players removed from the pool"):
                    st.dataframe(dropped.sort_values(["Reason", "Position", "Name"]),
                                 width="stretch", hide_index=True)

            short, have = pool_shortfalls(merged_df)
            if short or len(merged_df) < roster_slots:
                lines = [f"- **{pos}**: roster needs {need}, projections cover {got}"
                         for pos, need, got in short]
                if len(merged_df) < roster_slots:
                    lines.append(f"- only {len(merged_df)} players left, roster needs "
                                 f"{roster_slots}")
                st.error(
                    "**The projections file doesn't cover enough of the slate to build a "
                    "lineup.**\n\n" + "\n".join(lines) +
                    "\n\nPlayers without a projection are removed, so a sheet that skips a "
                    "position leaves nothing to fill it. Add those players to the file."
                )
                st.stop()

            qb_teams = set(merged_df.loc[merged_df["Position"].str.contains("QB"), "Team"])
            no_qb_teams = sorted(set(merged_df["Team"]) - qb_teams)
            if no_qb_teams:
                st.warning(
                    f"{len(no_qb_teams)} team(s) have no projected QB "
                    f"({', '.join(no_qb_teams[:8])}{'…' if len(no_qb_teams) > 8 else ''}). "
                    "Those teams can never exceed the \"without its QB\" limit in section 4, "
                    "so game stacks may not be able to use them."
                )

        st.subheader("2. Player Pool — set Lock / Exclude / Exposure per player")

        pool_teams = sorted(merged_df["Team"].dropna().unique())
        fcol1, fcol2 = st.columns([2, 3])
        with fcol1:
            filter_positions = st.pills(
                "Filter by position", available_positions, selection_mode="multi",
                key="pos_filter", help="Narrows the table only — it does not remove "
                                       "anyone from the optimizer.",
            )
        with fcol2:
            filter_teams = st.multiselect(
                "Filter by team", pool_teams, key="team_filter",
                placeholder="All teams",
                help="Narrows the table only. Combine with the position filter and use "
                     "the buttons below to act on just these players.",
            )

        visible_mask = pd.Series(True, index=merged_df.index)
        if filter_positions:
            wanted = set(filter_positions)
            visible_mask &= merged_df["Position"].apply(
                lambda joined: bool(set(joined.split("/")) & wanted))
        if filter_teams:
            visible_mask &= merged_df["Team"].isin(filter_teams)
        filtered = bool(filter_positions or filter_teams)
        view_df = (merged_df[visible_mask].reset_index(drop=True) if filtered
                   else merged_df)
        shown_names = view_df["Name"].tolist()

        scope = "shown" if filtered else "all"
        with st.container(horizontal=True):
            st.button(f"🚫 Exclude {scope} ({len(shown_names)})",
                      on_click=apply_to_shown,
                      kwargs={"names": shown_names, "field": "exclude", "on": True},
                      help="Excludes every player the filters currently show.")
            st.button(f"Un-exclude {scope}",
                      on_click=apply_to_shown,
                      kwargs={"names": shown_names, "field": "exclude", "on": False})
            st.button(f"🔒 Lock {scope}",
                      on_click=apply_to_shown,
                      kwargs={"names": shown_names, "field": "lock", "on": True})
            st.button(f"Unlock {scope}",
                      on_click=apply_to_shown,
                      kwargs={"names": shown_names, "field": "lock", "on": False})

        with st.container(horizontal=True, vertical_alignment="bottom"):
            bulk_pct = st.number_input("Exposure %", min_value=0, max_value=100, value=40,
                                       step=5, key="bulk_exp_pct")
            st.button(f"🎯 Set Max Exp on {scope}",
                      on_click=set_exp_on_shown,
                      kwargs={"names": shown_names, "pct": bulk_pct, "field": "max"},
                      help="Writes this Max Exp % onto every player the filters show.")
            st.button(f"📌 Set Min Exp on {scope}",
                      on_click=set_exp_on_shown,
                      kwargs={"names": shown_names, "pct": bulk_pct, "field": "min"})
            st.button(f"Clear exp on {scope}",
                      on_click=set_exp_on_shown,
                      kwargs={"names": shown_names, "pct": None, "field": "max"})

        with st.container(horizontal=True):
            st.button("Clear locks", icon=":material/lock_open:",
                      on_click=clear_selections, kwargs={"lock": True})
            st.button("Clear excludes", icon=":material/block:",
                      on_click=clear_selections, kwargs={"exclude": True})
            st.button("Clear exposures", icon=":material/percent:",
                      on_click=clear_selections, kwargs={"exposure": True})
            st.button("Clear all", icon=":material/restart_alt:",
                      on_click=clear_selections,
                      kwargs={"lock": True, "exclude": True, "exposure": True})

        edited_df = st.data_editor(
            view_df,
            width="stretch",
            hide_index=True,
            column_config={
                "Lock": st.column_config.CheckboxColumn("🔒 Lock"),
                "Exclude": st.column_config.CheckboxColumn("🚫 Exclude"),
                "Min Exp %": st.column_config.NumberColumn(
                    "📌 Min Exp %", min_value=0, max_value=100, step=5,
                    help="Forces the player into at least this share of lineups.",
                ),
                "Max Exp %": st.column_config.NumberColumn(
                    "🎯 Max Exp %", min_value=0, max_value=100, step=5,
                    help="Blank = use the global cap. 0 removes the player entirely.",
                ),
            },
            disabled=[c for c in view_df.columns if c not in EDITABLE],
            # The editor's diff is keyed by row position, so a changed filter must
            # get a new widget — otherwise stale edits land on different players.
            key=f"player_editor_{st.session_state.editor_nonce}"
               f"_{'-'.join(sorted(filter_positions)) or 'all'}"
               f"_{'-'.join(sorted(filter_teams)) or 'all'}"
        )

        # Only the visible rows were editable, so merge them over what is stored
        # rather than replacing it — filtered-out players keep their settings.
        shown = set(edited_df["Name"])
        shown_locked = set(edited_df.loc[edited_df["Lock"], "Name"])
        shown_excluded = set(edited_df.loc[edited_df["Exclude"], "Name"])
        shown_max = {n: float(v) for n, v in zip(edited_df["Name"], edited_df["Max Exp %"])
                     if pd.notna(v)}
        shown_min = {n: float(v) for n, v in zip(edited_df["Name"], edited_df["Min Exp %"])
                     if pd.notna(v)}

        st.session_state.locked_names = (st.session_state.locked_names - shown) | shown_locked
        st.session_state.excluded_names = (st.session_state.excluded_names - shown) | shown_excluded
        st.session_state.max_exp = {
            **{n: v for n, v in st.session_state.max_exp.items() if n not in shown}, **shown_max}
        st.session_state.min_exp = {
            **{n: v for n, v in st.session_state.min_exp.items() if n not in shown}, **shown_min}

        # Generation uses the full stored selection, not just what is on screen.
        locked_names = sorted(st.session_state.locked_names)
        excluded_names = sorted(st.session_state.excluded_names)
        max_exp = st.session_state.max_exp
        min_exp = st.session_state.min_exp

        if filtered:
            bits = []
            if filter_positions:
                bits.append("/".join(sorted(filter_positions)))
            if filter_teams:
                bits.append(", ".join(sorted(filter_teams)))
            st.caption(f"Showing {len(view_df)} of {len(merged_df)} players "
                       f"({' · '.join(bits)}).")
        st.caption(
            f"Locked: {len(locked_names)} · Excluded: {len(excluded_names)} · "
            f"Min exp set: {len(min_exp)} · Max exp set: {len(max_exp)} "
            f"(counts cover the whole pool, not just the filtered rows)"
        )

        for name in sorted(set(max_exp) & set(min_exp)):
            if min_exp[name] > max_exp[name]:
                st.warning(
                    f"{name}: min exposure {min_exp[name]:.0f}% is above max "
                    f"{max_exp[name]:.0f}% — no lineup can satisfy both."
                )

        if is_showdown:
            st.markdown("**Showdown**")
            st.caption(
                "One game, so the opposing-team, same-team and stacking rules don't "
                "apply — every player already faces every other, and the roster is "
                "CPT + 5 FLEX from two teams. Use the player pool above for locks, "
                "excludes and exposure, and the sidebar for lineup count, global "
                "exposure, min-unique and randomness."
            )
            restrict_opposing = False
            first_team_positions, second_team_positions, max_allowed = [], [], 0
            same_team_pairs = []
            cap_no_qb = cap_with_qb = roster_slots
            qb_stack_specs, flex_rows, mix_rows = [], [], []
            mix_mode = False
            batch_plan = [{"specs": [], "flex": None, "weight": 1.0,
                           "label": "showdown", "count": num_lineups}]
            batch_counts = [num_lineups]
        else:
            st.subheader("3. Opposing Team Position Restriction")
            has_game_info = bool(optimizer.player_pool.games)

            restrict_opposing = False
            first_team_positions = []
            second_team_positions = []
            max_allowed = 0

            if not has_game_info:
                st.warning(
                    "No game info found in this salary file, so opposing-team rules can't be applied. "
                    "Re-export the salary CSV with the **Game Info** column included "
                    "(e.g. `NYJ@DET 09/27/2026 01:00PM ET`)."
                )
            else:
                restrict_opposing = st.toggle("Restrict positions for opposing team")
                if restrict_opposing:
                    first_team_positions = st.pills(
                        "Positions on one team",
                        available_positions,
                        selection_mode="multi",
                        key="opp_first",
                    )
                    second_team_positions = st.pills(
                        "Positions to limit on the opposing team",
                        available_positions,
                        selection_mode="multi",
                        key="opp_second",
                    )
                    max_allowed = st.number_input(
                        "Max allowed together", min_value=0, max_value=10, value=0, key="opp_max"
                    )
                    st.caption(
                        "When a lineup uses any of the first positions, at most this many players from "
                        "the second group on that player's opposing team may join them. `0` blocks the "
                        "pairing entirely. The rule applies in both directions, and the optimizer "
                        "supports only one pair of groups at a time."
                    )

            st.subheader("4. Same-Team Restrictions")
            st.caption(
                "Each row blocks that pair of positions from appearing together on the *same* team. "
                "Repeat a position to stop doubling up (RB + RB blocks two RBs from one team). "
                "Add or delete rows freely; leave the table empty for no same-team rules."
            )
            default_pairs = [p for p in [("RB", "RB"), ("QB", "DST")]
                             if p[0] in available_positions and p[1] in available_positions]
            # Mirrored into session state like the player pool: a data_editor's own
            # state is a row-keyed diff, which is the wrong thing to read back from
            # on a later rerun (e.g. after the Generate click).
            st.session_state.setdefault("same_team_pairs", default_pairs)
            st.session_state.setdefault("same_team_nonce", 0)

            pairs_df = pd.DataFrame(st.session_state.same_team_pairs,
                                    columns=["Position A", "Position B"])
            edited_pairs = st.data_editor(
                pairs_df,
                num_rows="dynamic",
                width="content",
                hide_index=True,
                column_config={
                    "Position A": st.column_config.SelectboxColumn(
                        "Position A", options=available_positions, required=True),
                    "Position B": st.column_config.SelectboxColumn(
                        "Position B", options=available_positions, required=True),
                },
                key=f"same_team_editor_{st.session_state.same_team_nonce}",
            )

            # Dedupe on the unordered pair — the rule is symmetric, so QB/DST and
            # DST/QB would build the same constraints twice.
            collected_pairs = []
            for a, b in zip(edited_pairs["Position A"], edited_pairs["Position B"]):
                if pd.isna(a) or pd.isna(b):
                    continue
                if a not in available_positions or b not in available_positions:
                    continue
                if tuple(sorted((a, b))) not in [tuple(sorted(p)) for p in collected_pairs]:
                    collected_pairs.append((str(a), str(b)))

            if collected_pairs != st.session_state.same_team_pairs:
                # Store the new list and rebuild the editor from it, so the stale
                # row diff can't be replayed on top of the updated rows.
                st.session_state.same_team_pairs = collected_pairs
                st.session_state.same_team_nonce += 1
                st.rerun()

            same_team_pairs = st.session_state.same_team_pairs

            if same_team_pairs:
                st.caption("Active: " + " · ".join(f"no {a} + {b} same team" for a, b in same_team_pairs))

            st.markdown("**Players per team**")
            tc1, tc2 = st.columns(2)
            with tc1:
                cap_no_qb = st.number_input(
                    "Max from a team WITHOUT its QB", min_value=1, max_value=roster_slots,
                    value=2, key="cap_no_qb",
                    help="Team defenses count. At 2, no team can reach 3 players unless its "
                         "QB is in the lineup.",
                )
            with tc2:
                cap_with_qb = st.number_input(
                    "Max from the QB's own team", min_value=1, max_value=roster_slots,
                    value=4, key="cap_with_qb",
                    help="Counts the QB himself, so 4 means QB + 3 teammates.",
                )
            if cap_with_qb < cap_no_qb:
                st.warning("The QB's team allowance is below the no-QB one, so the QB gives "
                           "no extra room.")
            st.caption(
                f"Up to **{cap_no_qb}** players from any team whose QB you don't roster, and up "
                f"to **{cap_with_qb}** from the team whose QB you do (QB + {cap_with_qb - 1} "
                f"teammates). DST counts as one of them."
            )

            st.subheader("5. QB Stacks")
            st.caption(
                "One row per stack shape. **QB side** is how many players come from the QB's own "
                "team *including him*, so 4 means QB + 3 teammates. **Bring-back** is how many "
                "come from his opponent. The QB is required by construction, so the shape is "
                "stated rather than inferred.\n\n"
                "**Share %** splits the run between rows: leave it blank on every row and all "
                "rows apply to every lineup at once (their players can't overlap, so the totals "
                "must fit one roster). Set it and each row gets its own slice of the lineups — "
                "the app runs the optimizer once per row and combines the results, which is "
                "app-side batching rather than a pydfs feature."
            )
            flex_options = ["Any"] + sorted(
                {pos for slot in optimizer.settings.positions if len(slot.positions) > 1
                 for pos in slot.positions},
                key=lambda x: available_positions.index(x) if x in available_positions else 99)

            qs_rows = []
            if not has_game_info:
                st.warning("QB stacks need the **Game Info** column — see section 3.")
            else:
                st.session_state.setdefault("qs_rows", [])
                st.session_state.setdefault("qs_nonce", 0)
                qs_df = pd.DataFrame(
                    [{"QB side": r["qb_side"], "Bring-back": r["bring_back"],
                      "Share %": r.get("share"), "Max Exp %": r["max_exp"]}
                     for r in st.session_state.qs_rows],
                    columns=["QB side", "Bring-back", "Share %", "Max Exp %"])
                qs_df["QB side"] = qs_df["QB side"].astype("Int64")
                qs_df["Bring-back"] = qs_df["Bring-back"].astype("Int64")
                qs_df["Share %"] = qs_df["Share %"].astype("Float64")
                qs_df["Max Exp %"] = qs_df["Max Exp %"].astype("Float64")
                edited_qs = st.data_editor(
                    qs_df,
                    num_rows="dynamic",
                    width="content",
                    hide_index=True,
                    column_config={
                        "QB side": st.column_config.NumberColumn(
                            "QB side", min_value=1, max_value=roster_slots, step=1, required=True,
                            help="Players from the QB's team, counting the QB."),
                        "Bring-back": st.column_config.NumberColumn(
                            "Bring-back", min_value=0, max_value=roster_slots, step=1,
                            required=True, help="Players from the opposing team."),
                        "Share %": st.column_config.NumberColumn(
                            "Share %", min_value=1, max_value=100, step=5,
                            help="Percent of lineups built on this row. Set it on every row "
                                 "(summing to 100) to split the run between shapes."),
                        "Max Exp %": st.column_config.NumberColumn(
                            "Max Exp %", min_value=1, max_value=100, step=5,
                            help="Blank = no cap. Caps how often any single game fills this "
                                 "shape — it does not reduce how often the row applies."),
                    },
                    key=f"qs_editor_{st.session_state.qs_nonce}",
                )
                collected_qs = []
                for _, row in edited_qs.iterrows():
                    if pd.isna(row["QB side"]):
                        continue
                    collected_qs.append({
                        "qb_side": int(row["QB side"]),
                        "bring_back": int(row["Bring-back"]) if pd.notna(row["Bring-back"]) else 0,
                        "share": float(row["Share %"]) if pd.notna(row["Share %"]) else None,
                        "max_exp": float(row["Max Exp %"]) if pd.notna(row["Max Exp %"]) else None,
                    })
                if collected_qs != st.session_state.qs_rows:
                    st.session_state.qs_rows = collected_qs
                    st.session_state.qs_nonce += 1
                    st.rerun()
                qs_rows = st.session_state.qs_rows

            # Validate each row against the per-team limits from section 4.
            qb_stack_specs = []
            for idx, row in enumerate(qs_rows, start=1):
                total = row["qb_side"] + row["bring_back"]
                if row["qb_side"] > cap_with_qb:
                    st.error(f"Row {idx}: QB side {row['qb_side']} exceeds the "
                             f"{cap_with_qb}-player limit for the QB's own team (section 4).")
                    continue
                if row["bring_back"] > cap_no_qb:
                    st.error(f"Row {idx}: bring-back {row['bring_back']} exceeds the "
                             f"{cap_no_qb}-player limit for a team without its QB (section 4).")
                    continue
                if total > roster_slots:
                    st.error(f"Row {idx}: {total} players needed but the roster holds "
                             f"{roster_slots}.")
                    continue
                qb_stack_specs.append(row)
                cap = f", max {row['max_exp']:.0f}%/game" if row["max_exp"] else ", no cap"
                st.caption(f"Stack {idx}: QB + {row['qb_side'] - 1} teammate"
                           f"{'s' if row['qb_side'] - 1 != 1 else ''} + {row['bring_back']} "
                           f"back = {total} from one game{cap}")

            st.markdown("**Flex mix**")
            st.caption(
                "Share of lineups whose flex slot holds each position. Leave the table empty "
                "to let the optimizer choose — which tends to produce no WRs, because the "
                "good ones are already in the three dedicated WR slots."
            )
            st.session_state.setdefault("flex_rows", [])
            st.session_state.setdefault("flex_nonce", 0)
            flex_df = pd.DataFrame(
                [{"Position": r["position"], "Share %": r["share"]}
                 for r in st.session_state.flex_rows],
                columns=["Position", "Share %"])
            flex_df["Share %"] = flex_df["Share %"].astype("Float64")
            edited_flex = st.data_editor(
                flex_df,
                num_rows="dynamic",
                width="content",
                hide_index=True,
                column_config={
                    "Position": st.column_config.SelectboxColumn(
                        "Position", options=[p for p in flex_options if p != "Any"],
                        required=True),
                    "Share %": st.column_config.NumberColumn(
                        "Share %", min_value=1, max_value=100, step=5, required=True),
                },
                key=f"flex_editor_{st.session_state.flex_nonce}",
            )
            collected_flex = []
            for _, row in edited_flex.iterrows():
                if pd.isna(row["Position"]) or pd.isna(row["Share %"]):
                    continue
                if row["Position"] in {r["position"] for r in collected_flex}:
                    continue
                collected_flex.append({"position": str(row["Position"]),
                                       "share": float(row["Share %"])})
            if collected_flex != st.session_state.flex_rows:
                st.session_state.flex_rows = collected_flex
                st.session_state.flex_nonce += 1
                st.rerun()
            flex_rows = st.session_state.flex_rows
            if flex_rows:
                ftot = sum(r["share"] for r in flex_rows)
                if abs(ftot - 100) > 0.5:
                    st.warning(f"Flex shares add up to {ftot:.0f}%, not 100% — "
                               "they'll be scaled proportionally.")
                st.caption("Flex: " + " · ".join(f"{r['position']} {r['share']:.0f}%"
                                                 for r in flex_rows))

            mix_rows = [r for r in qb_stack_specs if r.get("share")]
            if mix_rows and len(mix_rows) < len(qb_stack_specs):
                st.warning(
                    f"{len(qb_stack_specs) - len(mix_rows)} stack row(s) have no Share % and "
                    "are ignored while splitting. Give every row a share, or clear them all."
                )
            if mix_rows:
                qtot = sum(r["share"] for r in mix_rows)
                if abs(qtot - 100) > 0.5:
                    st.warning(f"Stack shares add up to {qtot:.0f}%, not 100% — "
                               "they'll be scaled proportionally.")

            # Batches are the cross product of the two mixes. Either can be empty:
            # with neither, this collapses to a single run of everything.
            qb_units = [(r, r["share"]) for r in mix_rows] or [(None, 100.0)]
            flex_units = [(r["position"], r["share"]) for r in flex_rows] or [(None, 100.0)]
            qsum = sum(w for _, w in qb_units)
            fsum = sum(w for _, w in flex_units)

            batch_plan = []
            for spec, qw in qb_units:
                for pos, fw in flex_units:
                    batch_plan.append({
                        "specs": [spec] if spec else qb_stack_specs,
                        "flex": pos,
                        "weight": (qw / qsum) * (fw / fsum),
                        "label": " / ".join(filter(None, [
                            (f"QB + {spec['qb_side'] - 1} + {spec['bring_back']} back"
                             if spec else None),
                            (f"flex {pos}" if pos else None),
                        ])) or "all rules together",
                    })

            mix_mode = len(batch_plan) > 1
            raw = [b["weight"] * num_lineups for b in batch_plan]
            batch_counts = [int(x) for x in raw]
            for i in sorted(range(len(raw)), key=lambda j: raw[j] - batch_counts[j],
                            reverse=True)[:num_lineups - sum(batch_counts)]:
                batch_counts[i] += 1
            for b, c in zip(batch_plan, batch_counts):
                b["count"] = c

            if mix_mode:
                for b in batch_plan:
                    st.caption(f"→ {b['label']}: {b['weight']:.0%} → {b['count']} lineup"
                               f"{'s' if b['count'] != 1 else ''}")
                if any(b["count"] < 1 for b in batch_plan):
                    smallest = min(b["weight"] for b in batch_plan)
                    needed = int(-(-1 // smallest)) if smallest else len(batch_plan)
                    st.warning(
                        f"**Number of lineups is {num_lineups}**, too few to split across "
                        f"{len(batch_plan)} combinations — "
                        f"{sum(1 for b in batch_plan if b['count'] < 1)} get 0 lineups. "
                        f"Raise it to at least {max(needed, len(batch_plan))}."
                    )
            elif len(qb_stack_specs) > 1:
                needed = sum(r["qb_side"] + r["bring_back"] for r in qb_stack_specs)
                if needed > roster_slots:
                    st.error(
                        f"These rows apply together and need {needed} separate players, but the "
                        f"roster holds {roster_slots}. Stacks can't share players — give each row "
                        "a **Share %** so they run as separate batches instead."
                    )
                else:
                    st.caption(f"Using {needed} of {roster_slots} roster spots — these rows apply "
                               "together, each on a different game.")

        st.subheader(("4." if is_showdown else "6.") + " Generate Lineups")
        if st.button("Generate Lineups"):

            def configure_optimizer(report=True, solver=None):
                """A fresh optimizer with every section's settings applied."""
                opt = load_optimizer(content, solver=solver)
                pool = opt.player_pool
                # Showdown holds each player twice (CPT and FLEX), so a name maps
                # to a list. Settings apply to every copy.
                by_name = defaultdict(list)
                for p in pool.all_players:
                    by_name[p.full_name].append(p)

                # Replace the salary file's points with the uploaded projection.
                # fppg is what StandardFantasyPointsStrategy maximises.
                if proj_map:
                    for nm, val in proj_map.items():
                        for pl in by_name.get(nm, ()):
                            # The importer already scaled CPT points by 1.5;
                            # overwriting fppg has to reapply it or captains
                            # become worthless at 1.5x the salary.
                            mult = 1.5 if "CPT" in pl.positions else 1.0
                            pl.fppg = float(val) * mult
                    for nm in unprojected_names:
                        for pl in by_name.get(nm, ()):
                            pool.remove_player(pl)

                # Exposure is a property of the Player, so set it before locking —
                # lock_player rejects anyone capped at 0.
                for nm, pct in max_exp.items():
                    for pl in by_name.get(nm, ()):
                        pl.max_exposure = pct / 100
                for nm, pct in min_exp.items():
                    for pl in by_name.get(nm, ()):
                        pl.min_exposure = pct / 100

                for nm in excluded_names:
                    for pl in by_name.get(nm, ()):
                        pool.remove_player(pl)

                # lock_player enforces budget, roster size and max-from-one-team,
                # raising rather than silently dropping. In showdown only one copy
                # can be locked — locking both would eat two roster spots.
                for nm in locked_names:
                    if nm in excluded_names:
                        continue
                    copies = by_name.get(nm, ())
                    if not copies:
                        continue
                    target = next((c for c in copies if "CPT" not in c.positions), copies[0])
                    try:
                        pool.lock_player(target)
                    except LineupOptimizerException as exc:
                        if report:
                            st.warning(f"Could not lock {nm}: {exc}")

                if restrict_opposing:
                    if first_team_positions and second_team_positions:
                        opt.restrict_positions_for_opposing_team(
                            first_team_positions, second_team_positions, max_allowed)
                    elif report:
                        st.warning("Opposing-team restriction skipped — pick at least one "
                                   "position on each side.")

                if min_unique:
                    # max_repeating is expressed from the other side: roster size
                    # minus the unique players you want between any two lineups.
                    repeats = max(roster_slots - int(min_unique), 1)
                    try:
                        opt.set_max_repeating_players(repeats)
                    except LineupOptimizerException as exc:
                        if report:
                            st.warning(f"Min unique players ignored: {exc}")

                if deviation:
                    # set_fantasy_points_strategy rather than optimize(randomness=True):
                    # that parameter is deprecated in this version.
                    opt.set_fantasy_points_strategy(
                        RandomFantasyPointsStrategy(0.0, deviation))

                if same_team_pairs:
                    opt.restrict_positions_for_same_team(*same_team_pairs)

                # Conditional per-team cap: more players allowed from the team whose
                # QB is rostered. DST counts as a non-QB player.
                opt.team_cap_cfg = (int(cap_no_qb), int(cap_with_qb))
                opt.add_new_rule(TeamCapUnlessQBRule)

                return opt

            def add_qb_stack(opt, spec, report=True):
                stack = build_qb_stack(
                    opt.player_pool, spec["qb_side"], spec["bring_back"],
                    max_exposure=spec["max_exp"] / 100 if spec["max_exp"] else None,
                )
                if stack is None:
                    if report:
                        st.warning("No game on this slate can supply that shape.")
                    return
                try:
                    opt.add_stack(stack)
                except LineupOptimizerException as exc:
                    if report:
                        st.warning(f"QB stack skipped: {exc}")


            def make_optimizer(specs, solver, report, flex=None):
                opt = configure_optimizer(report=report, solver=solver)
                if flex:
                    # An extra required player at a position pushes one into the
                    # flex slot: the roster needs 3 WRs, so asking for 4 puts a
                    # WR there.
                    opt.set_players_with_same_position({flex: 1})
                for spec in specs:
                    add_qb_stack(opt, spec, report=report)
                return opt

            def run_batch(specs, count, report=True, exclude=(), flex=None):
                """Solve on the fast engine, fall back to the library default.

                The fallback exists for the error message: PuLP names the
                constraints it couldn't satisfy, which the diagnostics rely on.
                `exclude` carries lineups already built in earlier batches —
                each batch is its own optimizer run, so without it the same
                roster can come back twice.
                """
                kwargs = dict(optimize_kwargs)
                if exclude:
                    kwargs["exclude_lineups"] = list(exclude)
                if FAST_SOLVER is not None:
                    try:
                        return list(make_optimizer(specs, FAST_SOLVER, report, flex)
                                    .optimize(count, **kwargs))
                    except LineupOptimizerException:
                        pass
                return list(make_optimizer(specs, None, report, flex)
                            .optimize(count, **kwargs))

            optimize_kwargs = {"exposure_strategy": strategy_map[strategy_label]}
            if global_max_exp < 100:
                optimize_kwargs["max_exposure"] = global_max_exp / 100

            all_lineups = []
            lineups = []
            mix_report = []
            def show_diagnostics():
                have = Counter()
                for joined in merged_df["Position"]:
                    for one in str(joined).split("/"):
                        have[one] += 1
                info = {
                    "Lineups requested": num_lineups,
                    "Pool size": len(merged_df),
                    "Pool by position": dict(sorted(have.items())),
                    "Projections file": "yes" if cheatsheet_file is not None else "no",
                    "Removed (no projection)": len(unprojected_names),
                    "Locked": f"{len(locked_names)} {locked_names[:6]}",
                    "Excluded": f"{len(excluded_names)} {excluded_names[:6]}",
                    "Min exp set": f"{len(min_exp)} {list(min_exp.items())[:4]}",
                    "Max exp set": f"{len(max_exp)} {list(max_exp.items())[:4]}",
                    "Global max exposure %": global_max_exp,
                    "Min unique players": min_unique,
                    "Contest / randomness": f"{contest_mode} ({deviation:.0%})",
                    "Exposure strategy": strategy_label,
                    "Max from team without QB": cap_no_qb,
                    "Max from QB's team": cap_with_qb,
                    "Same-team pairs": same_team_pairs,
                    "Opposing restriction": (
                        f"{first_team_positions} vs {second_team_positions}, "
                        f"max {max_allowed}" if restrict_opposing else "off"),
                    "QB stacks": [(r["qb_side"], r["bring_back"], r.get("share"),
                                   r["max_exp"]) for r in qb_stack_specs],
                    "Flex mix": [(r["position"], r["share"]) for r in flex_rows],
                }
                with st.expander("Diagnostics — the exact settings in play", expanded=True):
                    st.code("\n".join(f"{k}: {v}" for k, v in info.items()), language="text")

            trouble = (
                "The rules above contradict each other. Most common causes:\n"
                "- Same-team restrictions that leave no valid roster\n"
                "- A QB stack shape larger than the per-team limits in section 4 allow\n"
                "- Min exposure totals that exceed the roster, or too few players left "
                "after excludes\n\n"
                "The reported constraint names come from the solver and often point at "
                "knock-on effects rather than the rule you actually set."
            )

            if mix_mode:
                # One optimizer run per (stack shape x flex position) combination.
                # This batching is the app's, not the library's.
                for idx, b in enumerate(batch_plan):
                    if b["count"] < 1:
                        mix_report.append({"Rule": b["label"], "Share %":
                                           round(b["weight"] * 100, 1), "Lineups": 0})
                        continue
                    try:
                        batch = run_batch(b["specs"], b["count"], report=(idx == 0),
                                          exclude=lineups, flex=b["flex"])
                    except LineupOptimizerException as exc:
                        st.error(f"**\"{b['label']}\" failed:** {exc}\n\n" + trouble)
                        show_diagnostics()
                        st.stop()
                    lineups.extend(batch)
                    mix_report.append({"Rule": b["label"], "Share %":
                                       round(b["weight"] * 100, 1), "Lineups": len(batch)})
            else:
                try:
                    lineups = run_batch(qb_stack_specs, num_lineups)
                except LineupOptimizerException as exc:
                    st.error(f"**Couldn't generate lineups:** {exc}\n\n" + trouble)
                    show_diagnostics()
                    st.stop()

            for i, lineup in enumerate(lineups, start=1):
                for player in lineup.players:
                    all_lineups.append({
                        "Lineup": i,
                        "Position": player.lineup_position,
                        "Name": player.full_name,
                        "ID": player.id,
                        "Team": player.team,
                        "Salary": player.salary,
                        "AvgPoints": player.fppg
                    })
                all_lineups.append({
                    "Lineup": i, "Position": "TOTAL", "Name": "",
                    "Team": "", "Salary": lineup.salary_costs,
                    "AvgPoints": round(lineup.fantasy_points_projection, 2)
                })

            # Stash the results: rendering happens outside the button block so a
            # filter click doesn't make the whole section disappear.
            result_df = pd.DataFrame(all_lineups)
            picks = result_df[result_df["Position"] != "TOTAL"]
            total = int(result_df["Lineup"].max())
            exposure_df = (
                picks.groupby(["Name", "Team"]).size().reset_index(name="Lineups")
            )
            exposure_df["Exposure %"] = (exposure_df["Lineups"] / total * 100).round(1)
            # real position, not the roster slot the player filled (FLEX etc.)
            real_pos = dict(zip(merged_df["Name"], merged_df["Position"]))
            exposure_df.insert(1, "Position", exposure_df["Name"].map(real_pos))
            exposure_df["Min set %"] = exposure_df["Name"].map(min_exp)
            exposure_df["Max set %"] = exposure_df["Name"].map(max_exp)
            exposure_df = exposure_df.sort_values("Lineups", ascending=False)

            # DK upload format: one row per lineup, one column per roster slot,
            # each cell the salary file's own "Name + ID" string.
            slot_names = [pos.name for pos in optimizer.settings.positions]
            dk_raw = pd.read_csv(io.StringIO(content))
            if "Name + ID" in dk_raw.columns:
                id_to_cell = dict(zip(dk_raw["ID"].astype(str),
                                      dk_raw["Name + ID"].astype(str)))
            else:
                id_to_cell = {str(i): f"{n} ({i})"
                              for i, n in zip(dk_raw["ID"], dk_raw["Name"])}
            upload_rows = []
            for _lid, grp in picks.groupby("Lineup", sort=True):
                cells = [id_to_cell.get(str(pid), str(nm))
                         for pid, nm in zip(grp["ID"], grp["Name"])]
                if len(cells) == len(slot_names):
                    upload_rows.append(cells)
            upload_df = (pd.DataFrame(upload_rows, columns=slot_names)
                         if upload_rows else None)

            # Team-level exposure: how often each team appears, and how often it
            # shows up as a genuine stack rather than a one-off.
            per_team = picks.groupby(["Team", "Lineup"]).size().rename("spots").reset_index()
            team_df = (per_team.groupby("Team")
                       .agg(Lineups=("Lineup", "nunique"), Spots=("spots", "sum"))
                       .reset_index())
            team_df["Exposure %"] = (team_df["Lineups"] / total * 100).round(1)
            team_df["Spots/lineup"] = (team_df["Spots"] / team_df["Lineups"]).round(2)
            stacked = (per_team[per_team["spots"] >= 3].groupby("Team")["Lineup"]
                       .nunique().rename("3+ stack"))
            team_df = team_df.merge(stacked, on="Team", how="left")
            team_df["3+ stack"] = team_df["3+ stack"].fillna(0).astype(int)
            team_df = team_df.sort_values("Lineups", ascending=False)

            # What actually fills the multi-position slots (FLEX on DK NFL).
            # Showdown: report what position the captain was. Classic: the flex slot.
            flex_slots = (["CPT"] if is_showdown
                          else [pos.name for pos in optimizer.settings.positions
                                if len(pos.positions) > 1])
            flex_df = None
            if flex_slots:
                flex = picks[picks["Position"].isin(flex_slots)].copy()
                if len(flex):
                    flex["Real position"] = flex["Name"].map(real_pos)
                    flex_df = (flex.groupby("Real position")
                               .agg(Lineups=("Lineup", "nunique"))
                               .reset_index().sort_values("Lineups", ascending=False))
                    flex_df["% of lineups"] = (flex_df["Lineups"] / total * 100).round(1)

            mix_df = None
            if mix_report:
                mix_df = pd.DataFrame(mix_report)
                mix_df["Actual %"] = (mix_df["Lineups"] / max(len(lineups), 1) * 100).round(1)

            st.session_state.results = {
                "lineups": result_df,
                "exposure": exposure_df,
                "mix": mix_df,
                "upload": upload_df,
                "teams": team_df,
                "flex": flex_df,
            }

        results = st.session_state.get("results")
        if results is not None:
            st.subheader("Generated Lineups")
            st.dataframe(results["lineups"], width="stretch")

            if results["mix"] is not None:
                st.subheader("Stack Mix")
                st.dataframe(results["mix"], width="content", hide_index=True)

            if results.get("upload") is not None:
                st.download_button(
                    "⬇️ Download DK upload CSV",
                    results["upload"].to_csv(index=False),
                    "dk_upload.csv", "text/csv", type="primary",
                    help="One row per lineup, one column per roster slot, "
                         "with the DK player IDs — ready to upload.")

            st.subheader("Exposure Summary")
            exp = results["exposure"]
            exp_positions = sorted(
                {p for joined in exp["Position"].dropna() for p in str(joined).split("/")},
                key=lambda x: available_positions.index(x) if x in available_positions else 99,
            )
            exp_filter = st.pills(
                "Filter by position", exp_positions, selection_mode="multi",
                key="exp_pos_filter", label_visibility="collapsed",
            )
            if exp_filter:
                wanted = set(exp_filter)
                shown = exp[exp["Position"].apply(
                    lambda j: bool(set(str(j).split("/")) & wanted))]
                st.caption(f"Showing {len(shown)} of {len(exp)} players "
                           f"({'/'.join(sorted(wanted))}).")
            else:
                shown = exp
            st.dataframe(shown, width="stretch", hide_index=True)

            tcol, fcol = st.columns([3, 2])
            with tcol:
                st.subheader("Team Exposure")
                st.caption("Lineups using at least one player from the team. "
                           "**3+ stack** counts lineups where that team supplied "
                           "three or more.")
                st.dataframe(results["teams"], width="stretch", hide_index=True)
            with fcol:
                st.subheader("Captain Breakdown" if is_showdown else "FLEX Breakdown")
                if results.get("flex") is None:
                    st.caption("No multi-position slots on this roster.")
                else:
                    st.caption("Which position was captained." if is_showdown
                               else "Which position actually filled the flex slot.")
                    st.dataframe(results["flex"], width="stretch", hide_index=True)

    except Exception as e:
        st.error(f"Error processing file: {e}")
else:
    st.info("Upload a DK/FD salary CSV to get started.")
