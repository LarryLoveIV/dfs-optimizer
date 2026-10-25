import streamlit as st
import numpy as np
import pandas as pd
import tempfile
import os
import re
import io
import time
from collections import Counter, defaultdict
from itertools import combinations
from pydfs_lineup_optimizer import (get_optimizer, Site, Sport, LineupOptimizerException,
                                    TotalExposureStrategy, AfterEachExposureStrategy,
                                    PlayersGroup, Stack, RandomFantasyPointsStrategy)
# NestedPlayersGroup isn't re-exported from the package root, so it has to come
# from the module that defines it.
from pydfs_lineup_optimizer.stacks import NestedPlayersGroup
from pydfs_lineup_optimizer.rules import OptimizerRule
from pydfs_lineup_optimizer.solvers import SolverSign

import sim

try:
    # Reads DraftKings' public lobby endpoints, no login. Optional: without the
    # package the app only offers the CSV upload.
    import dk_lobby
except Exception:
    dk_lobby = None

try:
    # Lineup checks and the chat assistant; optional like the lobby.
    import agent
except Exception:
    agent = None

try:
    # Same model, faster engine. Verified to produce identical lineups; if the
    # import fails the app just falls back to the library's default solver.
    from or_solver import ORToolsSat as FAST_SOLVER
except Exception:
    FAST_SOLVER = None

# Contest tiers by entry count — the user's own definitions. Each tier sets two
# things:
#   - Randomness: jitter applied to each projection before every solve. The
#     percentages come from optimizer guidance (Footballguys, Stokastic) putting
#     the usable band at 2-5% with 5% a ceiling; neither ETR nor OWS prescribes
#     a number. Measured on a 100-lineup run with this pool: 2-5% costs ~0.1
#     projection points while cutting top-player exposure from 85% to 70%; 12%
#     costs a full point and 25% costs 3.4. Smaller fields reward tight rosters,
#     bigger ones accepting variance to win outright (One Week Season's framing).
#   - Ownership: TwoGunGPT files carry a small-field and a large-field (Milly
#     Maker) column; the tiers between step from one to the other.
# (upper entry limit, name, randomness, share of small-field ownership)
CONTEST_TIERS = [
    (3_000,        "Small",  0.02, 1.0),
    (5_000,        "Medium", 0.03, 0.67),
    (14_000,       "Large",  0.04, 0.33),
    (float("inf"), "Milly",  0.05, 0.0),
]


def contest_tier(entries):
    """(name, randomness, small-field ownership share) for a contest size."""
    for limit, name, dev, small in CONTEST_TIERS:
        if entries is not None and entries <= limit:
            return name, dev, small
    return CONTEST_TIERS[-1][1:]


def showdown_defaults(entries):
    """(max exposure %, lineups built per lineup kept, target finish %) for a
    showdown field. Money-tested in Oct 2026 on 7 real showdowns with DK payout
    tables (237-entry booster, 5,945 and 59,453-entry Dime Packages, 28-entry
    Quintuple Up): 100% exposure beat 50% by 37-150+ ROI points in multi-entry
    fields; 40 per lineup beat 20 only in the mid-size Dime; a top-5% target
    beat 0.5% only in the tiny Quintuple Up. Without a field size the old
    defaults stand."""
    if entries is None:
        return 50, 20, 0.5
    if entries <= 100:
        return 50, 20, 5.0
    if 3_000 < entries <= 14_000:
        return 100, 40, 0.5
    return 100, 20, 0.5


def randomness_for_field(entries):
    name, dev, _ = contest_tier(entries)
    lower = {"Small": "up to 3,000", "Medium": "3,001-5,000", "Large": "5,001-14,000",
             "Milly": "over 14,000"}[name]
    return dev, f"{name} tier ({lower} entries)"


# Simulated opponents per run. Enough to place the field's top 1% each sim;
# more costs memory (field x sims scores) without moving the threshold much.
FIELD_LINEUPS = 3000


def ownership_label(entries):
    name, _, small = contest_tier(entries)
    if small >= 1:
        return f"small-field ownership, {name} tier"
    if small <= 0:
        return f"Milly (large-field) ownership, {name} tier"
    return f"{small:.0%} small-field / {1 - small:.0%} Milly ownership, {name} tier"


def blend_ownership(frame, entries):
    """The tier's mix of TwoGunGPT's small-field and Milly ownership columns."""
    large = pd.to_numeric(frame.get("Ownership"), errors="coerce")
    small = frame.get("Small Field Ownership")
    if small is None:
        return large
    small = pd.to_numeric(small, errors="coerce").fillna(large)
    w = contest_tier(entries)[2]
    return w * small + (1 - w) * large.fillna(small)


def sim_pool(frame, pool_players):
    """The table the simulator works on: one row per player, with his real
    position (Pos), opponent (Opp) and the projection used as the simulated
    mean (Proj) — the true projection even when the optimizer chases Ceiling
    or Nuke."""
    opp_of = {}
    for p in pool_players:
        gi = p.game_info
        if gi and gi.home_team and gi.away_team:
            opp_of[p.full_name] = gi.away_team if p.team == gi.home_team else gi.home_team
    pool = frame.copy()
    pool["Pos"] = pool["Position"].astype(str).str.split("/").str[0]
    pool["Opp"] = pool["Name"].map(opp_of)
    mean_col = next(c for c in ("Projection", "ppg_projection", "AvgPoints")
                    if c in pool.columns)
    pool["Proj"] = pd.to_numeric(pool[mean_col], errors="coerce").fillna(0)
    return pool.dropna(subset=["Opp"]).reset_index(drop=True)


@st.cache_data(ttl=300, show_spinner="Reading the DraftKings lobby…")
def lobby_slates():
    return dk_lobby.slates()


@st.cache_data(ttl=300, show_spinner="Loading players from DraftKings…")
def lobby_salaries(draft_group_id, kind):
    """(salary-export CSV text, time fetched) for one slate."""
    return dk_lobby.salaries_csv(draft_group_id, kind), time.strftime("%I:%M %p").lstrip("0")


@st.cache_data(ttl=120, show_spinner="Checking injury news on DraftKings…")
def live_status(ids, kind, draft_group_id):
    """{draftable ID: DraftKings injury tag} for the built lineups' players."""
    return dk_lobby.player_status(ids, kind, draft_group_id)


def secret(name):
    """A value from the environment or .streamlit/secrets.toml, else ""."""
    if os.environ.get(name):
        return os.environ[name]
    try:
        return st.secrets.get(name, "")
    except Exception:          # no secrets file at all
        return ""


@st.fragment
def lineup_assistant(results, merged_df, csv_text, is_showdown, target, excluded,
                     roster_size, budget, draft_group_id):
    """Checks every lineup, then answers questions about the build. A fragment,
    so a question reruns only this section, not the player pool above."""
    st.subheader("Lineup assistant")
    build = results.get("built_at")
    lineups = results["lineups"]
    picks = lineups[lineups["Position"] != "TOTAL"]

    live, live_err = None, None
    with st.container(horizontal=True, vertical_alignment="center"):
        if st.button("Recheck injuries", icon=":material/refresh:", key="assist_recheck",
                     help="Statuses are cached for 2 minutes. Recheck near lock."):
            live_status.clear()
        if dk_lobby is None:
            live_err = "the DraftKings lobby package isn't installed"
        else:
            try:
                live = live_status(tuple(sorted(map(str, picks["ID"]))),
                                   "showdown" if is_showdown else "classic", draft_group_id)
                if not live:
                    live, live_err = None, "no open DraftKings slate has these players"
            except Exception as exc:
                live_err = str(exc)
        st.caption(f"Live DraftKings status checked {time.strftime('%I:%M %p').lstrip('0')}."
                   if live else f"Live injury status unavailable ({live_err}).")

    found = agent.checks(lineups, merged_df, results["exposure"], live, excluded,
                         roster_size, budget)
    if found.empty:
        st.success("No problems found in these lineups.", icon=":material/check_circle:")
    else:
        fixes = int(found["Severity"].eq("Fix").sum())
        (st.error if fixes else st.warning)(
            f"{fixes} to fix, {len(found) - fixes} to watch.", icon=":material/warning:")
        st.dataframe(found, hide_index=True, width="stretch")

    provider = st.session_state.get("assist_provider", "Gemini")
    model = st.session_state.get("assist_model") or agent.PROVIDERS[provider]["model"]
    key = (st.session_state.get("assist_key") or secret("GEMINI_API_KEY")
           if agent.PROVIDERS[provider]["needs_key"] else "")
    if agent.PROVIDERS[provider]["needs_key"] and not key:
        st.info("Add a free Gemini API key under **Lineup assistant** in the sidebar to "
                "ask questions about these lineups.", icon=":material/key:")
        return

    # One conversation per build; a new build starts it over.
    if st.session_state.get("assist_build") != build:
        st.session_state.assist_build = build
        st.session_state.assist_chat = []
    chat = st.session_state.assist_chat

    box = st.container(height=520 if chat else "content")
    with box:
        for m in chat:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])
    suggestions = ["Summarize these lineups", "Are there any mistakes?",
                   "Is anyone not playing?", "Which lineups look strongest and why?"]
    # The chip is cleared as it's sent, so a failed answer can't resend it.
    def take_chip():
        st.session_state.assist_pending = st.session_state.assist_try
        st.session_state.assist_try = None
    st.pills("Try", suggestions, key="assist_try", on_change=take_chip,
             label_visibility="collapsed")
    prompt = (st.chat_input("Ask about these lineups", key="assist_input",
                            submit_mode="disable")
              or st.session_state.pop("assist_pending", None))
    if not prompt:
        return
    chat.append({"role": "user", "content": prompt})
    data = agent.context(results, merged_df, found, live is not None, csv_text,
                         is_showdown, target)
    messages = ([{"role": "system", "content": agent.system(is_showdown) + "\n\n" + data}]
                + chat[-12:])
    with box:
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            try:
                reply = st.write_stream(agent.stream(provider, model, key, messages))
            except Exception as exc:
                reply = None
                st.error(f"The assistant couldn't answer: {exc}")
    if reply:
        chat.append({"role": "assistant", "content": reply})
    else:
        chat.pop()


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

if contest_mode != "Tournament (GPP)":
    field_entries = None
sim_on = False
if sport == "NFL" and contest_mode != "Cash / single entry":
    st.sidebar.subheader("Simulation")
    sim_on = st.sidebar.toggle(
        "Pick lineups by simulation", value=True,
        help="Builds extra lineups, simulates the slate thousands of times with "
             "teammates' and opponents' scores correlated the way they are in real games, "
             "then keeps the set of lineups that most often finishes near the top. Stack, "
             "flex and QB + RB shares are kept.")
    if sim_on and field_entries is None:
        field_entries = st.sidebar.number_input(
            "Field size (entries)", min_value=2, max_value=2_000_000, value=5_000,
            step=500, key="sim_field_entries",
            help="Picks large- or small-field ownership and sets how many duplicates "
                 "to expect.")
# Showdown defaults follow the field size; changing the field resets them.
sd_exp, sd_mult, sd_top = showdown_defaults(field_entries)
if sim_on:
    sim_mult = st.sidebar.select_slider(
        "Lineups to build and compare",
        options=[2, 3, 5, 8, 10, 20, 30, 40, 60], value=sd_mult if is_showdown else 30,
        format_func=lambda m: f"{m * num_lineups:,}",
        help="The optimizer builds this many lineups under all your rules, the sim "
             "plays the slate out thousands of times, and only your best "
             f"{num_lineups} are kept. A bigger pile gives the sim more to choose "
             "from, at the cost of a slower build. Main slate: 30 per lineup beat 5 "
             "on every build and measure (Week 5, about 90 seconds for 40). "
             "Showdown, money-tested on 7 games: 40 per lineup beat 20 in the "
             "5,945-entry Dime Package, so fields of 3,001-14,000 default to 40; "
             "in boosters and 50k+ Dimes more lineups didn't pay.")
    st.sidebar.caption(
        f"Build **{sim_mult * num_lineups:,}** → keep the best **{num_lineups}** "
        f"(1 in {sim_mult}).")
    sim_runs = st.sidebar.select_slider(
        "Simulations", options=[1000, 2500, 5000, 10000, 50000],
        value=50000 if is_showdown else 5000,
        help="More sims separate genuine upside from lucky draws, which matters "
             "more the bigger the pile. On a showdown 50,000 adds only seconds; on a "
             "main slate it needs a lot of memory, so start at 5,000 there.")
    sim_top = st.sidebar.select_slider(
        "Target finish", options=[0.1, 0.5, 1.0, 2.0, 5.0],
        value=sd_top if is_showdown else 0.5,
        format_func=lambda v: f"top {v:g}%",
        help="Tested for money (Week 3 and 5 main, Super Booster and standard GPP "
             "payouts, 8 builds each): top 0.5% beat 1% in every contest and scenario. "
             "On showdown, top 5% wins only in tiny fields (100 entries or fewer). "
             "A lineup scores a hit in a sim when it lands in this top slice of all "
             "the lineups built. They're all optimized, so this is a much higher bar "
             "than the same slice of a real contest field.") / 100

# After the contest settings, so the showdown default can follow the field size.
st.sidebar.subheader("Exposure")
global_max_exp = st.sidebar.slider(
    "Global max exposure %", min_value=0, max_value=100,
    value=sd_exp if is_showdown else 50, step=5,
    help=("Set from the field size. Money-tested on 7 showdowns: with several "
          "entries, 100% (no cap) beat 50% by a wide margin in boosters and Dime "
          "Packages, and 80% lost; 50% stays only for tiny fields of 100 entries or "
          "fewer."
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
         "the main diversification lever (1 and 3 both lost money in testing).",
)

strategy_label = st.sidebar.selectbox(
    "Exposure strategy", ["Total", "After each lineup"],
    help="Total: share of all lineups. After each lineup: share of lineups generated so far.",
)
strategy_map = {"Total": TotalExposureStrategy, "After each lineup": AfterEachExposureStrategy}

if agent is not None:
    with st.sidebar.expander("Lineup assistant", icon=":material/smart_toy:"):
        assist_provider = st.segmented_control(
            "Model", list(agent.PROVIDERS), default="Gemini", required=True,
            key="assist_provider",
            help="Gemini: Google's free API tier (a key from aistudio.google.com, "
                 "limited requests per day). Ollama: a free model on this computer — "
                 "install Ollama and run `ollama pull qwen3:14b` first.")
        st.text_input("Model name", placeholder=agent.PROVIDERS[assist_provider]["model"],
                      key="assist_model",
                      help="Leave blank for the default. Change it if the provider "
                           "retires the model.")
        if agent.PROVIDERS[assist_provider]["needs_key"]:
            st.text_input("Gemini API key", type="password", key="assist_key",
                          placeholder="saved key in use" if secret("GEMINI_API_KEY") else "",
                          help="Kept only for this session. To skip typing it, put "
                               "GEMINI_API_KEY = \"...\" in ~/.streamlit/secrets.toml.")

site_map = {"DraftKings": Site.DRAFTKINGS,
            "DraftKings Showdown": Site.DRAFTKINGS_CAPTAIN_MODE}
sport_map = {"NFL": Sport.FOOTBALL, "NBA": Sport.BASKETBALL, "MLB": Sport.BASEBALL, "NHL": Sport.HOCKEY}

st.subheader("1. Upload Files")
col1, col2 = st.columns(2)
with col1:
    dk_content = None
    lobby_group_id = None
    sources = (["DraftKings lobby", "Upload CSV"] if sport == "NFL" and dk_lobby
               else ["Upload CSV"])
    salary_source = st.segmented_control("Salaries", sources, default=sources[0],
                                         required=True, key="salary_source")
    if salary_source == "Upload CSV":
        dk_file = st.file_uploader("DK/FD Salaries CSV", type="csv", key="dk")
        st.link_button("Get DraftKings salaries", "https://www.draftkings.com/lobby#/NFL",
                       icon=":material/open_in_new:",
                       help="Opens the NFL lobby. Open a contest for the slate you want and "
                            "use Export to CSV (Export Players List), then drop the file in "
                            "the box above.")
        if dk_file is not None:
            dk_content = dk_file.getvalue().decode("utf-8-sig")
    else:
        kind = "showdown" if is_showdown else "classic"
        lobby_error, open_slates = None, []
        try:
            open_slates = [x for x in lobby_slates() if x["kind"] == kind]
        except Exception as exc:
            lobby_error = exc
        if lobby_error:
            st.warning(f"Couldn't reach the DraftKings lobby ({lobby_error}). Switch to "
                       "**Upload CSV** and use the exported salary file instead.")
        elif not open_slates:
            st.info(f"No open {kind} slates in the DraftKings lobby right now.")
        else:
            # Labels are the options; a repeat (same tag, time and contest
            # count) gets its draft group id so each stays distinct.
            seen = Counter(x["label"] for x in open_slates)
            by_label = {(x["label"] if seen[x["label"]] == 1
                         else f"{x['label']} · #{x['id']}"): x for x in open_slates}
            labels = list(by_label)
            main = next((i for i, lab in enumerate(labels) if lab.startswith("Main")), 0)
            pick = by_label[st.selectbox("Slate", labels, index=main,
                                         key=f"dk_slate_{kind}")]
            lobby_group_id = pick["id"]
            try:
                dk_content, fetched_at = lobby_salaries(pick["id"], kind)
            except Exception as exc:
                st.warning(f"Couldn't load that slate's players ({exc}). Try again, or "
                           "switch to **Upload CSV**.")
            with st.container(horizontal=True, vertical_alignment="center"):
                if dk_content:
                    n = dk_content.count("\n") - 1
                    st.caption(f"{n} players from DraftKings, fetched {fetched_at}.")
                if st.button("Refresh", icon=":material/refresh:", key="dk_refresh",
                             help="Salaries rarely move, but players get scratched — "
                                  "refresh near lock to drop anyone already ruled out."):
                    lobby_slates.clear()
                    lobby_salaries.clear()
                    st.rerun()
with col2:
    cheatsheet_file = st.file_uploader("Projections Cheatsheet CSV", type="csv", key="cheat")
    # The download itself needs a signed-in browser (TwoGunGPT's terms bar
    # scripted access), so the app links to the board rather than fetching it.
    st.link_button("Get TwoGunGPT projections", "https://twogungpt.com/projections/",
                   icon=":material/open_in_new:",
                   help="Opens the Ceiling Board. Sign in, pick the slate and press "
                        "DOWNLOAD, then drop the CSV in the box above.")


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


class CombinedExposureRule(OptimizerRule):
    """Cap a player's exposure across every pool entry they occupy.

    Showdown holds each player twice (CPT and FLEX) as separate Player objects,
    so the library's own max_exposure caps each copy independently — a 50% cap
    lets a player reach 100% of lineups. This counts the real player once per
    lineup and blocks every copy once the cap is reached.
    """

    def __init__(self, optimizer, players_dict, context):
        super().__init__(optimizer, players_dict, context)
        self.caps = getattr(optimizer, "combined_exposure", None) or {}
        self.total = context.total_lineups or 1
        self.used = Counter()
        self.var_key = {}
        self.key_vars = defaultdict(list)
        for player, variable in players_dict.items():
            key = player.full_name
            self.var_key[variable.name] = key
            self.key_vars[key].append(variable)

    def apply_for_iteration(self, solver, result):
        for key, cap in self.caps.items():
            if not cap:
                continue
            if self.used[key] / self.total >= cap:
                variables = self.key_vars.get(key)
                if variables:
                    solver.add_constraint(variables, None, SolverSign.EQ, 0)

    def post_optimize(self, solved_variables):
        keys = {self.var_key[name] for name in solved_variables if name in self.var_key}
        for key in keys:
            self.used[key] += 1


class PlayerComboRule(OptimizerRule):
    """Min/max share of lineups in which a set of players all appear together.

    The library only knows per-player exposure, so a combo needs its own rule.
    Each combo arrives as absolute lineup counts for this run (the caller
    converts percentages, so mixed batches can share one overall target).

    Max: once the combo has been used max_count times, the players' variables
    may sum to at most k-1, i.e. never all of them again.
    Min: forcing is paced across the run rather than bunched at the start or
    end, and skipped for any lineup where one of the players is at their own
    max exposure — forcing then would make the lineup infeasible. Under "after
    each lineup" a capped player is blocked every other lineup, so the pacing
    catches up on the next open one. If a player is capped for good, the combo
    falls short rather than failing the run; the results table shows the gap.
    """

    def __init__(self, optimizer, players_dict, context):
        super().__init__(optimizer, players_dict, context)
        self.combos = getattr(optimizer, "player_combos", None) or []
        self.total = context.total_lineups or 1
        self.used = [0] * len(self.combos)
        self.var_key = {}
        self.key_vars = defaultdict(list)
        caps = {}
        for player, variable in players_dict.items():
            self.var_key[variable.name] = player.full_name
            self.key_vars[player.full_name].append(variable)
            cap = (player.max_exposure if player.max_exposure is not None
                   else context.max_exposure)
            if cap is not None:
                caps[player.full_name] = cap
        # Mirror the library's own cap bookkeeping so we know who is blocked.
        # Showdown caps live in CombinedExposureRule, which counts on totals.
        combined = getattr(optimizer, "combined_exposure", None)
        if combined:
            caps.update(combined)
            strategy = TotalExposureStrategy
        else:
            strategy = context.exposure_strategy
        self.caps = strategy(caps, self.total)

    def apply_for_iteration(self, solver, result):
        built = self.total - self.context.remaining_lineups
        for i, combo in enumerate(self.combos):
            names = combo["players"]
            if any(not self.key_vars.get(nm) for nm in names):
                continue          # someone isn't in the pool; can't apply
            max_count, min_count = combo.get("max_count"), combo.get("min_count")
            if max_count is not None and self.used[i] >= max_count:
                variables = [v for nm in names for v in self.key_vars[nm]]
                solver.add_constraint(variables, None, SolverSign.LTE, len(names) - 1)
            elif (min_count and not any(self.caps.is_reached_exposure(nm) for nm in names)
                  and (self.used[i] < -(-min_count * (built + 1) // self.total)
                       or self.slots_left(names) <= min_count - self.used[i])):
                # Showdown copies: at least one of each player's CPT/FLEX vars.
                for nm in names:
                    solver.add_constraint(self.key_vars[nm], None, SolverSign.GTE, 1)

    def slots_left(self, names):
        """Lineups the most-capped player can still join under a total cap.

        A player who keeps getting picked on his own can burn through his cap
        before the paced combo is done; once his slots are down to what the
        combo still owes, the combo has to take them.
        """
        if not isinstance(self.caps, TotalExposureStrategy):
            return float("inf")
        left = float("inf")
        for nm in names:
            cap = self.caps.exposures.get(nm)
            if cap:
                # The epsilon absorbs float error: a batch cap of a/c times c
                # can land a hair above a and ceil to a+1.
                left = min(left, -(-(cap * self.total - 1e-9) // 1)
                           - self.caps.used_vars.get(nm, 0))
        return left

    def post_optimize(self, solved_variables):
        picked = {self.var_key[name] for name in solved_variables if name in self.var_key}
        self.caps.set_used(list(picked))
        for i, combo in enumerate(self.combos):
            if set(combo["players"]) <= picked:
                self.used[i] += 1


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


STACK_CATCHERS = ("WR", "TE")


def build_qb_stack(pool, qb_side, bring_back, max_exposure=None, partner="pass"):
    """A single Stack requiring: a QB, (qb_side-1) of his teammates, and
    `bring_back` players from his opponent.

    Teammates and bring-backs count only if they catch passes (WR/TE): a QB's
    points move with his receivers' (r ~0.35) but not his RB's (~0.05) or his
    DST's, so a QB + RB "stack" pays the cost of stacking for none of the
    correlation. partner="rb" is the deliberate exception — one of the
    teammates must be an RB, the rest are still WR/TE.

    One stack rather than two means the QB can anchor the same-team part and the
    game part at once — separate stacks may not share a player, and with one QB
    slot that makes a QB-anchored stack plus a game stack impossible.
    """
    catches = lambda p: any(pos in STACK_CATCHERS for pos in p.positions)
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
            mates = [p for p in by_team[own] if "QB" not in p.positions and catches(p)]
            rbs = [p for p in by_team[own] if "RB" in p.positions]
            others = [p for p in by_team[opp] if catches(p)]
            need_rb = partner == "rb" and qb_side > 1
            need_mates = qb_side - 1 - need_rb
            if (not qbs or len(mates) < need_mates or len(others) < bring_back
                    or (need_rb and not rbs)):
                continue
            members = [PlayersGroup(qbs, min_from_group=1)]
            if need_rb:
                members.append(PlayersGroup(rbs, min_from_group=1))
            if need_mates:
                members.append(PlayersGroup(mates, min_from_group=need_mates))
            if bring_back:
                members.append(PlayersGroup(others, min_from_group=bring_back))
            groups.append(NestedPlayersGroup(groups=members, max_exposure=max_exposure))
    return Stack(groups=groups) if groups else None


def count_combos(lineups_df, size):
    """How many lineups contain each group of `size` players.

    Every subset of every lineup is counted, so a 9-man roster yields 36 pairs,
    84 trios and 126 foursomes per lineup — a few thousand at 150 lineups.
    """
    picks = lineups_df[lineups_df["Position"] != "TOTAL"]
    counts = Counter()
    for _, names in picks.groupby("Lineup")["Name"]:
        counts.update(combinations(sorted(set(names)), size))
    return counts


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


if dk_content is not None:
    content = dk_content

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
        proj_label = "ppg_projection"
        if cheatsheet_file is not None:
            cheat_df = pd.read_csv(cheatsheet_file)
            cheat_cols = list(cheat_df.columns)   # before Name/_key are added
            # TwoGunGPT's Ceiling Board download: Projection/Median/Ceiling/Nuke
            # instead of ppg_projection, and DraftKings IDs to match on.
            is_twogun = {"Name", "Projection", "Median", "Ceiling", "Nuke"} <= set(cheat_cols)
            if is_twogun:
                # A showdown board carries CPT columns; a main-slate board doesn't.
                # The wrong one would mix ownership: a showdown file's "Ownership"
                # is captain + flex for two teams, not a classic field's ownership.
                file_is_showdown = bool({"CPT Salary", "CPT Ownership"} & set(cheat_cols))
                if file_is_showdown != is_showdown:
                    st.error(
                        f"This is a TwoGunGPT **{'showdown' if file_is_showdown else 'main-slate'}** "
                        f"file, but the site is set to **{site}**. Switch the site in the "
                        "sidebar or upload the "
                        f"{'showdown' if is_showdown else 'main-slate'} file — mixing them would "
                        "use the wrong projections and ownership.")
                    st.stop()
                proj_label = st.segmented_control(
                    "Optimize on", ["Projection", "Median", "Ceiling", "Nuke"],
                    default="Ceiling", required=True, key="twogun_target",
                    help="TwoGunGPT file detected. Projection is his average game, "
                         "Median his 50th percentile, Ceiling a top-20% game and Nuke "
                         "a top-5% game. Ceiling is the default: money-tested on Weeks 3-5 "
                         "with the simulation on, Ceiling with an open flex beat every other "
                         "setup in boosters and standard GPPs.")
                cheat_df["ppg_projection"] = cheat_df[proj_label]
                # "Out ..." -> O, "Questionable (assumed playing)" -> Q, so the
                # existing ruled-out filter applies.
                status = cheat_df.get("Injury Status", pd.Series(index=cheat_df.index))
                cheat_df["injury_status"] = (status.fillna("").astype(str).str.strip()
                                             .str[:1].str.upper())
                # Key by DraftKings ID where the salary file has it, so name
                # spellings (Josh/Joshua, DST nicknames) can't break the match.
                # The pool holds CPT and FLEX copies on showdown; both IDs map.
                name_by_id = {str(p.id): p.full_name for p in players}
                ids = (pd.to_numeric(cheat_df["ID"], errors="coerce").astype("Int64")
                       .astype(str) if "ID" in cheat_df.columns
                       else pd.Series("", index=cheat_df.index))
                cheat_df["Name"] = ids.map(name_by_id).fillna(cheat_df["Name"])
            if not is_twogun and {"first_name", "last_name"} <= set(cheat_df.columns):
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
            if is_twogun:
                present = ["injury_status", "ppg_projection",
                           "Projection", "Median", "Ceiling", "Nuke"]
                present += [c for c in ("Ownership", "Small Field Ownership",
                                        "CPT Ownership", "FLEX Ownership")
                            if c in cheat_df.columns]
                missing = []
            else:
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
                f"Optimizing on **{proj_label}** — {len(proj_map)} players. "
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

        # --- Optimal %: how often each player is in a simulated game's best
        # lineup. Beside projected ownership it reads as leverage.
        opt_key = (hash(content), is_showdown,
                   getattr(cheatsheet_file, "file_id", None) if cheatsheet_file else None)
        if sport == "NFL" and cheatsheet_file is not None:
            opt_runs = 2000 if is_showdown else 1000
            with st.container(horizontal=True, vertical_alignment="center"):
                if st.button("Calculate Optimal %", icon=":material/insights:",
                             help=f"Simulates {opt_runs:,} games and solves each one's "
                                  "highest-scoring legal lineup. Optimal % is how often a "
                                  "player is in it; Leverage is Optimal % minus his "
                                  "projected ownership — positive means the field is "
                                  "likely underplaying him."):
                    with st.spinner(f"Solving the best lineup in {opt_runs:,} simulated "
                                    "games…"):
                        opool = sim_pool(merged_df, players)
                        opt, cpt = sim.optimal_rates(opool, sim.simulate(opool, opt_runs),
                                                     n=opt_runs, showdown=is_showdown)
                    st.session_state.optimal = {
                        "key": opt_key, "runs": opt_runs,
                        "opt": dict(zip(opool["Name"], opt.round(1))),
                        "cpt": dict(zip(opool["Name"], cpt.round(1))) if cpt is not None
                               else None}
                stored = st.session_state.get("optimal")
                if stored and stored["key"] == opt_key:
                    st.caption(f"Optimal % from {stored['runs']:,} simulated games.")
            stored = st.session_state.get("optimal")
            if stored and stored["key"] == opt_key:
                merged_df["Optimal %"] = merged_df["Name"].map(stored["opt"])
                if stored["cpt"] is not None:
                    merged_df["CPT opt %"] = merged_df["Name"].map(stored["cpt"])
                own_ref = (pd.to_numeric(merged_df.get("Ownership"), errors="coerce")
                           if is_showdown else
                           blend_ownership(merged_df, field_entries)
                           if "Ownership" in merged_df.columns else None)
                if own_ref is not None:
                    merged_df["Leverage"] = (merged_df["Optimal %"] - own_ref).round(1)

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
                "Ownership": (st.column_config.NumberColumn(
                    "Total own", format="%.1f%%",
                    help="TwoGunGPT's projected ownership, captain and flex combined.")
                    if is_showdown else st.column_config.NumberColumn(
                    "Milly", format="%.1f%%",
                    help="TwoGunGPT's projected ownership in a large-field (Milly Maker) "
                         "contest.")),
                "Optimal %": st.column_config.NumberColumn(
                    "Optimal %", format="%.1f%%",
                    help="Share of simulated games in which he's in that game's "
                         "highest-scoring legal lineup."),
                "CPT opt %": st.column_config.NumberColumn(
                    "CPT opt %", format="%.1f%%",
                    help="Share of simulated games in which he's the optimal captain."),
                "Leverage": st.column_config.NumberColumn(
                    "Leverage", format="%+.1f",
                    help="Optimal % minus projected ownership (the contest tier's mix "
                         "on a classic slate; total ownership on a showdown). Positive "
                         "means the field likely underplays him."),
                "CPT Ownership": st.column_config.NumberColumn(
                    "CPT own", format="%.1f%%",
                    help="TwoGunGPT's projected share of lineups with him at captain."),
                "FLEX Ownership": st.column_config.NumberColumn(
                    "FLEX own", format="%.1f%%",
                    help="TwoGunGPT's projected share of lineups with him at flex."),
                "Small Field Ownership": st.column_config.NumberColumn(
                    "Small", format="%.1f%%",
                    help="TwoGunGPT's projected ownership in a small-field contest."),
                # DraftKings' season average — 0 from the lobby, and unused once a
                # projections file sets the points, so it's hidden then.
                **({"AvgPoints": None} if cheatsheet_file is not None else {}),
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

        st.markdown("**Player combos**")
        st.caption(
            "Share of lineups in which every player on a row appears *together*. Set "
            "Min and Max to the same value for an exact share — e.g. Bowers + Kelce at "
            "20/20 puts the pair in 20% of lineups. Min alone forces at least that "
            "many; Max alone caps how often the pair can show up. Each player's own "
            "exposure limits still apply, so a combo can't exceed either player's max."
        )
        st.session_state.setdefault("combo_rows", [])
        st.session_state.setdefault("combo_nonce", 0)
        pool_names = sorted(merged_df["Name"].unique())
        combo_df = pd.DataFrame(
            [{"Players": r["players"], "Min %": r["min"], "Max %": r["max"]}
             for r in st.session_state.combo_rows],
            columns=["Players", "Min %", "Max %"])
        combo_df["Min %"] = combo_df["Min %"].astype("Float64")
        combo_df["Max %"] = combo_df["Max %"].astype("Float64")
        edited_combos = st.data_editor(
            combo_df,
            num_rows="dynamic",
            width="stretch",
            hide_index=True,
            column_config={
                "Players": st.column_config.MultiselectColumn(
                    "Players", options=pool_names, width="large", required=True,
                    help="Two or more players who must appear together."),
                "Min %": st.column_config.NumberColumn(
                    "Min %", min_value=0, max_value=100, step=5,
                    help="At least this share of lineups contain all of them."),
                "Max %": st.column_config.NumberColumn(
                    "Max %", min_value=0, max_value=100, step=5,
                    help="At most this share of lineups contain all of them. "
                         "0 means never together."),
            },
            key=f"combo_editor_{st.session_state.combo_nonce}",
        )
        collected_combos = []
        for _, row in edited_combos.iterrows():
            names = row["Players"]
            names = [str(n) for n in names] if isinstance(names, (list, tuple)) else []
            if not names:
                continue
            collected_combos.append({
                "players": sorted(set(names)),
                "min": float(row["Min %"]) if pd.notna(row["Min %"]) else None,
                "max": float(row["Max %"]) if pd.notna(row["Max %"]) else None,
            })
        if collected_combos != st.session_state.combo_rows:
            st.session_state.combo_rows = collected_combos
            st.session_state.combo_nonce += 1
            st.rerun()

        name_positions = {n: str(p).split("/")
                          for n, p in zip(merged_df["Name"], merged_df["Position"])}

        def combo_fits(names, flex=None):
            """Can these players share one roster when the flex is forced to `flex`?

            Two TEs need the flex to take one of them, so a combo like Bowers +
            Kelce is impossible in a batch that forces an RB or WR into the flex.
            Showdown slots take any position, so everything fits there.
            """
            if is_showdown:
                return True
            slots = [([flex] if flex and len(slot.positions) > 1 else list(slot.positions))
                     for slot in optimizer.settings.positions]

            def place(i, free):
                if i == len(names):
                    return True
                for j in free:
                    if set(name_positions.get(names[i], ())) & set(slots[j]):
                        if place(i + 1, free - {j}):
                            return True
                return False

            return place(0, frozenset(range(len(slots))))

        # Only rows that can actually constrain anything reach the optimizer.
        player_combos = []
        for idx, row in enumerate(st.session_state.combo_rows, start=1):
            label = " + ".join(row["players"])
            if len(row["players"]) < 2:
                st.warning(f"Combo {idx}: pick at least two players.")
                continue
            if row["min"] is None and row["max"] is None:
                st.warning(f"Combo {idx} ({label}): set a Min %, a Max %, or both.")
                continue
            if row["min"] is not None and row["max"] is not None and row["min"] > row["max"]:
                st.warning(f"Combo {idx} ({label}): min {row['min']:.0f}% is above max "
                           f"{row['max']:.0f}%.")
                continue
            if not combo_fits(row["players"]):
                st.warning(f"Combo {idx} ({label}): those positions can't all fit in "
                           "one roster.")
                continue
            gone = [n for n in row["players"] if n in excluded_names]
            if gone:
                st.warning(f"Combo {idx} ({label}): {', '.join(gone)} excluded, so the "
                           "combo is skipped.")
                continue
            if row["min"]:
                low = [n for n in row["players"]
                       if max_exp.get(n, global_max_exp) < row["min"]]
                if low:
                    st.warning(f"Combo {idx} ({label}): {', '.join(low)} capped below "
                               f"{row['min']:.0f}% — the combo min can't be reached.")
            player_combos.append(row)
            if row["min"] is not None and row["min"] == row["max"]:
                desc = f"exactly {row['min']:.0f}%"
            else:
                desc = " and ".join(filter(None, [
                    f"at least {row['min']:.0f}%" if row["min"] else None,
                    f"at most {row['max']:.0f}%" if row["max"] is not None else None,
                ]))
            st.caption(f"Combo {idx}: {label} together in {desc} of lineups")

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

            st.markdown("**Captain mix**")
            st.caption(
                "Minimum share of lineups captained by each position. The captain is worth "
                "1.5x, so it is the biggest single decision on a showdown slate. Shares under "
                "100% leave the rest open for the optimizer to choose (so QB 60% means at "
                "least 60% QB captains). Leave the table empty to let projections choose — "
                "which tends to settle on a handful of captains. Across 163 studied winning "
                "showdown slates the top-1% captains were roughly WR 33%, RB 28%, QB 21%."
            )
            cpt_positions = sorted(
                {r["Position"] for _, r in merged_df.iterrows() if r["Position"] != "?"},
                key=lambda x: available_positions.index(x)
                if x in available_positions else 99)
            st.session_state.setdefault("cpt_rows", [])
            st.session_state.setdefault("cpt_nonce", 0)
            cpt_df = pd.DataFrame(
                [{"Position": r["position"], "Share %": r["share"]}
                 for r in st.session_state.cpt_rows],
                columns=["Position", "Share %"])
            cpt_df["Share %"] = cpt_df["Share %"].astype("Float64")
            edited_cpt = st.data_editor(
                cpt_df, num_rows="dynamic", width="content", hide_index=True,
                column_config={
                    "Position": st.column_config.SelectboxColumn(
                        "Position", options=cpt_positions, required=True),
                    "Share %": st.column_config.NumberColumn(
                        "Share %", min_value=1, max_value=100, step=5, required=True),
                },
                key=f"cpt_editor_{st.session_state.cpt_nonce}",
            )
            collected_cpt = []
            for _, row in edited_cpt.iterrows():
                if pd.isna(row["Position"]) or pd.isna(row["Share %"]):
                    continue
                if row["Position"] in {r["position"] for r in collected_cpt}:
                    continue
                collected_cpt.append({"position": str(row["Position"]),
                                      "share": float(row["Share %"])})
            if collected_cpt != st.session_state.cpt_rows:
                st.session_state.cpt_rows = collected_cpt
                st.session_state.cpt_nonce += 1
                st.rerun()
            cpt_rows = st.session_state.cpt_rows

            if cpt_rows:
                ctot = sum(r["share"] for r in cpt_rows)
                if ctot > 100.5:
                    st.warning(f"Captain shares add up to {ctot:.0f}%, more than 100% — "
                               "they'll be scaled down proportionally.")
                # Under 100%, the remainder is an open batch where the optimizer picks
                # the captain, so each share works as a minimum.
                units = [(r["position"], r["share"]) for r in cpt_rows]
                if ctot < 99.5:
                    units.append((None, 100 - ctot))
                utot = sum(w for _, w in units)
                raw = [w / utot * num_lineups for _, w in units]
                counts = [int(x) for x in raw]
                for i in sorted(range(len(raw)), key=lambda j: raw[j] - counts[j],
                                reverse=True)[:num_lineups - sum(counts)]:
                    counts[i] += 1
                batch_plan = [{"specs": [], "flex": None, "captain": pos,
                               "weight": w / utot,
                               "label": f"{pos} captain" if pos else "open captain",
                               "count": c}
                              for (pos, w), c in zip(units, counts)]
                for b in batch_plan:
                    st.caption(f"→ {b['label']}: {b['weight']:.0%} → {b['count']} lineup"
                               f"{'s' if b['count'] != 1 else ''}")
                if any(b["count"] < 1 for b in batch_plan):
                    st.warning(
                        f"**Number of lineups is {num_lineups}**, too few to split across "
                        f"{len(batch_plan)} captain positions — some get 0."
                    )
                batch_counts = [b["count"] for b in batch_plan]
                mix_mode = True
            else:
                mix_mode = False
                batch_plan = [{"specs": [], "flex": None, "captain": None, "weight": 1.0,
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
            rb_stack_pct = 0
            if not has_game_info:
                st.warning("QB stacks need the **Game Info** column — see section 3.")
            else:
                # Default stack: QB + 2 of his WR/TEs + 1 bring-back. Tested for money
                # (Week 3 and 5 main, booster and standard GPP payouts, 8 builds per
                # shape): it beat no stack rule, QB+1, QB+2 and QB+1+1 everywhere.
                st.session_state.setdefault("qs_rows", [
                    {"qb_side": 3, "bring_back": 1, "share": None, "max_exp": None}])
                st.session_state.setdefault("qs_nonce", 0)
                # Blanks are stored as None but shown as what they mean: a lone row
                # gets 100% of lineups, and no cap is 100%. Collection maps them back,
                # so the build is the same as with blank cells.
                lone_row = len(st.session_state.qs_rows) == 1
                qs_df = pd.DataFrame(
                    [{"QB side": r["qb_side"], "Bring-back": r["bring_back"],
                      "Share %": (r.get("share") if r.get("share") is not None
                                  else 100.0 if lone_row else None),
                      "Max Exp %": r["max_exp"] if r["max_exp"] is not None else 100.0}
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
                            help="100 or blank = no cap. Caps how often any single game fills this "
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
                        "max_exp": (float(row["Max Exp %"])
                                    if pd.notna(row["Max Exp %"]) and row["Max Exp %"] < 100
                                    else None),
                    })
                if len(collected_qs) == 1 and collected_qs[0]["share"] == 100:
                    collected_qs[0]["share"] = None
                if collected_qs != st.session_state.qs_rows:
                    st.session_state.qs_rows = collected_qs
                    st.session_state.qs_nonce += 1
                    st.rerun()
                qs_rows = st.session_state.qs_rows
                if any(r["qb_side"] > 1 for r in qs_rows):
                    rb_stack_pct = st.number_input(
                        "QB + RB stacks %", min_value=0, max_value=100, value=5, step=5,
                        key="rb_stack_pct",
                        help="Share of stacked lineups whose QB is paired with his own RB. "
                             "Every other stack pairs him only with WRs/TEs, and bring-backs "
                             "are always WR/TE. A QB's score barely moves with his RB's "
                             "(r ~0.05, vs ~0.35 for his receivers), so keep this low.")

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
            # Default: no flex mix — the optimizer picks the flex. Money-tested on
            # Weeks 3-5 against a fixed RB 45 / WR 33 / TE 22 mix, the open flex won
            # in boosters and standard GPPs (and with Ceiling, everywhere).
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

            # Partner mix: each stacked batch splits into a WR/TE-partner slice
            # and a QB + RB slice. In the RB slice only the first stack that has
            # a teammate takes the RB, so one lineup holds one QB + RB pairing.
            def with_partner(specs, partner):
                out, used = [], False
                for s in specs:
                    take = partner == "rb" and not used and s["qb_side"] > 1
                    used |= take
                    out.append({**s, "partner": "rb" if take else "pass"})
                return out

            rb_share = rb_stack_pct / 100
            batch_plan = []
            for spec, qw in qb_units:
                specs = [spec] if spec else qb_stack_specs
                stacked = any(s["qb_side"] > 1 for s in specs)
                partners = ([("pass", 1 - rb_share), ("rb", rb_share)] if stacked
                            else [("pass", 1.0)])
                for partner, pw in partners:
                    if pw <= 0:
                        continue
                    for pos, fw in flex_units:
                        batch_plan.append({
                            "specs": with_partner(specs, partner),
                            "flex": pos,
                            "weight": (qw / qsum) * (fw / fsum) * pw,
                            "label": " / ".join(filter(None, [
                                (f"QB + {spec['qb_side'] - 1} + {spec['bring_back']} back"
                                 if spec else None),
                                ("QB + RB" if partner == "rb" else None),
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
                empty = [b["label"] for b in batch_plan if b["count"] < 1]
                if empty:
                    # Small shares (e.g. 5% QB + RB x 22% TE flex = 1.1%) round to
                    # zero at normal lineup counts. That's expected, not a problem —
                    # building more lineups than you'll enter just to fill them
                    # would be worse.
                    st.caption(
                        f"At {num_lineups} lineups, {len(empty)} small combination"
                        f"{'s' if len(empty) != 1 else ''} round to 0 lineups "
                        f"({'; '.join(empty)}) — their shares are too small to fill a whole "
                        "lineup at this count. The rest are built as normal.")
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

        # A combo can only be built in batches whose forced flex leaves room for
        # it, so its whole target has to come out of those batches.
        for row in player_combos:
            room = sum(b["count"] for b in batch_plan
                       if combo_fits(row["players"], b.get("flex")))
            label = " + ".join(row["players"])
            need = int((row["min"] or 0) / 100 * num_lineups + 0.5)
            if room < num_lineups:
                flexes = sorted({b["flex"] for b in batch_plan
                                 if b.get("flex") and combo_fits(row["players"], b["flex"])})
                st.caption(f"{label} only fits when the flex is "
                           f"{' or '.join(flexes) or 'free'}, so it's built in those "
                           f"{room} lineups.")
            if need > room:
                st.warning(f"{label}: min {row['min']:.0f}% needs {need} lineups but only "
                           f"{room} have a flex that fits both — it will fall short.")
            elif room < num_lineups and need:
                # Exposure caps apply inside each batch, so a player capped at
                # 50% can fill at most half of the batches the combo fits in.
                cap = min(max_exp.get(n, global_max_exp) for n in row["players"])
                if need > room * cap / 100 + 0.5:
                    st.warning(
                        f"{label}: min {row['min']:.0f}% needs {need} of those {room} "
                        f"lineups ({need / room:.0%}), but a {cap:.0f}% max exposure is "
                        f"applied inside each batch, so it tops out near "
                        f"{int(room * cap / 100 + 0.5)}. Raise the flex share or those "
                        "players' Max Exp %.")

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
                if is_showdown:
                    # One cap per real player, not per slot copy.
                    combined = {}
                    for nm in by_name:
                        pct = max_exp.get(nm, global_max_exp if global_max_exp < 100 else None)
                        if pct:
                            combined[nm] = pct / 100
                    opt.combined_exposure = combined
                    opt.add_new_rule(CombinedExposureRule)
                    for nm, pct in min_exp.items():
                        for pl in by_name.get(nm, ()):
                            pl.min_exposure = pct / 100
                else:
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
                    partner=spec.get("partner", "pass"),
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


            def make_optimizer(specs, solver, report, flex=None, captain=None, combos=None,
                               shares=None, count=None):
                opt = configure_optimizer(report=report, solver=solver)
                if shares is not None:
                    apply_batch_shares(opt, shares, count)
                if combos:
                    opt.player_combos = combos
                    opt.add_new_rule(PlayerComboRule)
                if captain:
                    # Every player sits in the pool twice. Dropping the CPT copy
                    # of anyone at another position leaves only this position
                    # able to fill the captain slot.
                    for pl in list(opt.player_pool.all_players):
                        if "CPT" in pl.positions and real_by_id.get(str(pl.id)) != captain:
                            opt.player_pool.remove_player(pl)
                if flex:
                    # An extra required player at a position pushes one into the
                    # flex slot: the roster needs 3 WRs, so asking for 4 puts a
                    # WR there.
                    opt.set_players_with_same_position({flex: 1})
                for spec in specs:
                    add_qb_stack(opt, spec, report=report)
                return opt

            def run_batch(specs, count, report=True, exclude=(), flex=None, captain=None,
                          idx=None):
                """Solve on the fast engine, fall back to the library default.

                The fallback exists for the error message: PuLP names the
                constraints it couldn't satisfy, which the diagnostics rely on.
                `exclude` carries lineups already built in earlier batches —
                each batch is its own optimizer run, so without it the same
                roster can come back twice.
                """
                combos = combo_targets(exclude, count, idx)
                shares = exposure_shares(exclude, count) if idx is not None else None

                def attempt(solver, report_):
                    opt = make_optimizer(specs, solver, report_, flex, captain, combos,
                                         shares, count)
                    kwargs = dict(optimize_kwargs)
                    if shares is not None:
                        # Caps are now this batch's own counts, which only mean
                        # what they say when measured against the batch total.
                        kwargs.pop("max_exposure", None)
                        kwargs["exposure_strategy"] = TotalExposureStrategy
                    if exclude:
                        # Batches can have different pools — forcing a captain
                        # position drops the other CPT copies. The library looks
                        # excluded lineups up in its player map without guarding,
                        # so anything it can no longer see has to be dropped here.
                        have = {pl.id for pl in opt.player_pool.filtered_players}
                        keep = [lu for lu in exclude
                                if all(pl.id in have for pl in lu.players)]
                        if keep:
                            kwargs["exclude_lineups"] = keep
                    return list(opt.optimize(count, **kwargs))

                if FAST_SOLVER is not None:
                    try:
                        return attempt(FAST_SOLVER, report)
                    except LineupOptimizerException:
                        pass
                return attempt(None, report)

            def exposure_shares(prior, count):
                """Each player's min and max lineup counts for one batch of a mix.

                The library applies exposure inside each optimizer run, and a run
                of 4 lineups can't hold anyone under 25% — one appearance already
                is 1 in 4. Across nine small batches a 10% cap came out at 13%.
                So the caps are converted to counts on a cumulative schedule: by
                the end of this batch a player may have at most ceil(cap x lineups
                so far), minus what earlier batches already used. Mins follow the
                same schedule, so they never ask for more than the max allows.
                """
                upto = len(prior) + count
                used = Counter(pl.full_name for lu in prior for pl in lu.players)
                caps, mins = {}, {}
                for nm in merged_df["Name"]:
                    if nm in locked_names:
                        continue
                    pct = max_exp.get(nm, global_max_exp if global_max_exp < 100 else None)
                    if pct is not None:
                        caps[nm] = max(-(-(pct / 100 * upto - 1e-9) // 1) - used[nm], 0)
                    if min_exp.get(nm):
                        need = int(min_exp[nm] / 100 * upto + 0.5) - used[nm]
                        mins[nm] = min(max(need, 0), count, caps.get(nm, count))
                return {"max": caps, "min": mins}

            def apply_batch_shares(opt, shares, count):
                """Swap the library's percentage caps for this batch's counts."""
                by_name = defaultdict(list)
                for pl in opt.player_pool.all_players:
                    by_name[pl.full_name].append(pl)
                combined = {}
                for nm, copies in by_name.items():
                    allowed = shares["max"].get(nm)
                    need = shares["min"].get(nm)
                    if allowed == 0:
                        # A cap of 0 reads as "no cap" to the library, so a
                        # player with nothing left is taken out of the batch.
                        for pl in copies:
                            opt.player_pool.remove_player(pl)
                        continue
                    for pl in copies:
                        pl.max_exposure = (None if is_showdown or allowed is None
                                           else allowed / count)
                        pl.min_exposure = need / count if need else None
                    if is_showdown and allowed is not None:
                        combined[nm] = allowed / count
                if is_showdown:
                    opt.combined_exposure = combined

            def combo_targets(prior, count, idx=None):
                """Turn each combo's percentages into lineup counts for one run.

                Targets are cumulative: a batch gets whatever the run so far still
                owes, so the overall share holds even when the lineups are split
                into small batches that would each round badly on their own. The
                schedule only advances through batches whose flex can hold the
                combo — the rest owe nothing, and the ones that can make it up.
                """
                plan = batch_plan if idx is not None else [{"count": count, "flex": None}]
                at = idx or 0
                out = []
                for row in player_combos:
                    wanted = set(row["players"])
                    done = sum(1 for lu in prior
                               if wanted <= {pl.full_name for pl in lu.players})
                    fits = [combo_fits(row["players"], b.get("flex")) for b in plan]
                    room = sum(b["count"] for b, f in zip(plan, fits) if f) or 1
                    upto = sum(b["count"] for b, f in zip(plan[:at + 1], fits) if f)
                    total_n = sum(b["count"] for b in plan)

                    def owed(pct):
                        if not fits[at]:
                            return 0
                        return min(max(int(pct / 100 * total_n * upto / room + 0.5)
                                       - done, 0), count)
                    out.append({
                        "players": row["players"],
                        "min_count": owed(row["min"]) if row["min"] else None,
                        "max_count": owed(row["max"]) if row["max"] is not None else None,
                    })
                return out

            optimize_kwargs = {"exposure_strategy": strategy_map[strategy_label]}
            if global_max_exp < 100 and not is_showdown:
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
                    "Player combos": [(r["players"], r["min"], r["max"])
                                      for r in player_combos],
                }
                with st.expander("Diagnostics — the exact settings in play", expanded=True):
                    st.code("\n".join(f"{k}: {v}" for k, v in info.items()), language="text")

            trouble = (
                "The rules above contradict each other. Most common causes:\n"
                "- Same-team restrictions that leave no valid roster\n"
                "- A QB stack shape larger than the per-team limits in section 4 allow\n"
                "- Min exposure totals that exceed the roster, or too few players left "
                "after excludes\n"
                "- A player combo whose min is above one of its players' max exposure, "
                "or that can't fit under the per-team limits\n\n"
                "The reported constraint names come from the solver and often point at "
                "knock-on effects rather than the rule you actually set."
            )

            run_started = time.time()
            # Simulation builds sim_mult candidates per lineup, batch by batch, and
            # picks the final set from them afterwards.
            build_n = num_lineups * sim_mult if sim_on else num_lineups
            for b in batch_plan:
                b["keep"] = b["count"]
                if sim_on:
                    b["count"] *= sim_mult
            batch_of = []
            progress = st.status(
                f"Building {build_n:,} lineups to keep the best {num_lineups}…" if sim_on
                else f"Building {num_lineups} lineups…", expanded=True)

            if mix_mode:
                # One optimizer run per (stack shape x flex position) combination.
                # This batching is the app's, not the library's.
                for idx, b in enumerate(batch_plan):
                    progress.write(f"{idx + 1} of {len(batch_plan)} — {b['label']} "
                                   f"({b['count']} lineup{'s' if b['count'] != 1 else ''})")
                    if b["count"] < 1:
                        mix_report.append({"Rule": b["label"], "Share %":
                                           round(b["weight"] * 100, 1), "Lineups": 0})
                        continue
                    try:
                        batch = run_batch(b["specs"], b["count"], report=(idx == 0),
                                          exclude=lineups, flex=b["flex"],
                                          captain=b.get("captain"), idx=idx)
                    except LineupOptimizerException as exc:
                        st.error(f"**\"{b['label']}\" failed:** {exc}\n\n" + trouble)
                        show_diagnostics()
                        st.stop()
                    lineups.extend(batch)
                    batch_of.extend([idx] * len(batch))
                    mix_report.append({"Rule": b["label"], "Share %":
                                       round(b["weight"] * 100, 1), "Lineups": len(batch)})
            else:
                progress.write(f"solving {build_n} lineups…")
                try:
                    lineups = run_batch(qb_stack_specs, build_n)
                except LineupOptimizerException as exc:
                    st.error(f"**Couldn't generate lineups:** {exc}\n\n" + trouble)
                    show_diagnostics()
                    st.stop()
                batch_of = [0] * len(lineups)
                batch_plan[0]["keep"] = num_lineups

            sim_df = sim_note = None
            if sim_on and lineups:
                progress.write(f"simulating the slate {sim_runs:,} times…")
                pool_df = sim_pool(merged_df, players)
                row_of = {n: i for i, n in enumerate(pool_df["Name"])}
                sims = sim.simulate(pool_df, sim_runs)
                members = [[(row_of[p.full_name], 1.5 if p.lineup_position == "CPT" else 1.0)
                            for p in lu.players if p.full_name in row_of] for lu in lineups]
                scores = sim.score_lineups(members, sims)

                # The field: simulated opponents drawn from projected ownership —
                # Milly/small-field on a classic slate, CPT/FLEX on a showdown —
                # whenever the projections file carries it.
                has = lambda c: c in pool_df.columns and pool_df[c].notna().any()
                field_on = (has("CPT Ownership") and has("FLEX Ownership") if is_showdown
                            else has("Ownership"))
                thresh = fresh_thresh = share = None
                if field_on:
                    progress.write("simulating the field from projected ownership…")
                    # The field draws from salaried players; its row numbers are
                    # positions in that subset, mapped back to pool rows after.
                    salaried = np.flatnonzero(pool_df["Salary"].notna().to_numpy())
                    sub = pool_df.iloc[salaried].reset_index(drop=True)
                    if is_showdown:
                        cpt_own = pool_df["CPT Ownership"].fillna(0).to_numpy()
                        flex_own = pool_df["FLEX Ownership"].fillna(0).to_numpy()
                        f_cpt, f_flex, field_err = sim.build_showdown_field(
                            sub, sub["CPT Ownership"].to_numpy(),
                            sub["FLEX Ownership"].to_numpy(), n=FIELD_LINEUPS)
                        field_members = [[(salaried[c], 1.5)] + [(salaried[i], 1.0) for i in f]
                                         for c, f in zip(f_cpt, f_flex)]
                        # A captain's ownership is his CPT share, a flex player's
                        # his FLEX share.
                        slot_own = lambda i, mult: cpt_own[i] if mult > 1 else flex_own[i]
                    else:
                        pool_df["Own"] = blend_ownership(pool_df, field_entries)
                        own_arr = pool_df["Own"].fillna(0).to_numpy()
                        field, field_err = sim.build_field(
                            sub, pool_df["Own"].to_numpy()[salaried], n=FIELD_LINEUPS)
                        field_members = [[(salaried[i], 1.0) for i in row] for row in field]
                        slot_own = lambda i, mult: own_arr[i]
                    thresh = np.quantile(sim.score_lineups(field_members, sims),
                                         1 - sim_top, axis=0)
                    lineup_own = [[slot_own(i, mult) for i, mult in m] for m in members]
                    dupes = sim.expected_duplicates(lineup_own, field_entries)
                    share = 1 / (1 + dupes)

                names = [{p.full_name for p in lu.players} for lu in lineups]
                caps = {}
                for nm in merged_df["Name"]:
                    pct = max_exp.get(nm, global_max_exp if global_max_exp < 100 else None)
                    if pct is not None and nm not in locked_names:
                        caps[nm] = -(-(pct / 100 * num_lineups - 1e-9) // 1)
                batches = [([i for i, bi in enumerate(batch_of) if bi == idx], b["keep"])
                           for idx, b in enumerate(batch_plan) if b.get("keep")]
                chosen, info = sim.select(scores, batches, names, top_q=sim_top, caps=caps,
                                          thresh=thresh, share=share)
                # Listed best first: each lineup's own top-% rate, discounted for
                # the duplicates that would split it. Same lineups, new order.
                alone = info["rate"] * (1 if share is None else share)
                chosen = sorted(chosen, key=lambda c: -alone[c])

                # Graded on a fresh set of sims, against what the optimizer alone
                # would have kept (each batch's first lineups). Grading on the sims
                # the picks were chosen on would flatter them.
                base = [c for cands, keep in batches for c in cands[:keep]]
                fresh_sims = sim.simulate(pool_df, sim_runs)
                fresh = sim.score_lineups(members, fresh_sims)
                if field_on:
                    fresh_thresh = np.quantile(sim.score_lineups(field_members, fresh_sims),
                                               1 - sim_top, axis=0)
                sim_note = {
                    "candidates": len(lineups), "runs": sim_runs, "top": sim_top,
                    "covered": sim.coverage(fresh, chosen, sim_top, fresh_thresh),
                    "baseline": sim.coverage(fresh, base, sim_top, fresh_thresh),
                    "over_cap": info["over_cap"],
                    "field": ({"lineups": len(field_members), "entries": field_entries,
                               "error": field_err,
                               "which": ("TwoGunGPT's CPT/FLEX ownership" if is_showdown
                                         else ownership_label(field_entries))}
                              if field_on else None),
                }
                sel = scores[chosen].astype(float)
                sim_df = pd.DataFrame({
                    "Lineup": range(1, len(chosen) + 1),
                    "Optimizer pts": [round(lineups[c].fantasy_points_projection, 1)
                                   for c in chosen],
                    "Sim avg": sel.mean(axis=1).round(1),
                    "Sim 90th": np.quantile(sel, 0.9, axis=1).round(1),
                    "Sim 99th": np.quantile(sel, 0.99, axis=1).round(1),
                    f"Top {sim_top:.1%} rate": (info["rate"][chosen] * 100).round(2),
                })
                if field_on:
                    sim_df["Total own %"] = [round(sum(lineup_own[c]), 1) for c in chosen]
                    sim_df["Exp. duplicates"] = dupes[chosen].round(2)
                for i in range(len(mix_report)):
                    mix_report[i]["Lineups"] = sum(1 for c in chosen if batch_of[c] == i)
                lineups = [lineups[c] for c in chosen]

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

            elapsed = time.time() - run_started
            progress.update(label=(f"Kept the best {len(lineups)} of {build_n:,} lineups "
                                   f"in {elapsed:.1f}s" if sim_df is not None
                                   else f"Built {len(lineups)} lineups in {elapsed:.1f}s"),
                            state="complete", expanded=False)

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
            stored = st.session_state.get("optimal")
            if stored and stored["key"] == opt_key:
                exposure_df["Optimal %"] = exposure_df["Name"].map(stored["opt"])
            if "Ownership" in merged_df.columns:
                own_now = (pd.to_numeric(merged_df["Ownership"], errors="coerce")
                           if is_showdown else blend_ownership(merged_df, field_entries))
                exposure_df["Proj own %"] = exposure_df["Name"].map(
                    dict(zip(merged_df["Name"], own_now.round(1))))
            exposure_df["Min set %"] = exposure_df["Name"].map(min_exp)
            exposure_df["Max set %"] = exposure_df["Name"].map(max_exp)
            exposure_df = exposure_df.sort_values("Lineups", ascending=False)

            # Showdown: the same player split by the slot they filled. The 1.5x
            # captain is the biggest single call, so it gets its own breakdown.
            captain_df = None
            if is_showdown:
                slot = picks["Position"].eq("CPT").map({True: "CPT", False: "FLEX"})
                captain_df = (pd.crosstab([picks["Name"], picks["Team"]], slot)
                              .reindex(columns=["CPT", "FLEX"], fill_value=0)
                              .rename_axis(columns=None)
                              .reset_index())
                captain_df.insert(1, "Position", captain_df["Name"].map(real_pos))
                captain_df["CPT %"] = (captain_df["CPT"] / total * 100).round(1)
                captain_df["FLEX %"] = (captain_df["FLEX"] / total * 100).round(1)
                captain_df["Total %"] = captain_df["CPT %"] + captain_df["FLEX %"]
                captain_df = (captain_df[["Name", "Position", "Team", "CPT", "CPT %",
                                          "FLEX", "FLEX %", "Total %"]]
                              .sort_values(["CPT", "Total %"], ascending=False))

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

            # How often each combo actually came out, against what was asked.
            combo_df = None
            if player_combos:
                lineup_sets = [set(g["Name"]) for _, g in picks.groupby("Lineup")]
                combo_df = pd.DataFrame([{
                    "Combo": " + ".join(r["players"]),
                    "Lineups": sum(1 for names in lineup_sets
                                   if set(r["players"]) <= names),
                    "Min set %": r["min"],
                    "Max set %": r["max"],
                } for r in player_combos])
                combo_df.insert(2, "Exposure %",
                                (combo_df["Lineups"] / total * 100).round(1))

            mix_df = None
            if mix_report:
                mix_df = pd.DataFrame(mix_report)
                mix_df["Actual %"] = (mix_df["Lineups"] / max(len(lineups), 1) * 100).round(1)

            st.session_state.results = {
                "built_at": time.strftime("%H:%M:%S"),
                "took": elapsed,
                "lineups": result_df,
                "exposure": exposure_df,
                "mix": mix_df,
                "upload": upload_df,
                "teams": team_df,
                "flex": flex_df,
                "captains": captain_df,
                "combos": combo_df,
                "sim": sim_df,
                "sim_note": sim_note,
            }

        results = st.session_state.get("results")
        if results is not None:
            st.subheader("Generated Lineups")
            if results.get("built_at"):
                st.caption(f"Built at **{results['built_at']}** in "
                           f"{results['took']:.1f}s. Generating again gives a different set "
                           "whenever randomness is on.")

            # Player pool actually used: distinct names across every lineup.
            used = results["lineups"]
            used = used.loc[used["Position"] != "TOTAL", "Name"].drop_duplicates()
            pos_of = dict(zip(merged_df["Name"], merged_df["Position"]))
            # First listed position only, so a WR/RB isn't counted twice.
            used_pos = Counter(str(pos_of.get(n, "?")).split("/")[0] for n in used)
            with st.container(horizontal=True):
                st.metric("Lineups", int(results["lineups"]["Lineup"].max()),
                          border=True, width="content")
                st.metric("Player pool", len(used), border=True, width="content",
                          help="Unique players across all lineups.")
                for pos in order_positions(used_pos, sport):
                    st.metric(pos, used_pos[pos], border=True, width="content")

            st.dataframe(results["lineups"], width="stretch")

            if results.get("sim") is not None:
                note = results["sim_note"]
                st.subheader("Simulation")
                with st.container(horizontal=True):
                    st.metric(f"Any lineup in top {note['top']:.1%}", f"{note['covered']:.0%}",
                              delta=f"{(note['covered'] - note['baseline']) * 100:+.0f} pts "
                                    "vs optimizer order",
                              help="Share of simulated slates in which at least one of your "
                                   "lineups finishes in the target slice (of the simulated "
                                   "field when there is one, else of all lineups built). "
                                   "The comparison is the same number of lineups taken in "
                                   "the order the optimizer built them. Both are graded on "
                                   "a second, independent set of sims.")
                    st.metric("Lineups compared", f"{note['candidates']:,}")
                    st.metric("Simulations", f"{note['runs']:,}")
                if note.get("field"):
                    fld = note["field"]
                    st.caption(
                        f"Judged against **{fld['lineups']:,} simulated opponents** built from "
                        f"{fld['which']} ({fld['entries']:,}-entry contest; their player "
                        f"exposures match it to within {fld['error']:.1f} pts). A hit is a "
                        f"lineup scoring in the field's top {note['top']:.1%}, and it counts "
                        "less the more copies of it the field is expected to enter.")
                else:
                    st.caption("No ownership in the projections file, so lineups are judged "
                               "against each other rather than a simulated field.")
                if note.get("over_cap"):
                    over = ", ".join(f"{n} ({k})" for n, k in note["over_cap"].items())
                    st.warning(f"No set of the lineups built kept everyone under their exposure cap, "
                               f"so these players are over it (lineups shown): {over}. Loosen "
                               "their caps or build and compare more lineups.")
                st.caption("Per lineup: its simulated average, 90th and 99th percentile "
                           "scores, and how often it lands in the target slice.")
                st.dataframe(results["sim"], width="content", hide_index=True)

            if results["mix"] is not None:
                st.subheader("Stack Mix")
                st.dataframe(results["mix"], width="content", hide_index=True)

            if results.get("upload") is not None:
                with st.container(horizontal=True):
                    st.download_button(
                        "⬇️ Download DK upload CSV",
                        results["upload"].to_csv(index=False),
                        "dk_upload.csv", "text/csv", type="primary",
                        help="One row per lineup, one column per roster slot, "
                             "with the DK player IDs — ready to upload.")
                    st.link_button(
                        "Upload lineups to DraftKings", "https://www.draftkings.com/lineup/upload",
                        icon=":material/open_in_new:",
                        help="DraftKings' lineup upload page — choose the dk_upload.csv "
                             "you just downloaded.")
                    st.link_button(
                        "My Contests", "https://www.draftkings.com/mycontests",
                        icon=":material/open_in_new:",
                        help="After the games: open a finished contest and export the "
                             "standings CSV, so the sim's ownership and duplicate models "
                             "can be checked against what really happened.")

            if results.get("combos") is not None:
                st.subheader("Player Combos")
                st.dataframe(results["combos"], width="content", hide_index=True)

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

            if is_showdown and results.get("captains") is not None:
                st.subheader("Captain Exposure")
                caps_df = results["captains"]
                st.caption(
                    f"{int((caps_df['CPT'] > 0).sum())} different captains across "
                    f"{int(results['lineups']['Lineup'].max())} lineups. **CPT** and **FLEX** count the "
                    "lineups where the player filled that slot; a player is never both "
                    "in one lineup, so **Total %** is their overall exposure.")
                st.dataframe(caps_df, width="stretch", hide_index=True,
                             column_config={
                                 "CPT %": st.column_config.ProgressColumn(
                                     "CPT %", format="%.1f%%", min_value=0, max_value=100),
                             })

            st.subheader("Common Combos")
            st.caption("Groups of players that land in the same lineup most often, "
                       "whichever slots they fill.")
            lineup_df = results["lineups"]
            n_built = int(lineup_df["Lineup"].max())
            with st.container(horizontal=True, vertical_alignment="bottom"):
                combo_size = st.segmented_control(
                    "Group size", [2, 3, 4], default=2, key="combo_size",
                    format_func=lambda k: {2: "Pairs", 3: "Trios", 4: "Foursomes"}[k])
                must_have = st.multiselect(
                    "Containing", sorted(lineup_df.loc[lineup_df["Position"] != "TOTAL",
                                                       "Name"].unique()),
                    key="combo_must_have", placeholder="Any player",
                    help="Only show groups that include all of these players.")
                min_seen = st.number_input(
                    "Seen in at least", min_value=1, max_value=max(n_built, 1), value=2,
                    key="combo_min_seen", help="Lineups. Hides one-off groups.")
            combo_size = combo_size or 2
            if len(must_have) > combo_size:
                st.caption(f"{len(must_have)} players can't all fit in a group of "
                           f"{combo_size}.")
            else:
                pos_of = dict(zip(merged_df["Name"], merged_df["Position"]))
                team_of = dict(zip(lineup_df["Name"], lineup_df["Team"]))
                wanted = set(must_have)
                rows = [{
                    "Players": " + ".join(group),
                    "Positions": " / ".join(str(pos_of.get(n, "?")) for n in group),
                    "Teams": " / ".join(str(team_of.get(n, "?")) for n in group),
                    "Lineups": seen,
                    "Exposure %": round(seen / n_built * 100, 1),
                } for group, seen in count_combos(lineup_df, combo_size).items()
                    if seen >= min_seen and wanted <= set(group)]
                if rows:
                    top = (pd.DataFrame(rows)
                           .sort_values(["Lineups", "Players"], ascending=[False, True])
                           .reset_index(drop=True))
                    st.caption(f"{len(top)} groups. Showing the most common first.")
                    st.dataframe(top, width="stretch", hide_index=True, height=420)
                else:
                    st.caption("No groups match.")

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

            if agent is not None:
                lineup_assistant(results, merged_df, content, is_showdown, proj_label,
                                 set(excluded_names), len(optimizer.settings.positions),
                                 optimizer.settings.budget, lobby_group_id)

    except Exception as e:
        st.error(f"Error processing file: {e}")
else:
    st.info("Upload a DK/FD salary CSV to get started.")
