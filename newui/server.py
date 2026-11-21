"""New UI backend: runs the real DFS app (app.py) headlessly and serves its state as JSON.

Every setting, default, rule, warning and lineup comes from app.py itself; this file only
translates between the new page and that headless session, so builds are the same code
path as the Streamlit app.

usage: venv/bin/python server.py [port]      (default 8510, http://localhost:8510)
env:   NEWUI_APP  folder holding app.py (default: the live NFL folder, read-only use)
"""
import base64
import io
import json
import math
import os
import sys
import threading
import time
import traceback

import numpy as np
import pandas as pd
import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, Response
from starlette.routing import Route

import streamlit.testing.v1.app_test as app_test_mod
from streamlit.proto.WidgetStates_pb2 import WidgetState
from streamlit.runtime.memory_media_file_storage import MemoryMediaFileStorage
from streamlit.testing.v1 import AppTest
from streamlit.testing.v1 import element_tree as et

APP_DIR = os.environ.get(
    "NEWUI_APP", os.path.expanduser("~/Library/CloudStorage/OneDrive-Comcast/JLab/NFL"))
HERE = os.path.dirname(os.path.abspath(__file__))

# Download buttons store their bytes in a media store that AppTest makes per run; keep
# the last one so the page can fetch a download by its URL.
_media = {"store": None}


class _KeptStorage(MemoryMediaFileStorage):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        _media["store"] = self


app_test_mod.MemoryMediaFileStorage = _KeptStorage

# Editable tables that the app backs with a list in session state. The page sends whole
# tables; they're written to those lists and the editor's nonce is bumped so it reloads.
TABLES = {
    "combo_editor_": ("combo_rows", "combo_nonce",
                      lambda r: {"players": list(r.get("Players") or []),
                                 "min": _num(r.get("Min %")), "max": _num(r.get("Max %"))}),
    "cpt_editor_": ("cpt_rows", "cpt_nonce",
                    lambda r: {"position": str(r.get("Position")), "share": _num(r.get("Share %"))}),
    "flex_editor_": ("flex_rows", "flex_nonce",
                     lambda r: {"position": str(r.get("Position")), "share": _num(r.get("Share %"))}),
    "same_team_editor_": ("same_team_pairs", "same_team_nonce",
                          lambda r: (str(r.get("Position A")), str(r.get("Position B")))),
    "qs_editor_": ("qs_rows", "qs_nonce",
                   lambda r: {"qb_side": int(_num(r.get("QB side")) or 1),
                              "bring_back": int(_num(r.get("Bring-back")) or 0),
                              "share": _num(r.get("Share %")),
                              "max_exp": (None if _num(r.get("Max Exp %")) in (None, 100.0)
                                          else _num(r.get("Max Exp %")))}),
}


def _num(v):
    if v is None or v == "":
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


def _clean(v):
    """JSON-safe scalar."""
    if v is None:
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        return None if (math.isnan(v) or math.isinf(v)) else float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, (list, tuple, set, np.ndarray)):
        return [_clean(x) for x in v]
    if isinstance(v, (pd.Timestamp,)):
        return str(v)
    if isinstance(v, dict):
        return {str(k): _clean(x) for k, x in v.items()}
    if isinstance(v, (str, int, bool)):
        return v
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    return str(v)


def _frame(df, limit=2000):
    df = df.reset_index(drop=True) if not isinstance(df.index, pd.RangeIndex) else df
    rows = [[_clean(x) for x in r] for r in df.head(limit).itertuples(index=False, name=None)]
    return {"cols": [str(c) for c in df.columns], "rows": rows, "total": int(len(df))}


NAME_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}
EDIT_COLS = {  # pool column -> header spellings a projections file may use
    "Projection": ("projection", "proj"), "Median": ("median",), "Ceiling": ("ceiling", "ceil"),
    "Nuke": ("nuke",), "Ownership": ("ownership", "own"),
    "Small Field Ownership": ("small field ownership",), "CPT Ownership": ("cpt ownership",),
    "FLEX Ownership": ("flex ownership",), "Minutes": ("minutes", "min"),
}
SHAPE = ("Median", "Ceiling", "Nuke")


def name_key(name):
    """Same loose match the app uses (normalize_name): no punctuation, case or Jr/III."""
    import re
    cleaned = re.sub(r"[^a-z0-9\s]", " ", str(name).lower())
    return " ".join(w for w in cleaned.split() if w not in NAME_SUFFIXES)


SAVE_DIR = os.environ.get("NEWUI_SAVE", os.path.join(HERE, "saved"))
SNAP_KEYS = ("locked_names", "excluded_names", "max_exp", "min_exp", "combo_rows", "cpt_rows",
             "flex_rows", "same_team_pairs", "qs_rows")
NONCES = ("editor_nonce", "combo_nonce", "cpt_nonce", "flex_nonce", "same_team_nonce", "qs_nonce")


class Session:
    def __init__(self):
        self.lock = threading.Lock()
        self.busy = None          # {"what": str, "since": t} while a long run is going
        self.error = None
        self.files = {}           # uploader id -> file name shown on the page
        self.cheat = None         # the projections file as uploaded, plus your edits to it
        self.replaying = False
        self.log = []             # setup steps in order (latest value per control)
        self.start()
        self.restore()

    def start(self):
        os.chdir(APP_DIR)
        self.app_mtime = os.path.getmtime(os.path.join(APP_DIR, "app.py"))
        self.at = AppTest.from_file(os.path.join(APP_DIR, "app.py"), default_timeout=7200)
        self.at.run()

    def refresh_app(self):
        """app.py changed on disk: load the new version, put the setup back and keep
        the built lineups, so changes to the app show up without losing work."""
        try:
            m = os.path.getmtime(os.path.join(APP_DIR, "app.py"))
        except OSError:
            return
        if m == self.app_mtime:
            return
        ss = self.at.session_state
        kept = ss["results"] if "results" in ss else None
        self.save()
        self.files, self.cheat, self.log = {}, None, []
        self.start()
        self.restore()
        if kept is not None:
            self.at.session_state["results"] = kept
            self.run()

    # ---------- running ----------
    def run(self, overrides=None):
        tree = self.at._tree
        ws = tree.get_widget_states()
        if overrides:
            for w in ws.widgets:
                if w.id in overrides:
                    w.CopyFrom(overrides[w.id])
        self.at._run(ws)

    def find(self, wid):
        for node in self.at._tree:
            if isinstance(node, et.Widget) and node.id == wid:
                return node
        raise KeyError(f"control not found (the page may be out of date): {wid}")

    # ---------- actions ----------
    def act(self, a):
        """Do a page action, then note it so a restart can put the setup back."""
        entry = self._note(a)
        self._act(a)
        if entry is not None and not self.replaying:
            self._remember(entry)

    def _act(self, a):
        op = a.get("op")
        if op == "set":
            w = self.find(a["id"])
            v = a.get("value")
            ws = WidgetState()
            ws.id = w.id
            if isinstance(w, (et.Selectbox, et.Radio)):
                if v is None:
                    self.run({w.id: ws})
                    return
                ws.string_value = w.options[int(v)]
                self.run({w.id: ws})
            elif isinstance(w, (et.Multiselect, et.ButtonGroup)):
                idx = [] if v is None else ([int(v)] if isinstance(v, (int, float)) else [int(i) for i in v])
                ws.string_array_value.data[:] = [w.options[i] for i in idx]
                self.run({w.id: ws})
            elif isinstance(w, et.SelectSlider):
                ws.string_array_value.data[:] = [w.options[int(v)]]
                self.run({w.id: ws})
            elif isinstance(w, et.NumberInput):
                if v in (None, ""):
                    w.set_value(None)
                else:
                    f = float(v)
                    w.set_value(int(f) if (w.proto.data_type == w.proto.INT) else f)
                self.run()
            elif isinstance(w, et.Slider):
                w.set_value(type(w.value)(v) if not isinstance(w.value, (list, tuple)) else v)
                self.run()
            elif isinstance(w, (et.Checkbox, et.Toggle)):
                w.set_value(bool(v))
                self.run()
            elif isinstance(w, (et.TextInput, et.TextArea)):
                w.set_value(v or "")
                self.run()
            elif type(w).__name__ == "TimeInput":
                import datetime as _dt
                w.set_value(_dt.time.fromisoformat(v) if v else None)
                self.run()
            elif type(w).__name__ == "DateInput":
                import datetime as _dt
                w.set_value(_dt.date.fromisoformat(v) if v else None)
                self.run()
            elif isinstance(w, et.ChatInput):
                w.set_value(v)
                self.run()
            else:
                w.set_value(v)
                self.run()
        elif op == "click":
            w = self.find(a["id"])
            w.click() if hasattr(w, "click") else w.set_value(True)
            self.run()
        elif op == "upload":
            w = self.find(a["id"])
            if a.get("clear"):
                w.set_value(None)
                self.files.pop(a["id"], None)
            else:
                data = base64.b64decode(a["b64"])
                w.set_value((a["name"], data, a.get("mime") or "text/csv"))
                self.files[a["id"]] = a["name"]
            if "projection" in w.label.lower():
                self.cheat = None if a.get("clear") else self._read_cheat(a["id"], a["name"], data)
            self.run()
        elif op == "table":
            key = a["key"]
            for prefix, (rows_key, nonce_key, conv) in TABLES.items():
                if key.startswith(prefix):
                    rows = [conv(r) for r in a["rows"]]
                    self.at.session_state[rows_key] = rows
                    self.at.session_state[nonce_key] = self.at.session_state[nonce_key] + 1
                    self.run()
                    return
            raise KeyError(f"unknown table {key}")
        elif op == "pool":
            ss = self.at.session_state
            name, field, v = a["name"], a["field"], a.get("value")
            if field in ("lock", "exclude"):
                k = "locked_names" if field == "lock" else "excluded_names"
                s = set(ss[k])
                (s.add if v else s.discard)(name)
                ss[k] = s
            else:
                k = "max_exp" if field == "max" else "min_exp"
                d = dict(ss[k])
                f = _num(v)
                if f is None:
                    d.pop(name, None)
                else:
                    d[name] = f
                ss[k] = d
            ss["editor_nonce"] = ss["editor_nonce"] + 1
            self.run()
        elif op == "proj":
            self.edit_projection(a.get("name"), a.get("col"), a.get("value"), a.get("reset"))
        elif op == "restart":
            self.files = {}
            self.cheat = None
            self.log = []
            self.save()
            self.start()
        else:
            raise ValueError(f"unknown action {op}")

    # ---------- remembering the setup across restarts ----------
    def _where(self, w):
        for node in self.at._tree.sidebar:
            if node is w:
                return "sidebar"
        return "main"

    def _note(self, a):
        """What to record for an action: the control by type/label/key (ids can change
        between runs) and the chosen option text rather than its position."""
        op = a.get("op")
        if op == "set":
            try:
                w = self.find(a["id"])
            except KeyError:
                return None
            sel = {"w": type(w).__name__, "label": getattr(w, "label", ""),
                   "key": w.id.split("-", 2)[-1] if w.id.startswith("$$ID") else None,
                   "where": self._where(w)}
            v = a.get("value")
            if isinstance(w, (et.Selectbox, et.Radio, et.SelectSlider)):
                return {"op": "set", "sel": sel, "opt": None if v is None else w.options[int(v)]}
            if isinstance(w, (et.Multiselect, et.ButtonGroup)):
                idx = [] if v is None else ([int(v)] if isinstance(v, (int, float)) else [int(i) for i in v])
                return {"op": "set", "sel": sel, "opts": [w.options[i] for i in idx]}
            if isinstance(w, (et.TextInput, et.TextArea)) and getattr(w.proto, "type", 0) == 1:
                return None  # never write API keys or passwords to disk
            if isinstance(w, et.ChatInput):
                return None
            return {"op": "set", "sel": sel, "value": v}
        if op == "upload":
            try:
                w = self.find(a["id"])
            except KeyError:
                return None
            sel = {"w": "FileUploader", "label": w.label, "key": None, "where": self._where(w)}
            if a.get("clear"):
                return {"op": "upload", "sel": sel, "clear": True}
            os.makedirs(os.path.join(SAVE_DIR, "files"), exist_ok=True)
            import re
            path = os.path.join(SAVE_DIR, "files", re.sub(r"[^a-z0-9]+", "_", w.label.lower()).strip("_") + ".csv")
            with open(path, "wb") as fh:
                fh.write(base64.b64decode(a["b64"]))
            return {"op": "upload", "sel": sel, "name": a["name"], "mime": a.get("mime"), "path": path}
        return None  # clicks, pool/table/projection edits are saved as a snapshot instead

    def _remember(self, entry):
        ident = json.dumps(entry["sel"], sort_keys=True)
        # One slate/contest pick is enough: keys differ per slate and sport, and old
        # picks can't be put back once you've moved on.
        same = (lambda e: e["sel"]["label"] == entry["sel"]["label"]) if entry["sel"]["label"] in ("Slate", "Contest") else (lambda e: False)
        self.log = [e for e in self.log
                    if json.dumps(e["sel"], sort_keys=True) != ident and not same(e)] + [entry]

    def save(self):
        if self.replaying:
            return
        ss = self.at.session_state
        snap = {}
        for k in SNAP_KEYS:
            if k in ss:
                v = ss[k]
                snap[k] = sorted(v) if isinstance(v, set) else v
        data = {"saved": time.strftime("%Y-%m-%d %H:%M:%S"), "log": self.log, "snap": snap,
                "edits": self.cheat["edits"] if self.cheat else {}}
        os.makedirs(SAVE_DIR, exist_ok=True)
        tmp = os.path.join(SAVE_DIR, "setup.json.tmp")
        with open(tmp, "w") as fh:
            json.dump(_clean(data), fh)
        os.replace(tmp, os.path.join(SAVE_DIR, "setup.json"))

    def _find_sel(self, sel):
        block = self.at._tree.sidebar if sel.get("where") == "sidebar" else self.at._tree.main
        for node in block:
            if (isinstance(node, et.Widget) and type(node).__name__ == sel["w"]
                    and getattr(node, "label", "") == sel["label"]
                    and (sel.get("key") is None or node.id.endswith("-" + sel["key"]))):
                return node
        return None

    def restore(self):
        path = os.path.join(SAVE_DIR, "setup.json")
        if not os.path.exists(path):
            return
        try:
            with open(path) as fh:
                data = json.load(fh)
        except Exception:
            return
        self.replaying = True
        skipped = []
        norm = lambda o: str(o).replace(" · filled", "").replace(" · started", "").strip()
        try:
            for e in data.get("log", []):
                try:
                    w = self._find_sel(e["sel"])
                    if w is None:
                        skipped.append(e["sel"]["label"])
                        continue
                    if e["op"] == "upload":
                        if e.get("clear"):
                            self._act({"op": "upload", "id": w.id, "clear": True})
                        elif os.path.exists(e["path"]):
                            with open(e["path"], "rb") as fh:
                                b64 = base64.b64encode(fh.read()).decode()
                            self._act({"op": "upload", "id": w.id, "name": e["name"], "mime": e.get("mime"), "b64": b64})
                    elif "opt" in e:
                        opts = [norm(o) for o in w.options]
                        if e["opt"] is None:
                            self._act({"op": "set", "id": w.id, "value": None})
                        elif norm(e["opt"]) in opts:
                            self._act({"op": "set", "id": w.id, "value": opts.index(norm(e["opt"]))})
                        else:
                            skipped.append(e["sel"]["label"])
                            continue
                    elif "opts" in e:
                        opts = [norm(o) for o in w.options]
                        idx = [opts.index(norm(o)) for o in e["opts"] if norm(o) in opts]
                        self._act({"op": "set", "id": w.id, "value": idx})
                    else:
                        self._act({"op": "set", "id": w.id, "value": e.get("value")})
                    self.log.append(e)
                except Exception:
                    skipped.append(e["sel"].get("label", "?"))
            ss = self.at.session_state
            for k, v in data.get("snap", {}).items():
                ss[k] = set(v) if k in ("locked_names", "excluded_names") else (
                    [tuple(p) for p in v] if k == "same_team_pairs" else v)
            for nk in NONCES:
                if nk in ss:
                    ss[nk] = ss[nk] + 1
            self.run()
            if self.cheat and data.get("edits"):
                self.cheat["edits"] = data["edits"]
                self._write_cheat()
        finally:
            self.replaying = False
        # Only mention what is still missing: a later step for the same control may
        # have put it back (e.g. a contest from another sport, then today's contest).
        done = {e["sel"]["label"] for e in self.log}
        self.restored = {"saved": data.get("saved"), "skipped": [k for k in skipped if k not in done]}

    # ---------- projection edits ----------
    def _read_cheat(self, uid, name, data):
        try:
            df = pd.read_csv(io.BytesIO(data))
        except Exception:
            return None
        low = {str(c).strip().lower(): c for c in df.columns}
        name_col = low.get("name") or low.get("player")
        if name_col is None:
            return None
        cols = {pc: low[a] for pc, al in EDIT_COLS.items() for a in al if a in low}
        return {"id": uid, "name": name, "df": df, "name_col": name_col, "cols": cols,
                "keys": df[name_col].map(name_key), "edits": {}}

    def edit_projection(self, name, col, value, reset=False):
        c = self.cheat
        if not c:
            raise ValueError("Upload the projections cheatsheet first; edits are made to that file.")
        if reset:
            c["edits"] = {}
        else:
            if col not in c["cols"]:
                raise ValueError(f"{col} isn't in your projections file, so it can't be edited.")
            key = name_key(name)
            rows = c["keys"] == key
            if not rows.any():
                raise ValueError(f"{name} isn't in your projections file.")
            orig = lambda pc: _num(c["df"].loc[rows, c["cols"][pc]].iloc[0])
            v = _num(value)
            ed = c["edits"].setdefault(key, {})
            if v is None:  # blank = back to the file's number
                ed.pop(col, None)
                if col == "Projection":
                    for sc in SHAPE:
                        ed.pop(sc, None)
            else:
                ed[col] = v
                # A new projection moves the player's whole range with it, so the build
                # (whatever Optimize on says) and the simulation both see the change.
                o = orig("Projection") if col == "Projection" else None
                if col == "Projection" and o:
                    for sc in SHAPE:
                        if sc in c["cols"] and orig(sc) is not None:
                            ed[sc] = round(orig(sc) * v / o, 2)
            if not ed:
                c["edits"].pop(key, None)
        self._write_cheat()

    def _write_cheat(self):
        c = self.cheat
        df = c["df"].copy()
        for key, ed in c["edits"].items():
            rows = c["keys"] == key
            for pc, v in ed.items():
                df.loc[rows, c["cols"][pc]] = v
        w = self.find(c["id"])
        w.set_value((c["name"], df.to_csv(index=False).encode(), "text/csv"))
        self.run()

    def proj_info(self):
        c = self.cheat
        if not c:
            return None
        orig = {}
        for key, ed in c["edits"].items():
            rows = c["keys"] == key
            orig[key] = {pc: _num(c["df"].loc[rows, c["cols"][pc]].iloc[0]) for pc in ed}
        return {"cols": list(c["cols"]), "edits": c["edits"], "orig": orig}

    # ---------- serializing ----------
    def state(self):
        tree = self.at._tree
        out = {"busy": self.busy, "error": self.error, "files": self.files, "proj": self.proj_info(),
               "restored": getattr(self, "restored", None),
               "main": self.nodes(tree.main), "sidebar": self.nodes(tree.sidebar)}
        res = self.at.session_state["results"] if "results" in self.at.session_state else None
        out["has_results"] = res is not None
        out["exceptions"] = [e.value for e in tree.exception] if len(tree.exception) else []
        return out

    def nodes(self, block):
        return [n for c in block.children.values() if (n := self.node(c)) is not None]

    def node(self, c):
        try:
            return self._node(c)
        except Exception as exc:  # never let one odd element break the page
            return {"t": "alert", "kind": "warning", "text": f"(couldn't show a {type(c).__name__}: {exc})"}

    def _node(self, c):
        name = type(c).__name__
        if isinstance(c, et.Block):
            d = {"t": "block", "kind": c.type, "children": self.nodes(c)}
            p = c.proto
            if isinstance(c, (et.Expander, et.Status)):
                d["kind"] = "expander"
                d["label"] = c.label
                d["expanded"] = bool(getattr(p, "expanded", False))
            elif isinstance(c, et.Tab):
                d["kind"] = "tab"
                d["label"] = c.label
            elif c.type == "flex_container":
                fc = p.flex_container
                d["dir"] = "row" if fc.direction == 2 else "col"
                d["wrap"] = fc.wrap
            elif isinstance(c, et.Column):
                d["kind"] = "column"
                d["weight"] = getattr(c, "weight", 1)
            return d
        if name in ("Title", "Header", "Subheader"):
            return {"t": "heading", "level": {"Title": 1, "Header": 2, "Subheader": 3}[name], "text": c.value}
        if name == "Caption":
            return {"t": "caption", "text": c.value}
        if name == "Divider":
            return {"t": "divider"}
        if name in ("Markdown", "Latex", "Text"):
            return {"t": "md", "text": c.value}
        if name == "Code":
            return {"t": "code", "text": c.value}
        if name in ("Info", "Warning", "Error", "Success"):
            return {"t": "alert", "kind": name.lower(), "text": c.value}
        if name == "Exception":
            return {"t": "alert", "kind": "error", "text": "App error: " + str(c.value)}
        if name == "Metric":
            return {"t": "metric", "label": c.label, "value": c.value, "delta": c.delta,
                    "help": c.proto.help}
        if name in ("Dataframe", "Table"):
            p = c.proto
            d = {"t": "table", **_frame(c.value)}
            if name == "Dataframe":
                d["id"] = p.id
                d["key"] = p.id.split("-", 2)[-1] if p.id else ""
                d["edit"] = int(p.editing_mode)
                try:
                    d["config"] = json.loads(p.columns) if p.columns else {}
                except ValueError:
                    d["config"] = {}
            return d
        if name == "Space":
            return None
        if name == "UnknownElement":
            if c.type == "link_button":
                lb = c.proto.link_button if hasattr(c.proto, "link_button") else c.proto
                return {"t": "link", "label": lb.label, "url": lb.url, "help": lb.help}
            if "vega" in c.type:
                return self.chart(c.proto)
            if c.type in ("empty", "toast", "balloons", "snow"):
                return None
            return {"t": "alert", "kind": "info", "text": f"({c.type} not shown in this view)"}
        if isinstance(c, et.Widget):
            return self.widget(c, name)
        return None

    def chart(self, p):
        spec = json.loads(p.spec)
        import pyarrow as pa
        named = {}
        for ds in p.datasets:
            tbl = pa.ipc.open_stream(ds.data.data).read_all().to_pandas()
            named[ds.name] = json.loads(tbl.to_json(orient="records"))
        if named:
            spec.setdefault("datasets", {}).update(named)
        if p.data and p.data.data:
            tbl = pa.ipc.open_stream(p.data.data).read_all().to_pandas()
            spec["data"] = {"values": json.loads(tbl.to_json(orient="records"))}
        return {"t": "chart", "spec": spec}

    def widget(self, w, name):
        p = w.proto
        d = {"t": "widget", "w": name, "id": w.id, "label": getattr(w, "label", ""),
             "help": getattr(p, "help", "") or "", "disabled": bool(getattr(p, "disabled", False)),
             "key": w.id.split("-", 2)[-1] if w.id.startswith("$$ID") else None}
        try:
            if name in ("Selectbox", "Radio"):
                d["options"] = list(w.options)
                d["index"] = w.index
                d["placeholder"] = getattr(p, "placeholder", "")
            elif name in ("Multiselect", "ButtonGroup"):
                d["options"] = list(w.options)
                d["index"] = list(w.indices)
                if name == "ButtonGroup":
                    d["single"] = w._is_single_select
                    d["style"] = "pills" if p.style == p.PILLS else "segmented"
            elif name == "SelectSlider":
                d["options"] = list(w.options)
                v = w.value
                f = w.format_func
                d["index"] = d["options"].index(str(f(v)) if not isinstance(f(v), str) else f(v))
            elif name == "NumberInput":
                d.update(value=_clean(w.value), min=_clean(p.min) if p.has_min else None,
                         max=_clean(p.max) if p.has_max else None, step=_clean(p.step),
                         int=p.data_type == p.INT)
            elif name == "Slider":
                d.update(value=_clean(w.value), min=_clean(p.min), max=_clean(p.max), step=_clean(p.step))
            elif name in ("Checkbox", "Toggle"):
                d["value"] = bool(w.value)
                d["toggle"] = name == "Toggle" or getattr(p, "type", 0) == 1
            elif name in ("TextInput", "TextArea", "ChatInput"):
                d["value"] = w.value if name != "ChatInput" else ""
                d["password"] = getattr(p, "type", 0) == 1
                d["placeholder"] = getattr(p, "placeholder", "")
            elif name == "TimeInput":
                d["value"] = w.value.strftime("%H:%M") if w.value else ""
                d["step"] = int(getattr(p, "step", 900) or 900)
            elif name == "DateInput":
                d["value"] = str(w.value) if w.value and not isinstance(w.value, (list, tuple)) else ""
            elif name == "DownloadButton":
                d["url"] = p.url
            elif name == "FileUploader":
                d["file"] = self.files.get(w.id)
                d["types"] = list(p.type)
            elif name == "Button":
                d["primary"] = p.type == "primary"
        except Exception as exc:
            d["note"] = str(exc)
        return d


S = None


def get_session():
    global S
    if S is None:
        S = Session()
    return S


def _err(exc):
    return JSONResponse({"error": f"{type(exc).__name__}: {exc}", "trace": traceback.format_exc()[-3000:]},
                        status_code=500)


async def index(request):
    return FileResponse(os.path.join(HERE, "index.html"), headers={"Cache-Control": "no-store"})


def state(request):
    s = get_session()
    if s.busy:
        return JSONResponse({"busy": s.busy, "now": time.time()})
    with s.lock:
        try:
            s.refresh_app()
            return JSONResponse(s.state())
        except Exception as exc:
            return _err(exc)


async def act(request: Request):
    a = await request.json()
    s = get_session()
    if s.busy:
        return JSONResponse({"busy": s.busy, "now": time.time()}, status_code=409)
    if a.get("long"):  # a click that can take minutes (Generate): run in the background
        def work():
            with s.lock:
                s.busy = {"what": a.get("what", "Working"), "since": time.time()}
                s.error = None
                try:
                    s.act(a)
                    s.save()
                except Exception as exc:
                    s.error = f"{type(exc).__name__}: {exc}"
                finally:
                    s.busy = None
        threading.Thread(target=work, daemon=True).start()
        time.sleep(0.2)
        return JSONResponse({"busy": s.busy or {"what": a.get("what", "Working"), "since": time.time()},
                             "now": time.time()})

    def do():
        with s.lock:
            s.refresh_app()
            s.act(a)
            try:
                s.save()
            except Exception:
                pass
            return s.state()
    try:
        from starlette.concurrency import run_in_threadpool
        return JSONResponse(await run_in_threadpool(do))
    except Exception as exc:
        return _err(exc)


def media(request):
    url = request.query_params.get("url", "")
    store = _media["store"]
    if not store:
        return Response("nothing to download yet", status_code=404)
    fid = url.rsplit("/", 1)[-1].split(".")[0]
    try:
        f = store.get_file(fid)
    except Exception:
        return Response("download expired; build or rerun again", status_code=404)
    name = f.filename or "download"
    return Response(f.content, media_type=f.mimetype,
                    headers={"Content-Disposition": f'attachment; filename="{name}"'})


app = Starlette(routes=[
    Route("/", index),
    Route("/api/state", state),
    Route("/api/act", act, methods=["POST"]),
    Route("/api/media", media),
])

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8510
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
