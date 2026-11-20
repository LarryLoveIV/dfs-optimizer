"""Check: a Cash-mode build (no randomness) through the new UI matches a plain headless
app run, lineup for lineup.  usage: venv/bin/python check_same.py [lineups]"""
import json, base64, os, sys
from streamlit.testing.v1 import AppTest
import server
D=os.path.expanduser("~/dfs_opt")  # wk09_salaries.csv + wk09_main_tg.csv live here; NFL=server.APP_DIR
SAL=open(f"{D}/wk09_salaries.csv","rb").read(); PROJ=open(f"{D}/wk09_main_tg.csv","rb").read()
N=int(sys.argv[1]) if len(sys.argv)>1 else 10
# A: plain Streamlit-style headless run (how the test harness drives the app)
lab=lambda els,l:[e for e in els if getattr(e,"label",None)==l]
at=AppTest.from_file(f"{NFL}/app.py", default_timeout=3600); at.run()
lab(at.button_group,"Salaries")[0].set_value("Upload CSV"); at.run()
at.sidebar.number_input[0].set_value(N); at.run()
lab(at.sidebar.radio,"Type")[0].set_value("Cash (double-up)"); at.run()
at.file_uploader(key="dk").set_value(("DKSalaries.csv",SAL,"text/csv"))
at.file_uploader(key="cheat").set_value(("proj.csv",PROJ,"text/csv")); at.run()
lab(at.button,"Generate Lineups")[0].click(); at.run()
lu=at.session_state["results"]["lineups"]; lu=lu[lu.Position!="TOTAL"]
A=[sorted(g.Name) for _,g in lu.groupby("Lineup")]
# B: the new UI's session, driven the way the page drives it
s=server.Session()
def walk(ns):
    for n in ns:
        yield n; yield from walk(n.get("children") or [])
def W(w,label):
    st=s.state(); return next(n for n in walk(st["main"]+st["sidebar"]) if n.get("w")==w and label in n["label"])
src=W("ButtonGroup","Salaries"); s.act({"op":"set","id":src["id"],"value":src["options"].index("Upload CSV")})
s.act({"op":"set","id":W("NumberInput","Number of lineups")["id"],"value":N})
t=W("Radio","Type"); s.act({"op":"set","id":t["id"],"value":t["options"].index("Cash (double-up)")})
s.act({"op":"upload","id":W("FileUploader","Salaries")["id"],"name":"DKSalaries.csv","b64":base64.b64encode(SAL).decode()})
s.act({"op":"upload","id":W("FileUploader","Projections")["id"],"name":"proj.csv","b64":base64.b64encode(PROJ).decode()})
s.act({"op":"click","id":W("Button","Generate Lineups")["id"]})
lu=s.at.session_state["results"]["lineups"]; lu=lu[lu.Position!="TOTAL"]
B=[sorted(g.Name) for _,g in lu.groupby("Lineup")]
print(f"Cash, {N} lineups: Streamlit {len(A)} vs new UI {len(B)} -> IDENTICAL in order: {A==B}")
