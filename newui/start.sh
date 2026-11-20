#!/bin/zsh
# Starts the new DFS UI on http://localhost:8510 (the Streamlit app on 8501 is separate).
cd ~/dfs_opt/newui
pkill -f "newui/server.py" 2>/dev/null; sleep 1
nohup ~/Library/CloudStorage/OneDrive-Comcast/JLab/NFL/venv/bin/python ~/dfs_opt/newui/server.py 8510 > ~/dfs_opt/newui/server.log 2>&1 &
sleep 8; open http://localhost:8510
