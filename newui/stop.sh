#!/bin/zsh
pkill -f "newui/server.py" && echo "new UI stopped" || echo "new UI was not running"
