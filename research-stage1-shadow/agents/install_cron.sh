#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CRON_LINE="* * * * * bash ${SCRIPT_DIR}/watchdog.sh >> /dev/null 2>&1"

(crontab -l 2>/dev/null | grep -v "watchdog.sh"; echo "$CRON_LINE") | crontab -
echo "Cron installed: watchdog runs every 60s"
