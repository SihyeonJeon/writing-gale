#!/usr/bin/env bash
# Starts one run of the note writer workflow on GitHub Actions. Run it from any always-on machine
# every 30 minutes (see the systemd units next to this file); GitHub's own schedule trigger skipped
# most runs. The workflow's WRITER_MODE switch still decides whether a run does anything.
#
# Needs a fine-grained GitHub token with "Actions: read and write" on this one repository, stored
# in ~/.config/writing-gale/token (chmod 600). Without the file it does nothing.
set -euo pipefail
TOKEN_FILE="$HOME/.config/writing-gale/token"
REPO="${WRITER_REPO:?set WRITER_REPO to owner/name}"
WORKFLOW="community_note_writer.yaml"
if [ ! -s "$TOKEN_FILE" ]; then echo "no token at $TOKEN_FILE; nothing to do"; exit 0; fi
code=$(curl -sS -o /tmp/writing-gale-trigger.out -w '%{http_code}' --max-time 30 -X POST \
  -H "Authorization: Bearer $(cat "$TOKEN_FILE")" \
  -H "Accept: application/vnd.github+json" -H "X-GitHub-Api-Version: 2022-11-28" \
  "https://api.github.com/repos/$REPO/actions/workflows/$WORKFLOW/dispatches" -d '{"ref":"main"}')
case "$code" in 2*) echo "dispatched ($code)";; *) echo "dispatch failed: HTTP $code"; head -c 300 /tmp/writing-gale-trigger.out; exit 1;; esac
