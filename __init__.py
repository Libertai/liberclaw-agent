# Bump this when agent code changes in ways that require redeployment.
# Bumped to 8: per-channel pending cursors — GET /pending/channel/{channel}
# (non-destructive) + POST /pending/ack replace the destructive GET /pending
# for multi-channel delivery; legacy path kept for old control planes.
AGENT_VERSION = 8
