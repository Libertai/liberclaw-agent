# Bump this when agent code changes in ways that require redeployment.
# Bumped to 6: cap images sent to inference (keep most recent N, drop older)
# so vision requests don't fail when more than a handful of images accumulate.
AGENT_VERSION = 6
