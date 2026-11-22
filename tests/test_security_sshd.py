"""Bash guard keeps sshd configuration and key files out of the agent's reach."""

from __future__ import annotations

import pytest

from baal_agent.security import check_command_safety


@pytest.mark.parametrize(
    "cmd",
    [
        "echo 'ssh-ed25519 AAAA me' >> /root/.ssh/authorized_keys",
        "mkdir -p ~/.ssh && cat k.pub > ~/.ssh/authorized_keys2",
        "sed -i 's/^Pass.*/PasswordAuthentication yes/' /etc/ssh/sshd_config",
        "cp x /etc/ssh/sshd_config.d/99.conf",
        "truncate -s0 /etc/ssh/liberclaw_authorized_keys",
        "cd /etc && vi ssh/sshd_config",
        "cat /etc/ssh/liberclaw-lockdown/lockdown.sh",
    ],
)
def test_blocks_sshd_paths(cmd):
    assert check_command_safety(cmd) is not None


@pytest.mark.parametrize(
    "cmd",
    [
        "ls -la /opt/baal-agent/workspace",
        "git clone git@github.com:o/r.git",
        "ssh-keygen -t ed25519 -f ./deploy_key -N ''",
    ],
)
def test_allows_ordinary_commands(cmd):
    assert check_command_safety(cmd) is None
