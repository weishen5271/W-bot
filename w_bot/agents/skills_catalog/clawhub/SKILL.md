---
name: clawhub
description: Use ClawHub CLI to search, install, update, and list skills in workspace.
metadata: {"requires": {"bins": ["npx"], "env": []}, "always": false}
always: false
allow_writes: true
allow_exec: true
---

# ClawHub Skill

Use this skill when user wants to discover or install third-party skills.

## Important

- This skill requires `exec` tool and `npx` on host.
- Install target must be workspace skills directory.

## Standard commands

1. Search skills:

```bash
npx --yes clawhub@latest search "<query>" --limit 5
```

2. Install a skill:

```bash
npx --yes clawhub@latest install <slug> --workdir .
```

3. Update all installed skills:

```bash
npx --yes clawhub@latest update --all --workdir .
```

4. List installed skills:

```bash
npx --yes clawhub@latest list --workdir .
```

## Usage notes

- Run commands via `exec` tool.
- After installation, ask user to send next message so refreshed skill summary can include newly installed skills.
