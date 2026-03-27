---
name: project-analyzer
description: Analyze the local repository before making broad code changes.
metadata: {"requires": {"bins": ["rg"], "env": []}, "always": false}
always: false
---

# Project Analyzer Skill

Use this skill when a user asks for implementation work in the current repository.

## Workflow

1. Discover structure first.
2. Read relevant files before editing.
3. Implement minimal, targeted changes.
4. Verify with lightweight checks when possible.

## Suggested commands

- `rg --files`
- `rg -n "<keyword>" <path>`
- `python -m compileall src`

## Notes

- Do not modify unrelated files.
- Keep behavior backward compatible unless user asks otherwise.
