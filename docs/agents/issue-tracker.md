# Issue tracker: Linear

Issues and specs for this repo live in Linear, workspace team **pdemirdjian** (issue keys `PDE-*`).
Use the Linear MCP tools (`mcp__plugin_linear_linear__*`) for all operations — there is no CLI.

## Conventions

- **Create an issue**: `save_issue` with `team: "pdemirdjian"`, a title, and a markdown description.
- **Read an issue**: `get_issue` with the `PDE-n` identifier; `list_comments` for its discussion.
- **List issues**: `list_issues` filtered by `team`, `state`, or `label` as needed.
- **Comment**: `save_comment` on the issue.
- **Apply / remove labels**: `save_issue` with the `labels` field (pass the full resulting label set).
- **Close**: `save_issue` setting status — `Canceled` for wontfix/rejected, `Done` for completed,
  `Duplicate` for dupes.

Statuses for the team: Backlog, Todo, In Progress, In Review, Done, Canceled, Duplicate.

A bare `#42` or `PDE-42` in conversation refers to Linear issue `PDE-42`. GitHub Issues on the
repo are not a request surface (only the Renovate dependency dashboard lives there).

## Pull requests as a triage surface

**PRs as a request surface: no.** _(Set to `yes` if this repo treats external PRs as feature
requests; `/triage` reads this flag. PRs live on GitHub — use `gh pr view/diff/comment` there.)_

## When a skill says "publish to the issue tracker"

Create a Linear issue on team `pdemirdjian`.

## When a skill says "fetch the relevant ticket"

Call `get_issue` with the `PDE-n` identifier, plus `list_comments`.

## Wayfinding operations

The **map** is a Linear issue; **child tickets** are sub-issues created with `parentId` set to the
map. Blocking uses a `Blocked by: PDE-n` line at the top of the child description (Linear
blocking relations aren't exposed through the MCP tools). Claim by assigning yourself
(`assignee: "me"`); resolve by commenting the answer and setting status `Done`.
