# Issue tracker: Linear

Issues and specs for this repo live in Linear, team **PDE**. Issue IDs look like
`PDE-12` and are referenced in branch names, commit messages, and PR titles.
Use the Linear MCP tools (`mcp__plugin_linear_linear__*`) for all operations.

## Conventions

- **Create an issue**: `save_issue` with the PDE team. Give it a clear title and
  a markdown body (real newlines, no escaped `\n`).
- **Read an issue**: `get_issue` with the ID (e.g. `PDE-12`); `list_comments` for discussion.
- **List issues**: `list_issues` filtered by team PDE, with state/label filters as needed.
- **Comment**: `save_comment` on the issue.
- **Apply / remove labels**: update the issue's labels via `save_issue`
  (create missing labels with `create_issue_label`).
- **Close**: `save_issue` moving the status to Done (or Canceled for wontfix).

## Pull requests as a triage surface

**PRs as a request surface: no.** _(GitHub PRs on this repo are not treated as
feature requests; `/triage` reads only Linear issues.)_

## When a skill says "publish to the issue tracker"

Create a Linear issue in team PDE.

## When a skill says "fetch the relevant ticket"

Call `get_issue` with the `PDE-n` ID (plus `list_comments` when discussion matters).

## Wayfinding operations

Used by `/wayfinder`. The **map** is a single Linear issue with **child** sub-issues as tickets.

- **Map**: an issue labelled `wayfinder:map` holding the Notes / Decisions-so-far / Fog body.
- **Child ticket**: a sub-issue of the map (set `parent` via `save_issue`).
  Labels: `wayfinder:<type>` (`research`/`prototype`/`grilling`/`task`).
  Once claimed, assign the ticket to the driving dev.
- **Blocking**: Linear "blocked by" relations where available; otherwise a
  `Blocked by: PDE-n, PDE-n` line at the top of the child body. A ticket is
  unblocked when every blocker is Done.
- **Frontier query**: `list_issues` for the map's open children, drop any with an
  open blocker or an assignee; first in map order wins.
- **Claim**: assign the issue to yourself (`save_issue` with the assignee), the session's first write.
- **Resolve**: `save_comment` with the answer, move the issue to Done, then append
  a context pointer to the map's Decisions-so-far.
