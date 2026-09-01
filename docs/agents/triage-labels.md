# Triage Labels

The skills speak in terms of canonical triage roles. This file maps those roles to the actual
label strings used in this repo's issue tracker (Linear, team `pdemirdjian`). All labels below
already exist in Linear — apply them, don't create duplicates.

## State roles

| Label in mattpocock/skills | Label in Linear   | Meaning                                  |
| -------------------------- | ----------------- | ---------------------------------------- |
| `needs-triage`             | `needs-triage`    | Maintainer needs to evaluate this issue  |
| `needs-info`               | `needs-info`      | Waiting on reporter for more information |
| `ready-for-agent`          | `ready-for-agent` | Fully specified, ready for an AFK agent  |
| `ready-for-human`          | `ready-for-human` | Requires human implementation            |
| `wontfix`                  | `wontfix`         | Will not be actioned                     |

## Category roles

| Label in mattpocock/skills | Label in Linear | Meaning                    |
| -------------------------- | --------------- | -------------------------- |
| `bug`                      | `Bug`           | Something is broken        |
| `enhancement`              | `Feature`       | New feature or improvement |

Linear also has an `Improvement` label; it is for manual use only — triage never applies it.

When a skill mentions a role (e.g. "apply the AFK-ready triage label"), use the corresponding
label string from these tables.
