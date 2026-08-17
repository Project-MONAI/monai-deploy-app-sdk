<gsd-version v="2.1.4" />

<gsd-arguments>
  <settings>
    <keep-extra-args />
  </settings>
  <arg name="subcommand" type="string" optional />
  <arg name="name" type="string" optional />
</gsd-arguments>

<gsd-execute>
  <display msg="Loading workstream state..." />
  <shell command="pi-gsd-tools">
    <args>
      <arg string="workstream" />
      <arg string="list" />
      <arg string="--raw" />
    </args>
    <outs>
      <suppress-errors />
      <out type="string" name="ws-list" />
    </outs>
  </shell>
  <shell command="pi-gsd-tools">
    <args>
      <arg string="workstream" />
      <arg string="get" />
      <arg string="--raw" />
    </args>
    <outs>
      <suppress-errors />
      <out type="string" name="ws-active" />
    </outs>
  </shell>
  <shell command="pi-gsd-tools">
    <args>
      <arg string="state" />
      <arg string="json" />
      <arg string="--raw" />
    </args>
    <outs>
      <suppress-errors />
      <out type="string" name="state" />
    </outs>
  </shell>
</gsd-execute>

## Workstream Context (pre-injected by WXP)

**Subcommand:** <gsd-paste name="subcommand" />
**Name:** <gsd-paste name="name" />

**Active workstream:** <gsd-paste name="ws-active" />

**Workstream list:**
<gsd-paste name="ws-list" />

**State:**
<gsd-paste name="state" />

---

<purpose>
Manage GSD workstreams — isolated parallel tracks of work within a project.
Each workstream has its own ROADMAP.md, STATE.md, and phase history.

Subcommands: list, create &lt;name&gt;, switch &lt;name&gt;, status [name], complete &lt;name&gt;
</purpose>

<process>

<step name="route">
<!-- Context pre-injected above via WXP -->

Parse `subcommand` and `name` from injected variables.

**Route by subcommand:**

| Subcommand | Action |
|------------|--------|
| `list` (or empty) | → **list_workstreams** |
| `create <name>` | → **create_workstream** |
| `switch <name>` | → **switch_workstream** |
| `status [name]` | → **show_status** |
| `complete <name>` | → **complete_workstream** |

**If subcommand is unrecognised:** Show help (see offer_help step).
</step>

<step name="list_workstreams">
<!-- ws-list and ws-active are pre-injected -->

Parse `ws-list` JSON for workstream entries. Display:

```
## Workstreams

Active: {ws-active || "(none — on main planning root)"}

| Name | Status | Phases | Progress |
|------|--------|--------|---------|
| {name} | {active|inactive} | {phase_count} | {pct}% |
| ...  | ...    | ...    | ...     |

---
/gsd-workstreams create <name>   - create a new workstream
/gsd-workstreams switch <name>   - switch to a workstream
/gsd-workstreams status <name>   - detailed workstream status
/gsd-workstreams complete <name> - close a workstream
```

**If no workstreams exist:**
```
No workstreams yet. You're working in the main planning root.

Create a workstream to isolate parallel work:
  /gsd-workstreams create <name>
```
</step>

<step name="create_workstream">
**Require `name`:**
If `name` is empty, ask: "Workstream name? (lowercase, no spaces — e.g. mobile-app, api-v2)"

Validate: lowercase alphanumeric with hyphens/underscores only.

```bash
pi-gsd-tools workstream create {name}
```

Confirm:
```
✓ Workstream '{name}' created

To switch to it: /gsd-workstreams switch {name}
```
</step>

<step name="switch_workstream">
**Require `name`:** If empty, list available workstreams and ask user to choose.

```bash
pi-gsd-tools workstream set {name}
```

Confirm:
```
✓ Switched to workstream: {name}

All subsequent GSD commands will operate within this workstream.
To return to main: /gsd-workstreams switch main
```
</step>

<step name="show_status">
**Target:** `name` if provided, otherwise the active workstream.

```bash
pi-gsd-tools workstream status {name}
```

Display the full status output including phase progress, open todos, and blockers.
</step>

<step name="complete_workstream">
**Require `name`:** If empty, ask which workstream to complete.

Confirm before completing:
```
Complete workstream '{name}'?

This will:
- Mark all phases as complete
- Archive the workstream planning files

Continue? (yes / no)
```

If yes:
```bash
pi-gsd-tools workstream complete {name}
```

Confirm:
```
✓ Workstream '{name}' completed and archived.
```
</step>

<step name="offer_help">
```
## /gsd-workstreams

Manage parallel tracks of work within a project.

Usage:
  /gsd-workstreams                  - list all workstreams
  /gsd-workstreams create <name>    - create a new workstream
  /gsd-workstreams switch <name>    - activate a workstream
  /gsd-workstreams status [name]    - show workstream details
  /gsd-workstreams complete <name>  - close a finished workstream

Current: {ws-active || "main planning root"}
```
</step>

</process>

<success_criteria>
- [ ] Active workstream pre-injected (no runtime read needed)
- [ ] Workstream list pre-injected
- [ ] Subcommand routed correctly
- [ ] Each action calls the appropriate CLI command
- [ ] Confirmations displayed after mutations
</success_criteria>
