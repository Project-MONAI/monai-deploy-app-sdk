<gsd-version v="2.1.4" />

<gsd-arguments>
  <settings>
    <keep-extra-args />
  </settings>
  <arg name="subcommand" type="string" optional />
  <arg name="name" type="string" optional />
</gsd-arguments>

<gsd-execute>
  <display msg="Loading thread context..." />
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
  <shell command="pi-gsd-tools">
    <args>
      <arg string="current-timestamp" />
      <arg string="--raw" />
    </args>
    <outs>
      <out type="string" name="timestamp" />
    </outs>
  </shell>
</gsd-execute>

## Thread Context (pre-injected by WXP)

**Subcommand:** <gsd-paste name="subcommand" />
**Name:** <gsd-paste name="name" />
**Timestamp:** <gsd-paste name="timestamp" />

**State:**
<gsd-paste name="state" />

---

<purpose>
Manage context threads — saved conversation checkpoints that can be resumed later.
A thread is a snapshot of the current agent context (active phase, decisions, blockers) that
survives `/clear` and can be handed off between sessions.

Subcommands: list, new [name], switch &lt;name&gt;
</purpose>

<process>

<step name="route">
<!-- Context pre-injected above via WXP -->

Parse `subcommand` and `name` from injected variables.

Route by subcommand:
- `list` (or empty) → **list_threads**
- `new [name]` → **new_thread**
- `switch <name>` → **switch_thread**
- Unknown → show help
</step>

<step name="list_threads">
Scan for thread files:

```bash
ls .planning/threads/*.md 2>/dev/null || echo "no threads"
```

**If no threads exist:**
```
No saved threads.

Create one to preserve context across /clear:
  /gsd-thread new <optional-name>
```
Exit.

**If threads found:**

For each thread file, read frontmatter and display:
```
## Context Threads

| Name | Phase | Created | Summary |
|------|-------|---------|---------|
| {name} | {phase} | {date} | {one-line} |

---
Switch to a thread:  /gsd-thread switch <name>
Create a new thread: /gsd-thread new [name]
```
</step>

<step name="new_thread">
Capture the current context as a named thread.

**Resolve name:**
- If `name` is provided, use it
- Otherwise generate from state: `{phase-slug}-{date}` (e.g., `auth-2025-01-15`)

**Ensure directory:**
```bash
mkdir -p .planning/threads
```

**Collect current context from state JSON:**
- `current_phase` + `phase_name`
- `milestone`
- `last_activity`
- Recent decisions (last 3 from STATE.md decisions table)
- Active blockers

**Write `.planning/threads/{name}.md`:**
```markdown
---
name: {name}
created: {timestamp}
phase: {current_phase}
phase_name: {phase_name}
milestone: {milestone}
status: active
---

## Thread: {name}

**Saved:** {timestamp}
**Phase:** {current_phase}: {phase_name}
**Milestone:** {milestone}

## Context Snapshot

{Summary of what's in progress: what was being worked on, key decisions made, any open questions}

## State at Save

[Key fields from STATE.md: current position, last activity, blockers]

## Resume Instructions

To resume this thread:
1. `/clear` - start fresh context
2. `/gsd-thread switch {name}` - restore this thread's context
3. `/gsd-resume-work` - re-orient with full project state
```

**Commit:**
```bash
pi-gsd-tools commit "docs: save context thread '{name}'" --files .planning/threads/{name}.md
```

**Confirm:**
```
✓ Thread saved: {name}

  Phase: {current_phase}: {phase_name}
  File: .planning/threads/{name}.md

Safe to /clear. Resume with: /gsd-thread switch {name}
```
</step>

<step name="switch_thread">
**Require `name`:** If empty, list available threads and ask user to choose.

```bash
cat .planning/threads/{name}.md 2>/dev/null || echo "Thread '{name}' not found."
```

**If not found:**
```
Thread '{name}' not found.
Available: [list]
```
Exit.

**If found:**

Read the thread file and display its full context:
```
## Restoring Thread: {name}

**Saved:** {created}
**Phase:** {phase}: {phase_name}

[Display the Context Snapshot section verbatim]

---
Thread context restored. Continuing from saved state.

Next:
- /gsd-execute-phase {phase} - continue executing
- /gsd-plan-phase {phase} - re-plan if context changed
- /gsd-resume-work - full project orientation
```

Mark the thread as resumed by updating its frontmatter `status: resumed` and adding a `resumed:` timestamp field.
</step>

</process>

<success_criteria>
- [ ] Subcommand routed correctly
- [ ] list: shows all saved threads with phase context
- [ ] new: saves full state snapshot to .planning/threads/
- [ ] switch: displays saved context and marks thread resumed
- [ ] Thread files committed to git
</success_criteria>
