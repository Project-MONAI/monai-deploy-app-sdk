<gsd-version v="2.1.4" />

<gsd-arguments>
  <settings><keep-extra-args /></settings>
</gsd-arguments>

<gsd-execute>
  <display msg="Loading backlog and todos..." />
  <shell command="pi-gsd-tools">
    <args>
      <arg string="init" />
      <arg string="todos" />
      <arg string="0" />
    </args>
    <outs>
      <out type="string" name="todos-data" />
    </outs>
  </shell>
  <if>
    <condition>
      <starts-with>
        <left name="todos-data" />
        <right type="string" value="@file:" />
      </starts-with>
    </condition>
    <then>
      <string-op op="split">
        <args>
          <arg name="todos-data" />
          <arg type="string" value="@file:" />
        </args>
        <outs>
          <out type="string" name="todos-data-file" />
        </outs>
      </string-op>
      <shell command="cat">
        <args>
          <arg name="todos-data-file" wrap='"' />
        </args>
        <outs>
          <out type="string" name="todos-data" />
        </outs>
      </shell>
    </then>
  </if>
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
      <arg string="roadmap" />
      <arg string="analyze" />
      <arg string="--raw" />
    </args>
    <outs>
      <suppress-errors />
      <out type="string" name="roadmap" />
    </outs>
  </shell>
</gsd-execute>

## Backlog Review Context (pre-injected by WXP)

**Todos:**
<gsd-paste name="todos-data" />

**Roadmap:**
<gsd-paste name="roadmap" />

**State:**
<gsd-paste name="state" />

---

<purpose>
Review accumulated backlog items (999.x phases and pending todos) and decide what to do with each: promote to a real phase, convert to a todo, discard, or keep.

This is the "inbox zero" command for ideas that were parked during active development.
</purpose>

<process>

<step name="load_backlog">
<!-- Context pre-injected above via WXP - variables available via <gsd-paste name="..."> -->

**Step A: Extract backlog phases from roadmap.**

From `roadmap` JSON, find all phases where `phase_number` starts with `999` (backlog entries added via `/gsd-add-backlog`).

**Step B: Extract pending todos from todos data.**

From `todos-data` JSON, extract `todos` array with fields: `id`, `title`, `area`, `created`, `problem`.

**If both backlog phases and todos are empty:**
```
Nothing in the backlog. The queue is clear.

To capture an idea: /gsd-add-backlog <idea>
To capture a todo: /gsd-add-todo <task>
```
Exit.
</step>

<step name="present_inventory">
Display a combined inventory:

```
## Backlog Review

### 999.x Backlog Phases ({N} items)
| Phase | Idea | Added |
|-------|------|-------|
| 999.1 | {idea} | {date} |
| 999.2 | {idea} | {date} |

### Pending Todos ({M} items)
| # | Title | Area | Created |
|---|-------|------|---------|
| 1 | {title} | {area} | {date} |
| 2 | {title} | {area} | {date} |

**Total:** {N+M} items to review
```

Ask:
```
Options:
1. Review each item interactively (recommended)
2. Promote a specific backlog phase → real phase number
3. Work on a specific todo
4. Discard a backlog phase
5. Done (keep everything)
```
</step>

<step name="interactive_review">
**If user chooses interactive review:**

For each backlog phase (999.x), present:
```
## Phase 999.{N}: {idea}

Options:
1. Promote to next available phase slot  ← recommended if actionable
2. Convert to a todo (more granular)
3. Keep in backlog
4. Discard (remove from ROADMAP.md)
```

**Promote:** Remove the 999.x entry, add as a properly numbered phase at the end of the current milestone using:
```bash
pi-gsd-tools roadmap add-phase "{next_available_number}" "{idea_text}" --raw
```
Then remove the 999.x placeholder:
```bash
pi-gsd-tools roadmap remove-phase "999.{N}" --raw
```

**Convert to todo:** Create a todo file (see `/gsd-add-todo` workflow) and remove the 999.x phase entry.

**Discard:**
```bash
pi-gsd-tools roadmap remove-phase "999.{N}" --raw
```

For each pending todo, present:
```
## Todo: {title}
Area: {area}
Problem: {problem excerpt}

Options:
1. Work on this now → promote to current phase plan
2. Keep as todo
3. Promote to backlog phase
4. Mark done (won't be worked on)
```
</step>

<step name="commit_changes">
After all reviews, commit any ROADMAP.md changes and completed todos:

```bash
pi-gsd-tools commit "docs: backlog review - promoted {X} items, discarded {Y}" --files .planning/ROADMAP.md .planning/todos/done/
```

Display summary:
```
## Backlog Review Complete

✓ Promoted: {list of promoted items}
✓ Converted: {list of todos created}
✓ Discarded: {count}
→ Kept: {count remaining}

{If phases promoted:}
Next: /gsd-plan-phase {new_phase_number}
```
</step>

</process>

<success_criteria>
- [ ] All 999.x backlog phases listed
- [ ] All pending todos listed
- [ ] Each item reviewed with a clear decision
- [ ] Promotions written to ROADMAP.md
- [ ] Discards removed from ROADMAP.md
- [ ] Changes committed to git
- [ ] User knows what's next
</success_criteria>
