<gsd-version v="2.1.4" />

<gsd-arguments>
  <settings>
    <keep-extra-args />
  </settings>
  <arg name="idea" type="string" optional />
</gsd-arguments>

<gsd-execute>
  <display msg="Loading backlog context..." />
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

## Backlog Context (pre-injected by WXP)

**Idea:** <gsd-paste name="idea" />

**Timestamp:** <gsd-paste name="timestamp" />

**State:**
<gsd-paste name="state" />

**Roadmap Analysis:**
<gsd-paste name="roadmap" />

---

<purpose>
Park an idea as a backlog entry (999.x numbered phase) in ROADMAP.md. Zero friction — one command captures an idea without interrupting current work. The idea sits in the backlog until promoted by `/gsd-review-backlog`.
</purpose>

<process>

<step name="validate_context">
<!-- State, roadmap, and idea pre-injected above via WXP -->

Check that `.planning/ROADMAP.md` exists (from `state` JSON field `roadmap_exists`).

**If roadmap missing:**
```
Error: No ROADMAP.md found. Run /gsd-new-project or /gsd-new-milestone first.
```
Exit.

**If idea is empty (no $ARGUMENTS):**
Ask the user: "What's the idea? (one sentence description)"
Use the response as `idea`.
</step>

<step name="find_next_slot">
From the roadmap analysis JSON, extract the `phases` array. Find all phases where `phase_number` starts with `999` (e.g., `999.1`, `999.2`).

Compute next backlog number:
- If no 999.x entries exist → use `999.1`
- Otherwise → use `999.(max_decimal + 1)`, e.g., if `999.3` exists → `999.4`

Set `BACKLOG_NUM` = next available 999.x slot.
</step>

<step name="add_entry">
Append to `.planning/ROADMAP.md` under a `## Backlog` section (create the section if missing):

```markdown
- [ ] **Phase {BACKLOG_NUM}**: {idea}
```

Use the roadmap `roadmap add-phase` command if available, or append directly:

```bash
pi-gsd-tools roadmap add-phase "{BACKLOG_NUM}" "{idea}" --raw
```

If the CLI command fails or is unavailable, append manually to ROADMAP.md.
</step>

<step name="commit">
```bash
pi-gsd-tools commit "docs: add backlog entry {BACKLOG_NUM} - {idea_slug}" --files .planning/ROADMAP.md
```
</step>

<step name="confirm">
```
✓ Backlog entry added

  Phase {BACKLOG_NUM}: {idea}

---

Review and promote backlog: /gsd-review-backlog
```
</step>

</process>

<success_criteria>
- [ ] ROADMAP.md has new 999.x entry
- [ ] 999.x number is sequential (no gaps or duplicates)
- [ ] Entry committed to git
- [ ] User sees confirmation with the assigned phase number
</success_criteria>
