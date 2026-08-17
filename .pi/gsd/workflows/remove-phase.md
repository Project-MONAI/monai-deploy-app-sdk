<gsd-version v="2.1.4" />

<gsd-arguments>
  <settings><keep-extra-args /></settings>
  <arg name="phase" type="number" />
</gsd-arguments>

<gsd-execute>
  <shell command="pi-gsd-tools">
    <args>
      <arg string="init" />
      <arg string="phase-op" />
    </args>
    <outs>
      <out type="string" name="init" />
    </outs>
  </shell>
  <if>
    <condition>
      <starts-with>
        <left name="init" />
        <right type="string" value="@file:" />
      </starts-with>
    </condition>
    <then>
      <string-op op="split">
        <args>
          <arg name="init" />
          <arg type="string" value="@file:" />
        </args>
        <outs>
          <out type="string" name="init-file" />
        </outs>
      </string-op>
      <shell command="cat">
        <args>
          <arg name="init-file" wrap='"' />
        </args>
        <outs>
          <out type="string" name="init" />
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
      <out type="string" name="state" />
    </outs>
  </shell>
</gsd-execute>

## Context (pre-injected)

**Phase:** <gsd-paste name="phase" />

**Phase Data:**
<gsd-paste name="init" />

<purpose>
Remove an unstarted future phase from the project roadmap, delete its directory, renumber all subsequent phases to maintain a clean linear sequence, and commit the change. The git commit serves as the historical record of removal.
</purpose>

<required_reading>
Read all files referenced by the invoking prompt's execution_context before starting.
</required_reading>

<process>

<step name="parse_arguments">
Parse the command arguments:
- Argument is the phase number to remove (integer or decimal)
- Example: `/gsd-remove-phase 17` → phase = 17
- Example: `/gsd-remove-phase 16.1` → phase = 16.1

If no argument provided:

```
ERROR: Phase number required
Usage: /gsd-remove-phase <phase-number>
Example: /gsd-remove-phase 17
```

Exit.
</step>

<step name="init_context">
Load phase operation context:

<!-- Context pre-injected above via WXP - variables available via <gsd-paste name="..."> -->

Extract: `phase_found`, `phase_dir`, `phase_number`, `commit_docs`, `roadmap_exists`.

Also read STATE.md and ROADMAP.md content for parsing current position.
</step>

<step name="validate_future_phase">
Verify the phase is a future phase (not started):

1. Compare target phase to current phase from STATE.md
2. Target must be > current phase number

If target <= current phase:

```
ERROR: Cannot remove Phase {target}

Only future phases can be removed:
- Current phase: {current}
- Phase {target} is current or completed

To abandon current work, use /gsd-pause-work instead.
```

Exit.
</step>

<step name="confirm_removal">
Present removal summary and confirm:

```
Removing Phase {target}: {Name}

This will:
- Delete: .planning/phases/{target}-{slug}/
- Renumber all subsequent phases
- Update: ROADMAP.md, STATE.md

Proceed? (y/n)
```

Wait for confirmation.
</step>

<step name="execute_removal">
**Delegate the entire removal operation to gsd-tools:**

```bash
RESULT=$(pi-gsd-tools phase remove "${target}")
```

If the phase has executed plans (SUMMARY.md files), gsd-tools will error. Use `--force` only if the user confirms:

```bash
RESULT=$(pi-gsd-tools phase remove "${target}" --force)
```

The CLI handles:
- Deleting the phase directory
- Renumbering all subsequent directories (in reverse order to avoid conflicts)
- Renaming all files inside renumbered directories (PLAN.md, SUMMARY.md, etc.)
- Updating ROADMAP.md (removing section, renumbering all phase references, updating dependencies)
- Updating STATE.md (decrementing phase count)

Extract from result: `removed`, `directory_deleted`, `renamed_directories`, `renamed_files`, `roadmap_updated`, `state_updated`.
</step>

<step name="commit">
Stage and commit the removal:

```bash
pi-gsd-tools commit "chore: remove phase {target} ({original-phase-name})" --files .planning/
```

The commit message preserves the historical record of what was removed.
</step>

<step name="completion">
Present completion summary:

```
Phase {target} ({original-name}) removed.

Changes:
- Deleted: .planning/phases/{target}-{slug}/
- Renumbered: {N} directories and {M} files
- Updated: ROADMAP.md, STATE.md
- Committed: chore: remove phase {target} ({original-name})

---

## What's Next

Would you like to:
- `/gsd-progress` - see updated roadmap status
- Continue with current phase
- Review roadmap

---
```
</step>

</process>

<anti_patterns>

- Don't remove completed phases (have SUMMARY.md files) without --force
- Don't remove current or past phases
- Don't manually renumber - use `gsd-tools phase remove` which handles all renumbering
- Don't add "removed phase" notes to STATE.md - git commit is the record
- Don't modify completed phase directories
</anti_patterns>

<success_criteria>
Phase removal is complete when:

- [ ] Target phase validated as future/unstarted
- [ ] `gsd-tools phase remove` executed successfully
- [ ] Changes committed with descriptive message
- [ ] User informed of changes
</success_criteria>
