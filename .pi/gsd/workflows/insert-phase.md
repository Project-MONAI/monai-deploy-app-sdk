<gsd-version v="2.1.4" />

<gsd-arguments>
  <settings>
    <keep-extra-args />
  </settings>
  <arg name="after-phase" type="number" />
  <arg name="description" type="string" optional />
</gsd-arguments>

<gsd-execute>
  <shell command="pi-gsd-tools">
    <args>
      <arg string="init" />
      <arg string="phase-op" />
      <arg name="after-phase" wrap='"' />
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

## Context (pre-injected by WXP)

**Insert after phase:** <gsd-paste name="after-phase" />

**Description:** <gsd-paste name="description" />

**Phase Init Data:**
<gsd-paste name="init" />

<process>

<step name="parse_arguments">
Parse the command arguments:
- First argument: integer phase number to insert after
- Remaining arguments: phase description

Example: `/gsd-insert-phase 72 Fix critical auth bug`
-> after = 72
-> description = "Fix critical auth bug"

If arguments missing:

```
ERROR: Both phase number and description required
Usage: /gsd-insert-phase <after> <description>
Example: /gsd-insert-phase 72 Fix critical auth bug
```

Exit.

Validate first argument is an integer.
</step>

<step name="init_context">
Load phase operation context:

<!-- Context pre-injected above via WXP - variables available via <gsd-paste name="..."> -->

Check `roadmap_exists` from init JSON. If false:
```
ERROR: No roadmap found (.planning/ROADMAP.md)
```
Exit.
</step>

<step name="insert_phase">
**Delegate the phase insertion to gsd-tools:**

```bash
RESULT=$(pi-gsd-tools phase insert "${after_phase}" "${description}")
```

The CLI handles:
- Verifying target phase exists in ROADMAP.md
- Calculating next decimal phase number (checking existing decimals on disk)
- Generating slug from description
- Creating the phase directory (`.planning/phases/{N.M}-{slug}/`)
- Inserting the phase entry into ROADMAP.md after the target phase with (INSERTED) marker

Extract from result: `phase_number`, `after_phase`, `name`, `slug`, `directory`.
</step>

<step name="update_project_state">
Update STATE.md to reflect the inserted phase:

1. Read `.planning/STATE.md`
2. Under "## Accumulated Context" → "### Roadmap Evolution" add entry:
   ```
   - Phase {decimal_phase} inserted after Phase {after_phase}: {description} (URGENT)
   ```

If "Roadmap Evolution" section doesn't exist, create it.
</step>

<step name="completion">
Present completion summary:

```
Phase {decimal_phase} inserted after Phase {after_phase}:
- Description: {description}
- Directory: .planning/phases/{decimal-phase}-{slug}/
- Status: Not planned yet
- Marker: (INSERTED) - indicates urgent work

Roadmap updated: .planning/ROADMAP.md
Project state updated: .planning/STATE.md

---

## Next Up

**Phase {decimal_phase}: {description}** -- urgent insertion

`/gsd-plan-phase {decimal_phase}`

<sub>`/new` first -> fresh context window</sub>

---

**Also available:**
- Review insertion impact: Check if Phase {next_integer} dependencies still make sense
- Review roadmap

---
```
</step>

</process>

<anti_patterns>

- Don't use this for planned work at end of milestone (use /gsd-add-phase)
- Don't insert before Phase 1 (decimal 0.1 makes no sense)
- Don't renumber existing phases
- Don't modify the target phase content
- Don't create plans yet (that's /gsd-plan-phase)
- Don't commit changes (user decides when to commit)
</anti_patterns>

<success_criteria>
Phase insertion is complete when:

- [ ] `gsd-tools phase insert` executed successfully
- [ ] Phase directory created
- [ ] Roadmap updated with new phase entry (includes "(INSERTED)" marker)
- [ ] STATE.md updated with roadmap evolution note
- [ ] User informed of next steps and dependency implications
</success_criteria>
