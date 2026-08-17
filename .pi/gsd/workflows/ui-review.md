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
      <arg string="resolve-model" />
      <arg string="gsd-ui-auditor" />
      <arg string="--raw" />
    </args>
    <outs>
      <suppress-errors />
      <out type="string" name="ui-auditor-model" />
    </outs>
  </shell>
  <shell command="pi-gsd-tools">
    <args>
      <arg string="agent-skills" />
      <arg string="gsd-ui-reviewer" />
    </args>
    <outs>
      <suppress-errors />
      <out type="string" name="agent-skills-ui-reviewer" />
    </outs>
  </shell>
</gsd-execute>

## Context (pre-injected)

**Phase:** <gsd-paste name="phase" />

**Phase Data:**
<gsd-paste name="init" />

**UI Auditor Model:** <gsd-paste name="ui-auditor-model" />

<purpose>
Retroactive 6-pillar visual audit of implemented frontend code. Standalone command that works on any project - GSD-managed or not. Produces scored UI-REVIEW.md with actionable findings.
</purpose>

<required_reading>
@.pi/gsd/references/ui-brand.md
</required_reading>

<available_agent_types>
Valid GSD subagent types (use exact names - do not fall back to 'general-purpose'):
- gsd-ui-auditor - Audits UI against design requirements
</available_agent_types>

<process>

## 0. Initialize

<!-- Context pre-injected above via WXP - variables available via <gsd-paste name="..."> -->

Parse: `phase_dir`, `phase_number`, `phase_name`, `phase_slug`, `padded_phase`, `commit_docs`.

```bash
UI_AUDITOR_MODEL=$(pi-gsd-tools resolve-model gsd-ui-auditor --raw)
```

Display banner:
```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 GSD ► UI AUDIT - PHASE {N}: {name}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

## 1. Detect Input State

```bash
SUMMARY_FILES=$(ls "${PHASE_DIR}"/*-SUMMARY.md 2>/dev/null)
UI_SPEC_FILE=$(ls "${PHASE_DIR}"/*-UI-SPEC.md 2>/dev/null | head -1)
UI_REVIEW_FILE=$(ls "${PHASE_DIR}"/*-UI-REVIEW.md 2>/dev/null | head -1)
```

**If `SUMMARY_FILES` empty:** Exit - "Phase {N} not executed. Run /gsd-execute-phase {N} first."

**If `UI_REVIEW_FILE` non-empty:** Use AskUserQuestion:
- header: "Existing UI Review"
- question: "UI-REVIEW.md already exists for Phase {N}."
- options:
  - "Re-audit - run fresh audit"
  - "View - display current review and exit"

If "View": display file, exit.
If "Re-audit": continue.

## 2. Gather Context Paths

Build file list for auditor:
- All SUMMARY.md files in phase dir
- All PLAN.md files in phase dir
- UI-SPEC.md (if exists - audit baseline)
- CONTEXT.md (if exists - locked decisions)

## 3. Spawn gsd-ui-auditor

```
◆ Spawning UI auditor...
```

Build prompt:

```markdown
Read .pi/gsd/agents/gsd-ui-auditor.md for instructions.

<objective>
Conduct 6-pillar visual audit of Phase {phase_number}: {phase_name}
{If UI-SPEC exists: "Audit against UI-SPEC.md design contract."}
{If no UI-SPEC: "Audit against abstract 6-pillar standards."}
</objective>

<files_to_read>
- {summary_paths} (Execution summaries)
- {plan_paths} (Execution plans - what was intended)
- {ui_spec_path} (UI Design Contract - audit baseline, if exists)
- {context_path} (User decisions, if exists)
</files_to_read>

${AGENT_SKILLS_UI_REVIEWER}

<config>
phase_dir: {phase_dir}
padded_phase: {padded_phase}
</config>
```

Omit null file paths.

```
Task(
  prompt=ui_audit_prompt,
  subagent_type="gsd-ui-auditor",
  model="{UI_AUDITOR_MODEL}",
  description="UI Audit Phase {N}"
)
```

## 4. Handle Return

**If `## UI REVIEW COMPLETE`:**

Display score summary:

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 GSD ► UI AUDIT COMPLETE ✓
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**Phase {N}: {Name}** - Overall: {score}/24

| Pillar            | Score |
| ----------------- | ----- |
| Copywriting       | {N}/4 |
| Visuals           | {N}/4 |
| Color             | {N}/4 |
| Typography        | {N}/4 |
| Spacing           | {N}/4 |
| Experience Design | {N}/4 |

Top fixes:
1. {fix}
2. {fix}
3. {fix}

Full review: {path to UI-REVIEW.md}

───────────────────────────────────────────────────────────────

## ▶ Next

- `/gsd-verify-work {N}` - UAT testing
- `/gsd-plan-phase {N+1}` - plan next phase

<sub>/new first → fresh context window</sub>

───────────────────────────────────────────────────────────────
```

## 5. Commit (if configured)

```bash
pi-gsd-tools commit "docs(${padded_phase}): UI audit review" --files "${PHASE_DIR}/${PADDED_PHASE}-UI-REVIEW.md"
```

</process>

<success_criteria>
- [ ] Phase validated
- [ ] SUMMARY.md files found (execution completed)
- [ ] Existing review handled (re-audit/view)
- [ ] gsd-ui-auditor spawned with correct context
- [ ] UI-REVIEW.md created in phase directory
- [ ] Score summary displayed to user
- [ ] Next steps presented
</success_criteria>
