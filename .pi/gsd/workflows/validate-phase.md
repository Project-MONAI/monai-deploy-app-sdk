<gsd-version v="2.1.4" />

<gsd-arguments>
  <settings>
    <keep-extra-args />
  </settings>
  <arg name="phase" type="number" />
</gsd-arguments>

<gsd-execute>
  <shell command="pi-gsd-tools">
    <args>
      <arg string="init" />
      <arg string="phase-op" />
      <arg name="phase" wrap='"' />
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
      <arg string="agent-skills" />
      <arg string="gsd-nyquist-auditor" />
    </args>
    <outs>
      <suppress-errors />
      <out type="string" name="agent-skills-auditor" />
    </outs>
  </shell>
  <shell command="pi-gsd-tools">
    <args>
      <arg string="resolve-model" />
      <arg string="gsd-nyquist-auditor" />
      <arg string="--raw" />
    </args>
    <outs>
      <suppress-errors />
      <out type="string" name="auditor-model" />
    </outs>
  </shell>
  <shell command="pi-gsd-tools">
    <args>
      <arg string="config-get" />
      <arg string="workflow.nyquist_validation" />
      <arg string="--raw" />
    </args>
    <outs>
      <suppress-errors />
      <out type="string" name="nyquist-cfg" />
    </outs>
  </shell>
</gsd-execute>

## Initialization Context (pre-injected by WXP)

**Phase:** <gsd-paste name="phase" />

**Phase Init Data:**
<gsd-paste name="init" />

**Auditor Model:** <gsd-paste name="auditor-model" />

<process>

## 0. Initialize

<!-- Context pre-injected above via WXP - variables available via <gsd-paste name="..."> -->

Parse: `phase_dir`, `phase_number`, `phase_name`, `phase_slug`, `padded_phase`.

```bash
AUDITOR_MODEL=$(pi-gsd-tools resolve-model gsd-nyquist-auditor --raw)
NYQUIST_CFG=$(pi-gsd-tools config-get workflow.nyquist_validation --raw)
```

If `NYQUIST_CFG` is `false`: exit with "Nyquist validation is disabled. Enable via /gsd-settings."

Display banner: `GSD > VALIDATE PHASE {N}: {name}`

## 1. Detect Input State

```bash
VALIDATION_FILE=$(ls "${PHASE_DIR}"/*-VALIDATION.md 2>/dev/null | head -1)
SUMMARY_FILES=$(ls "${PHASE_DIR}"/*-SUMMARY.md 2>/dev/null)
```

- **State A** (`VALIDATION_FILE` non-empty): Audit existing
- **State B** (`VALIDATION_FILE` empty, `SUMMARY_FILES` non-empty): Reconstruct from artifacts
- **State C** (`SUMMARY_FILES` empty): Exit - "Phase {N} not executed. Run /gsd-execute-phase {N} ${GSD_WS} first."

## 2. Discovery

### 2a. Read Phase Artifacts

Read all PLAN and SUMMARY files. Extract: task lists, requirement IDs, key-files changed, verify blocks.

### 2b. Build Requirement-to-Task Map

Per task: `{ task_id, plan_id, wave, requirement_ids, has_automated_command }`

### 2c. Detect Test Infrastructure

State A: Parse from existing VALIDATION.md Test Infrastructure table.
State B: Filesystem scan:

```bash
find . -name "pytest.ini" -o -name "jest.config.*" -o -name "vitest.config.*" -o -name "pyproject.toml" 2>/dev/null | head -10
find . \( -name "*.test.*" -o -name "*.spec.*" -o -name "test_*" \) -not -path "*/node_modules/*" 2>/dev/null | head -40
```

### 2d. Cross-Reference

Match each requirement to existing tests by filename, imports, test descriptions. Record: requirement → test_file → status.

## 3. Gap Analysis

Classify each requirement:

| Status  | Criteria                                  |
| ------- | ----------------------------------------- |
| COVERED | Test exists, targets behavior, runs green |
| PARTIAL | Test exists, failing or incomplete        |
| MISSING | No test found                             |

Build: `{ task_id, requirement, gap_type, suggested_test_path, suggested_command }`

No gaps → skip to Step 6, set `nyquist_compliant: true`.

## 4. Present Gap Plan

Call AskUserQuestion with gap table and options:
1. "Fix all gaps" → Step 5
2. "Skip - mark manual-only" → add to Manual-Only, Step 6
3. "Cancel" → exit

## 5. Spawn gsd-nyquist-auditor

```
Task(
  prompt="Read .pi/gsd/agents/gsd-nyquist-auditor.md for instructions.\n\n" +
    "<files_to_read>{PLAN, SUMMARY, impl files, VALIDATION.md}</files_to_read>" +
    "<gaps>{gap list}</gaps>" +
    "<test_infrastructure>{framework, config, commands}</test_infrastructure>" +
    "<constraints>Never modify impl files. Max 3 debug iterations. Escalate impl bugs.</constraints>" +
    "${AGENT_SKILLS_AUDITOR}",
  subagent_type="gsd-nyquist-auditor",
  model="{AUDITOR_MODEL}",
  description="Fill validation gaps for Phase {N}"
)
```

Handle return:
- `## GAPS FILLED` → record tests + map updates, Step 6
- `## PARTIAL` → record resolved, move escalated to manual-only, Step 6
- `## ESCALATE` → move all to manual-only, Step 6

## 6. Generate/Update VALIDATION.md

**State B (create):**
1. Read template from `.pi/gsd/templates/VALIDATION.md`
2. Fill: frontmatter, Test Infrastructure, Per-Task Map, Manual-Only, Sign-Off
3. Write to `${PHASE_DIR}/${PADDED_PHASE}-VALIDATION.md`

**State A (update):**
1. Update Per-Task Map statuses, add escalated to Manual-Only, update frontmatter
2. Append audit trail:

```markdown
## Validation Audit {date}
| Metric     | Count |
| ---------- | ----- |
| Gaps found | {N}   |
| Resolved   | {M}   |
| Escalated  | {K}   |
```

## 7. Commit

```bash
git add {test_files}
git commit -m "test(phase-${PHASE}): add Nyquist validation tests"

pi-gsd-tools commit "docs(phase-${PHASE}): add/update validation strategy"
```

## 8. Results + Routing

**Compliant:**
```
GSD > PHASE {N} IS NYQUIST-COMPLIANT
All requirements have automated verification.
▶ Next: /gsd-audit-milestone ${GSD_WS}
```

**Partial:**
```
GSD > PHASE {N} VALIDATED (PARTIAL)
{M} automated, {K} manual-only.
▶ Retry: /gsd-validate-phase {N} ${GSD_WS}
```

Display `/new` reminder.

</process>

<success_criteria>
- [ ] Nyquist config checked (exit if disabled)
- [ ] Input state detected (A/B/C)
- [ ] State C exits cleanly
- [ ] PLAN/SUMMARY files read, requirement map built
- [ ] Test infrastructure detected
- [ ] Gaps classified (COVERED/PARTIAL/MISSING)
- [ ] User gate with gap table
- [ ] Auditor spawned with complete context
- [ ] All three return formats handled
- [ ] VALIDATION.md created or updated
- [ ] Test files committed separately
- [ ] Results with routing presented
</success_criteria>
