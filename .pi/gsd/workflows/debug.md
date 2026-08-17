<gsd-version v="2.1.4" />

<gsd-arguments>
  <settings>
    <keep-extra-args />
  </settings>
  <arg name="description" type="string" optional />
</gsd-arguments>

<gsd-execute>
  <display msg="Loading debug context..." />
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
  <shell command="pi-gsd-tools">
    <args>
      <arg string="agent-skills" />
      <arg string="gsd-debugger" />
    </args>
    <outs>
      <suppress-errors />
      <out type="string" name="agent-skills-debugger" />
    </outs>
  </shell>
</gsd-execute>

## Debug Context (pre-injected by WXP)

**Problem:** <gsd-paste name="description" />

**Timestamp:** <gsd-paste name="timestamp" />

**Project State:**
<gsd-paste name="state" />

**Debugger Skills:**
<gsd-paste name="agent-skills-debugger" />

---

<purpose>
Systematic debugging session. Diagnoses failures, errors, and unexpected behavior using structured root-cause analysis. Spawns gsd-debugger with full project context for focused investigation.

For post-mortem investigation of completed phases, use `/gsd-forensics` instead.
</purpose>

<available_agent_types>
Valid GSD subagent types (use exact names - do not fall back to 'general-purpose'):
- gsd-debugger - Diagnoses and fixes issues
</available_agent_types>

<process>

<step name="gather_context">
<!-- State, timestamp, and description pre-injected above via WXP -->

Load current phase context:

```bash
PHASE_INFO=$(pi-gsd-tools roadmap analyze --raw 2>/dev/null || echo "{}")
```

Extract from state JSON:
- `current_phase` - what's being worked on
- `last_activity` - when was the last change
- `milestone` - current milestone name

If `description` is empty, ask the user:
```
What's broken? Describe the symptom in one sentence:
(e.g. "auth tokens expire immediately", "build fails with missing module", "tests pass locally but fail in CI")
```
Store response as `description`.
</step>

<step name="classify_issue">
Classify the issue from the description:

| Symptom pattern | Issue type |
|----------------|------------|
| Error/exception message | `runtime_error` |
| Test failures | `test_failure` |
| Build/compile error | `build_error` |
| Wrong behavior (no error) | `logic_error` |
| Performance problem | `performance` |
| Integration failure | `integration` |
| Unclear | `unknown` |

Set `ISSUE_TYPE`.
</step>

<step name="spawn_debugger">
Display banner:
```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 GSD ► DEBUG SESSION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Issue: {description}
Type:  {ISSUE_TYPE}
Phase: {current_phase || "not set"}

◆ Spawning debugger...
```

Resolve model:
```bash
DEBUGGER_MODEL=$(pi-gsd-tools resolve-model gsd-debugger --raw 2>/dev/null || echo "")
```

Debug prompt:
```markdown
<objective>
Debug the following issue in this project:
{description}

Issue type: {ISSUE_TYPE}
Current phase: {current_phase}
Milestone: {milestone}
</objective>

<files_to_read>
- .planning/STATE.md (Project state and recent decisions)
- .planning/ROADMAP.md (Phase context)
- ./GEMINI.md or ./CLAUDE.md (Project-specific guidelines, if exists)
</files_to_read>

${AGENT_SKILLS_DEBUGGER}

<investigation_protocol>
1. Reproduce: Identify the minimal steps to trigger the issue
2. Isolate: Narrow down to the failing component/file/function
3. Root cause: Identify WHY it fails, not just WHERE
4. Fix: Implement the smallest change that solves the root cause
5. Verify: Confirm the fix works and doesn't introduce regressions

Always check:
- Recent commits (git log --oneline -10) for what changed
- Related files for mismatched interfaces or broken contracts
- Test suite for existing coverage that should have caught this
</investigation_protocol>

<output_format>
## DEBUG COMPLETE

**Root cause:** [one sentence]
**Fix applied:** [what was changed]
**Files modified:** [list]
**Verification:** [how to confirm it's fixed]

OR

## DEBUG BLOCKED

**Investigated:** [what was tried]
**Blocker:** [what additional info is needed]
**Next step:** [what the human should provide or check]
</output_format>
```

```
Task(
  prompt=debug_prompt,
  subagent_type="gsd-debugger",
  model="{DEBUGGER_MODEL}",
  description="Debug: {description}"
)
```
</step>

<step name="handle_return">
**`## DEBUG COMPLETE`:**

Display root cause, fix, and verification steps. Offer:
```
1. Capture as todo (/gsd-add-todo) - if fix not yet applied
2. Continue with current phase (/gsd-execute-phase)
3. Done
```

**`## DEBUG BLOCKED`:**

Display blocker and next steps. Offer:
```
1. Provide additional context and retry
2. Try forensics mode (/gsd-forensics) - deeper investigation
3. Capture as todo and investigate later
```
</step>

<step name="persist_session">
If the debug session produced a fix or useful findings, offer to save:

```bash
mkdir -p .planning/debug
```

Write `.planning/debug/{YYYY-MM-DD}-{slug}.md`:
```markdown
---
created: {timestamp}
issue: {description}
type: {ISSUE_TYPE}
phase: {current_phase}
status: {resolved|blocked}
---

## Root Cause
{root cause summary}

## Fix
{what was changed}

## Verification
{how to confirm}
```

Commit:
```bash
pi-gsd-tools commit "docs: debug session - {description_slug}" --files .planning/debug/{filename}
```
</step>

</process>

<success_criteria>
- [ ] Problem description captured (from arg or prompt)
- [ ] Issue classified by type
- [ ] gsd-debugger spawned with full project context
- [ ] Root cause identified or blocker surfaced
- [ ] Fix applied or next steps clear
- [ ] Session optionally persisted in .planning/debug/
</success_criteria>
