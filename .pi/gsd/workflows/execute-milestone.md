<gsd-version v="2.1.4" />

<gsd-arguments>
  <settings><keep-extra-args /></settings>
  <arg name="from" type="number" optional />
  <arg name="uat-threshold" type="number" optional />
</gsd-arguments>

<gsd-execute>
  <shell command="pi-gsd-tools">
    <args>
      <arg string="roadmap" />
      <arg string="analyze" />
      <arg string="--raw" />
    </args>
    <outs>
      <out type="string" name="roadmap" />
    </outs>
  </shell>
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
  <shell command="pi-gsd-tools">
    <args>
      <arg string="progress" />
      <arg string="json" />
      <arg string="--raw" />
    </args>
    <outs>
      <out type="string" name="progress" />
    </outs>
  </shell>
  <!-- SET _auto_chain_active when milestone execution chain starts -->
  <shell command="pi-gsd-tools">
    <args>
      <arg string="config-set" />
      <arg string="workflow._auto_chain_active" />
      <arg string="true" />
    </args>
    <outs><suppress-errors /></outs>
  </shell>
</gsd-execute>

## Execution Context (pre-injected)

**Roadmap:**
<gsd-paste name="roadmap" />

**State:**
<gsd-paste name="state" />

**Progress:**
<gsd-paste name="progress" />

# execute-milestone workflow

Execute all planned phases in the current milestone with scope guardian, UAT gates, and configurable recovery.

---

## Worktree Check (always first)

```bash
git worktree list
```

If not in an isolated worktree:
> "Large-scale milestone execution should run in an isolated worktree to protect your main branch. Create one now? (y/n, default: y)"

If yes: `Skill(skill="gsd-new-workspace", args="milestone-exec")`, then continue in the new worktree.
If no: warn once, proceed on current branch.

---

## Mode Selection (step 1 - always second)

Ask the user ONE binary question:

> **"How should I behave when I hit a doubt, error, or scope deviation?"**
>
> - **Interactive** - Stop and ask me; I'll guide you through it
> - **Silent** - Try to self-correct; only surface unrecoverable blockers

Store as `MODE` (interactive | silent). Do not ask again.

---

## Phase Discovery

```bash
pi-gsd-tools roadmap analyze --raw
pi-gsd-tools progress json --raw
```

Build execution queue: phases with ≥1 PLAN.md and status ≠ Complete, in roadmap order.

If queue is empty: "All phases are already complete. Run /gsd-audit-milestone." Stop.

---

## Per-Phase Execution Loop

For each pending phase `N`:

### A. Scope Pre-check (lightweight, one LLM call)

Read:
- `.planning/REQUIREMENTS.md`
- Phase goal + success criteria from ROADMAP.md

Prompt (internal): *"Does executing this phase risk implementing anything not covered by active requirements, or conflict with what previous phases delivered? Rate: low / medium / high. One sentence reason."*

- **low** - continue silently
- **medium** - log in scope-log, continue
- **high + interactive** - surface to user, ask: proceed / adjust phase goal / stop
- **high + silent** - log prominently, continue, include in final report

### B. Execute Phase

```
Skill(skill="gsd-execute-phase", args="${N}")
```

### C. Scope Post-audit (full, one LLM call)

Read new SUMMARY.md files from the phase directory.

Check:
1. **Undelivered must-haves** - PLAN.md `must_haves` entries absent from SUMMARY
2. **Scope creep** - files modified that are outside this phase's stated scope
3. **Requirement drift** - work done that has no matching REQUIREMENTS entry

Classify result as `SCOPE_STATUS`:
- **clean** - continue
- **drift** - log + warn, continue
- **violation** - trigger recovery (see §F)

### D. Verify

```
Skill(skill="gsd-verify-work", args="${N}")
```

Compute UAT pass rate = passing items / total items.
Default threshold: **80%**. Override with `--uat-threshold N`.

### E. Gate Check

| Condition                 | Interactive                    | Silent                 |
| ------------------------- | ------------------------------ | ---------------------- |
| UAT pass rate < threshold | Ask: fix gaps now or continue? | → Recovery loop        |
| Context remaining < 20%   | Warn, ask: stop or continue?   | → Write HANDOFF, stop  |
| SCOPE_STATUS = violation  | Surface details, ask           | → Recovery loop        |
| All gates pass            | Continue to checkpoint         | Continue to checkpoint |

### F. Recovery Loop

When triggered:

```
1. pi-gsd-tools validate health --repair
2. Self-correct: identify root cause, patch, re-run verification
3. Re-check gates
4. Gates pass → continue to checkpoint
5. Still failing:
   - Interactive: explain issue, ask user how to resolve, loop from step 2
   - Silent: write HANDOFF files (see §G), stop
```

### G. Hard Stop - HANDOFF Files

On unrecoverable stop, write two files matching original GSD pause-work convention:

**`.planning/HANDOFF.json`** (machine-readable, consumed by `/gsd-resume-work`):
```json
{
  "stopped_at": "ISO-timestamp",
  "phase": "N",
  "phase_name": "phase name",
  "stop_reason": "uat_failure | scope_violation | context_exhausted | unrecoverable_error | audit_exhausted | audit_no_result",
  "outer_cycles": 0,
  "gaps_store": [],
  "debt_store": [],
  "gaps_phases_tried": [],
  "debt_phases_tried": [],
  "uat_pass_rate": 75,
  "scope_status": "violation",
  "phases_completed": ["1", "2", "3"],
  "phases_remaining": ["N", "N+1"],
  "scope_log": ["note 1", "note 2"],
  "next_action": "Run /gsd-execute-milestone --from N to resume"
}
```

**`.planning/phases/NN-name/.continue-here.md`** (human-readable):
```markdown
---
phase: N
status: stopped
stop_reason: [reason]
last_updated: [timestamp]
---

## What happened
[Clear explanation of why execution stopped]

## State at stop
- UAT pass rate: X%
- Scope status: [clean/drift/violation]
- Scope notes: [any flags]

## How to resume
Run: /gsd-execute-milestone --from N
Or fix the specific issue first: [specific suggestion]
```

### H. Checkpoint (on success)

```bash
pi-gsd-tools state update current_phase ${N}
pi-gsd-tools state update last_activity $(date -u +%Y-%m-%d)
pi-gsd-tools commit "chore: complete phase ${N}" --files .planning/
```

Announce: `✓ Phase ${N} complete - UAT: ${pass_rate}%  Scope: ${scope_status}`

---

## After All Phases - Mode Split

### Interactive mode

Do NOT auto-invoke the lifecycle. Surface the execution summary and hand back:

```
━━ execute-milestone: all phases done ━━━━━━━━━━━━
✓ Phases:   ${done}/${total} complete
📊 Avg UAT: ${avg_uat}%
⚠ Scope:   ${scope_flag_count} flag(s) (details above)

Next: /gsd-audit-milestone when you are ready to review
      and close the milestone.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

Stop. The user owns the audit decision.

---

### Silent mode - Auto Lifecycle

Only in silent mode. Display transition banner:
```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 All phases complete → lifecycle: audit → complete → cleanup
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

Read config once:
- `config.workflow.auto_retry_audit` (default: `true`)
- `config.workflow.auto_retry_audit_budget` (default: `1`)
- `config.workflow.auto_retry_tech_debt` (default: `true`)
- `config.workflow.auto_retry_tech_debt_budget` (default: `1`)

Initialise accumulators (persist across the outer loop):
- `gaps_store = []` - unsatisfied requirements not yet resolved
- `debt_store = []` - tech debt items not yet resolved
- `gaps_phases_tried = []` - inserted phases attempted for gap closure
- `debt_phases_tried = []` - inserted phases attempted for debt resolution
- `outer_cycles = 0`

---

#### OUTER LOOP - Full audit cycle

`LABEL: outer_loop`

**Step A - Full audit**

```
Skill(skill="gsd-audit-milestone")
```

If no result / malformed → Write HANDOFF (§G), stop.

Extract from audit result:
- `current_gaps[]` - unsatisfied requirement IDs + affected phase numbers
- `current_debt[]` - tech debt items + affected phase numbers

If both empty → AUDIT PASSED. Proceed to Step D (complete).

---

**Step B - Gap closure loop** (only if `current_gaps` non-empty)

If `auto_retry_audit = false`: add `current_gaps` to `gaps_store`, skip to Step C.

While `current_gaps` non-empty and `auto_retry_audit_budget > 0`:

```
1. Decrement auto_retry_audit_budget
2. Insert a gap-closure phase (decimal after last phase):
   Skill(skill="gsd-insert-phase", args="${last_phase}.${gap_cycle} 'Gap closure: ${gap_summary}'")
3. Plan it with gap context:
   Skill(skill="gsd-plan-phase", args="${new_phase} --auto")
   (Pass unsatisfied requirement details as planning context)
4. Execute it:
   Skill(skill="gsd-execute-phase", args="${new_phase}")
5. Track: append new_phase to gaps_phases_tried
6. Targeted re-audit - affected phases only:
   Skill(skill="gsd-audit-milestone", args="--phases ${gap_affected_phases}")
7. Re-read current_gaps from result
   - Resolved? current_gaps = [], break loop
   - Reduced? loop again if budget > 0
   - Same/worse? loop again if budget > 0
```

After loop:
- If `current_gaps` empty → gaps resolved ✅, `gaps_store = []`
- If still non-empty → `gaps_store = current_gaps` (budget exhausted or disabled)

---

**Step C - Tech debt loop** (only if `current_debt` non-empty)

If `auto_retry_tech_debt = false`: add `current_debt` to `debt_store`, skip to final gate.

While `current_debt` non-empty and `auto_retry_tech_debt_budget > 0`:

```
1. Decrement auto_retry_tech_debt_budget
2. Insert a debt-resolution phase (decimal after last phase):
   Skill(skill="gsd-insert-phase", args="${last_phase}.${debt_cycle} 'Tech debt: ${debt_summary}'")
3. Plan it with debt context:
   Skill(skill="gsd-plan-phase", args="${new_phase} --auto")
   (Pass tech debt item details as planning context)
4. Execute it:
   Skill(skill="gsd-execute-phase", args="${new_phase}")
5. Track: append new_phase to debt_phases_tried
6. Targeted re-audit - affected phases only:
   Skill(skill="gsd-audit-milestone", args="--phases ${debt_affected_phases}")
7. Re-read current_debt from result
   - Resolved? current_debt = [], break loop
   - Reduced? loop again if budget > 0
   - gaps_found in re-audit? add to gaps_store, break (handled in next outer cycle)
```

After loop:
- If `current_debt` empty → debt resolved ✅, `debt_store = []`
- If still non-empty → `debt_store = current_debt`

---

**Final gate (end of outer loop body)**

If `gaps_store` empty AND `debt_store` empty:
→ AUDIT CLEAN. Proceed to Step D.

If anything remains in stores:
- Increment `outer_cycles`
- If outer budget remaining (derived from max of both budgets > 0):
  → `GOTO outer_loop` (re-run full audit from top, fresh eyes)
- If exhausted:
  → Write enriched HANDOFF (§G), stop.

---

#### Step D - Complete Milestone

```
Skill(skill="gsd-complete-milestone", args="${milestone_version}")
```

Verify archive produced:
```bash
ls .planning/milestones/v${milestone_version}-ROADMAP.md 2>/dev/null || true
```
If absent → Write HANDOFF, stop. Message: "complete-milestone did not produce archive files."

#### Step E - Cleanup

```
Skill(skill="gsd-cleanup")
```

Cleanup handles its own dry-run and confirmation internally.

---

## Worktree Merge (both modes, after lifecycle or summary)

If running in an isolated worktree, ask:
> "Merge this worktree back to your main branch? (y/n, default: y)"

If yes:
```bash
git checkout main
git merge --no-ff milestone-exec -m "feat: complete milestone ${milestone_version}"
git worktree remove milestone-exec
```

If no: leave the worktree open. Tell the user how to merge manually.

---

## Final Banner (silent mode only, after full lifecycle)

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 GSD ► EXECUTE-MILESTONE ▸ COMPLETE 🎉
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

 Milestone: ${milestone_version} - ${milestone_name}
 Phases:    ${done}/${total} complete
 Avg UAT:   ${avg_uat}%
 Lifecycle: audit ✅ → complete ✅ → cleanup ✅

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```
