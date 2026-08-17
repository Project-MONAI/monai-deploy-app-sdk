<gsd-version v="2.1.4" />

<gsd-arguments>
  <settings><keep-extra-args /></settings>
  <arg name="interactive" type="flag" flag="--interactive" optional />
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
</gsd-execute>

## Context (pre-injected)

**Roadmap:**
<gsd-paste name="roadmap" />

**State:**
<gsd-paste name="state" />

# plan-milestone workflow

Plan all unplanned phases in the current milestone in a single orchestrated session.

---

## Mode Selection (step 0 - always first)

Ask the user ONE binary question:

> **"Should I ask you questions during planning, or plan everything silently and flag doubts at the end?"**
>
> - **Interactive** - I'll ask targeted questions per phase when I hit real ambiguity
> - **Silent** - Plan autonomously; collect flags for review at the end

Store the answer as `MODE` (interactive | silent). Do not ask again.

---

## Phase Discovery

```bash
pi-gsd-tools roadmap analyze --raw
pi-gsd-tools state json --raw
```

Identify all phases with **no PLAN.md files** in their phase directory.
Skip phases already Complete or already planned. Work in roadmap order.

---

## Per-Phase Planning Loop

For each unplanned phase `N`:

### 1. Scope Pre-check

Read `.planning/REQUIREMENTS.md` and the phase entry from ROADMAP.md (goal + success criteria).

Ask internally: *"Does executing this phase risk implementing anything not covered by active requirements, or conflict with what previous phases were meant to deliver?"*

Classify risk:
- **low** - continue silently
- **medium** - log in scope-notes, continue
- **high + interactive** - surface to user before proceeding, ask whether to adjust or continue
- **high + silent** - log prominently, continue, surface in final summary

### 2. Plan the Phase

Invoke:
```
Skill(skill="gsd-plan-phase", args="${N} --skip-research")
```
Unless the phase has no RESEARCH.md yet → drop `--skip-research`.
In **silent** mode, append `--auto` to suppress discussion prompts inside plan-phase.

### 3. Checkpoint

After each phase plan is committed:
```bash
pi-gsd-tools state update current_phase ${N}
```

Announce: `✓ Phase ${N} planned - ${plan_count} plan(s) created`

Check context remaining. If < 25%: stop immediately, emit summary of planned vs remaining phases, suggest `/gsd-plan-milestone --from ${next_unplanned}` to continue.

---

## Final Summary

```
━━ plan-milestone complete ━━━━━━━━━━━━━━━━━━━━━━━
✓ Planned:  [phase list]
⚠ Flags:   [scope notes from high-risk pre-checks]
↳ Next: /gsd-execute-milestone
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

If flags exist and `MODE=interactive`: present them for user review before suggesting execute-milestone.
If flags exist and `MODE=silent`: present all flags together at the end.
