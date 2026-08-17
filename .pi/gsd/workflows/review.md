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
Cross-AI peer review - invoke external AI CLIs to independently review phase plans.
Each CLI gets the same prompt (PROJECT.md context, phase plans, requirements) and
produces structured feedback. Results are combined into REVIEWS.md for the planner
to incorporate via --reviews flag.

This implements adversarial review: different AI models catch different blind spots.
A plan that survives review from 2-3 independent AI systems is more robust.
</purpose>

<process>

<step name="detect_clis">
Check which AI CLIs are available on the system:

<!-- Context pre-injected above via WXP - variables available via <gsd-paste name="..."> -->

Read from init: `phase_dir`, `phase_number`, `padded_phase`.

Then read:
1. `.planning/PROJECT.md` (first 80 lines - project context)
2. Phase section from `.planning/ROADMAP.md`
3. All `*-PLAN.md` files in the phase directory
4. `*-CONTEXT.md` if present (user decisions)
5. `*-RESEARCH.md` if present (domain research)
6. `.planning/REQUIREMENTS.md` (requirements this phase addresses)
</step>

<step name="build_prompt">
Build a structured review prompt:

```markdown
# Cross-AI Plan Review Request

You are reviewing implementation plans for a software project phase.
Provide structured feedback on plan quality, completeness, and risks.

## Project Context
{first 80 lines of PROJECT.md}

## Phase {N}: {phase name}
### Roadmap Section
{roadmap phase section}

### Requirements Addressed
{requirements for this phase}

### User Decisions (CONTEXT.md)
{context if present}

### Research Findings
{research if present}

### Plans to Review
{all PLAN.md contents}

## Review Instructions

Analyze each plan and provide:

1. **Summary** - One-paragraph assessment
2. **Strengths** - What's well-designed (bullet points)
3. **Concerns** - Potential issues, gaps, risks (bullet points with severity: HIGH/MEDIUM/LOW)
4. **Suggestions** - Specific improvements (bullet points)
5. **Risk Assessment** - Overall risk level (LOW/MEDIUM/HIGH) with justification

Focus on:
- Missing edge cases or error handling
- Dependency ordering issues
- Scope creep or over-engineering
- Security considerations
- Performance implications
- Whether the plans actually achieve the phase goals

Output your review in markdown format.
```

Write to a temp file: `/tmp/gsd-review-prompt-{phase}.md`
</step>

<step name="invoke_reviewers">
For each selected CLI, invoke in sequence (not parallel - avoid rate limits):

**Gemini:**
```bash
gemini -p "$(cat /tmp/gsd-review-prompt-{phase}.md)" 2>/dev/null > /tmp/gsd-review-gemini-{phase}.md
```

**the agent (separate session):**
```bash
claude -p "$(cat /tmp/gsd-review-prompt-{phase}.md)" --no-input 2>/dev/null > /tmp/gsd-review-claude-{phase}.md
```

**Codex:**
```bash
codex exec --skip-git-repo-check "$(cat /tmp/gsd-review-prompt-{phase}.md)" 2>/dev/null > /tmp/gsd-review-codex-{phase}.md
```

If a CLI fails, log the error and continue with remaining CLIs.

Display progress:
```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 GSD ► CROSS-AI REVIEW - Phase {N}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

◆ Reviewing with {CLI}... done ✓
◆ Reviewing with {CLI}... done ✓
```
</step>

<step name="write_reviews">
Combine all review responses into `{phase_dir}/{padded_phase}-REVIEWS.md`:

```markdown
---
phase: {N}
reviewers: [gemini, claude, codex]
reviewed_at: {ISO timestamp}
plans_reviewed: [{list of PLAN.md files}]
---

# Cross-AI Plan Review - Phase {N}

## Gemini Review

{gemini review content}

---

## the agent Review

{claude review content}

---

## Codex Review

{codex review content}

---

## Consensus Summary

{synthesize common concerns across all reviewers}

### Agreed Strengths
{strengths mentioned by 2+ reviewers}

### Agreed Concerns
{concerns raised by 2+ reviewers - highest priority}

### Divergent Views
{where reviewers disagreed - worth investigating}
```

Commit:
```bash
pi-gsd-tools commit "docs: cross-AI review for phase {N}" --files {phase_dir}/{padded_phase}-REVIEWS.md
```
</step>

<step name="present_results">
Display summary:

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 GSD ► REVIEW COMPLETE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Phase {N} reviewed by {count} AI systems.

Consensus concerns:
{top 3 shared concerns}

Full review: {padded_phase}-REVIEWS.md

To incorporate feedback into planning:
  /gsd-plan-phase {N} --reviews
```

Clean up temp files.
</step>

</process>

<success_criteria>
- [ ] At least one external CLI invoked successfully
- [ ] REVIEWS.md written with structured feedback
- [ ] Consensus summary synthesized from multiple reviewers
- [ ] Temp files cleaned up
- [ ] User knows how to use feedback (/gsd-plan-phase --reviews)
</success_criteria>
