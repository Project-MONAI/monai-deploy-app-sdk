<gsd-version v="2.1.4" />

<gsd-arguments>
  <settings>
    <keep-extra-args />
  </settings>
  <arg name="auto" type="flag" flag="--auto" optional />
</gsd-arguments>

<gsd-execute>
  <shell command="pi-gsd-tools">
    <args>
      <arg string="init" />
      <arg string="milestone-op" />
    </args>
    <outs>
      <out type="string" name="init" />
    </outs>
  </shell>
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

## Context (pre-injected)

**Init:**
<gsd-paste name="init" />

**Project State:**
<gsd-paste name="state" />

**Current Roadmap:**
<gsd-paste name="roadmap" />

---

<purpose>
Crystallize what the next milestone should deliver before starting the planning machinery. You are a thinking partner — a PM who knows what shipped, asks smart questions, and helps the user clarify product scope before committing to a roadmap.

Output: `.planning/MILESTONE-CONTEXT.md`, consumed by /gsd-new-milestone.

Optional step — /gsd-new-milestone works without it. The value is separating the "what do we build?" conversation from the requirements and roadmapping machinery.
</purpose>

<philosophy>
**User = product owner. Agent = PM/advisor.**

The user knows:
- What users are struggling with
- What the next logical product step is
- What MUST ship vs nice-to-have
- Any hard constraints (tech, team, timeline)

The user doesn't need to define:
- How to structure phases (that's the roadmapper)
- Implementation approach (that's research + discuss-phase)
- Which requirements to write (that's new-milestone)

Your job: help the user articulate a clear, scoped milestone intent that new-milestone can turn into requirements and a roadmap.
</philosophy>

<scope_guardrail>
**Product-level only.** This discussion is about WHAT the milestone delivers, not HOW.

**Allowed:**
- "Should we tackle X or defer it?"
- "What's the must-have vs nice-to-have split?"
- "Any hard constraints for this cycle?"
- "How will we know this milestone is done?"

**Not here:**
- "Should we use Redis or Postgres for this?"
- "Which architecture pattern?"
- "How should we structure the phases?"

If the user goes implementation-level, redirect:
```
"That's a planning question — /gsd-new-milestone and /gsd-discuss-phase will handle it.
For now: do you want [capability] in scope for this milestone?"
```
</scope_guardrail>

<answer_validation>
After every AskUserQuestion call, check if the response is empty or whitespace-only. If so:
1. Retry once with the same parameters
2. If still empty, present options as a plain-text numbered list

**Text mode (`workflow.text_mode: true` or `--text` flag):**
Replace ALL AskUserQuestion calls with plain-text numbered lists. User types a number.
Required for Claude Code remote sessions where TUI menus don't forward.
</answer_validation>

<process>

## 1. Initialize

<!-- Context pre-injected above via WXP -->

Parse init JSON for: `commit_docs`, `context_window`, `milestone_version`, `milestone_name`, `last_completed_milestone`, `roadmap_exists`, `state_exists`.

**If `state_exists` is false:**
```
No .planning/ directory found. Set up a project first:

/gsd-new-project
```
Exit workflow.

Read project files:
```bash
cat .planning/PROJECT.md 2>/dev/null || true
cat .planning/MILESTONES.md 2>/dev/null || true
```

Extract from PROJECT.md: project name, core value, non-negotiables, target users.
Extract from MILESTONES.md: what shipped in completed milestones (summaries, not full detail).

**Read text mode config:**
```bash
TEXT_MODE=$(pi-gsd-tools config-get workflow.text_mode 2>/dev/null || echo "false")
```
Enable text mode if `--text` in $ARGUMENTS OR `TEXT_MODE` is `true`.

## 2. Check Existing MILESTONE-CONTEXT.md

```bash
test -f .planning/MILESTONE-CONTEXT.md && echo "exists" || echo "absent"
```

**If exists:**

**If `--auto`:** Load existing content, continue to step 3 to refresh it. Log: `[auto] Existing MILESTONE-CONTEXT.md found — refreshing.`

**Otherwise,** use AskUserQuestion:
- header: "Context exists"
- question: "MILESTONE-CONTEXT.md already exists. What do you want to do?"
- options:
  - "Update it" — Revise and improve existing context
  - "View it" — Show current content, then decide
  - "Skip" — Use as-is, go straight to /gsd-new-milestone

If "View": display file contents, then re-ask "Update it" / "Skip".
If "Skip": display `Next: /gsd-new-milestone` and exit.
If "Update": load existing content, continue to step 3.

**If absent:** Continue to step 3.

## 3. Retrospective Framing

Sets the context for "what's next" based on what shipped.

**If `last_completed_milestone` is not null:**

Read the matching section in `.planning/MILESTONES.md` for `last_completed_milestone.version`.

Display (no user input needed):
```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 Last milestone: [version] — [name]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[2-3 sentence summary of what shipped from MILESTONES.md]
```

Then ask ONE freeform question (plain text, NOT AskUserQuestion):

> "What feedback or signals are shaping what you want to build next?"

Wait for response. Use the answer to seed the scope discussion. Do not ask follow-ups from this — carry the insight forward.

**If `--auto`:** Skip retrospective question. Read STATE.md accumulated context for any signals.

**If no completed milestones:** Skip retrospective entirely. Continue to step 4.

## 4. Gather Milestone Intent

Open question to surface rough direction before structuring.

Ask (plain text, NOT AskUserQuestion):

> "What do you want this milestone to deliver? Give me the rough picture — we'll tighten the scope next."

Wait for response. Parse it for:
- Feature/capability mentions → candidates for scope-in
- Exclusions or "not yet" signals → candidates for scope-out
- Urgency or priority cues
- Any constraints mentioned in passing

Reflect back in 2-3 sentences:
```
"So the core of this milestone is [X]. You also mentioned [Y],
and [Z] sounds like a natural boundary. Is that the right picture?"
```

If they confirm: proceed to step 5 with extracted candidates.
If they adjust: incorporate and reflect again. Max 2 loops, then proceed.

**If `--auto`:** Skip reflection loop. Extract candidates from the intent statement directly and proceed.

## 5. Scope Discussion

Turn the rough intent into a clear in/out split.

**Build candidate list** from:
- Step 4's response (feature/capability mentions)
- STATE.md accumulated context (pending items, blockers noted)
- MILESTONES.md "Future Requirements" or deferred items from last milestone
- Any items in `.planning/REQUIREMENTS.md` marked Out of Scope that might be reconsidered

Group related candidates into clusters (2-4 features per cluster). Present one cluster at a time.

**For each cluster:**

If text mode: present as numbered list with multi-select.
Otherwise use AskUserQuestion (multiSelect: true):
- header: "Scope: [cluster]" (max 12 chars)
- question: "Which of these belong in this milestone?"
- options: each candidate with a 1-line description

After all clusters, show a running tally:
```
Scoped in:  [N] capabilities
Deferred:   [M] capabilities
```

**Explicit exclusions** — after clusters are done:

If text mode: ask as plain-text.
Otherwise use AskUserQuestion:
- header: "Out of scope"
- question: "Anything to explicitly exclude — even if it seems related?"
- options:
  - "Nothing to add — the scope list covers it"
  - "Yes, I want to explicitly exclude something"

If "Yes": ask them to name it (plain text). Capture with reason.

**If `--auto`:** Include everything mentioned in step 4's intent or with a clear priority signal. Exclude only items the user explicitly flagged as "not now" or "next milestone".

## 6. Constraints

Anything that bounds how this milestone must be shaped.

If text mode: present as numbered multi-select list.
Otherwise use AskUserQuestion (multiSelect: true):
- header: "Constraints"
- question: "Any hard constraints for this milestone?"
- options:
  - "No breaking changes — existing integrations must keep working"
  - "No new external dependencies"
  - "Must maintain backwards compatibility with existing data"
  - "Performance budget — no regressions on key metrics"
  - "None — this milestone is unconstrained"
  - "Other — let me describe it"

If "Other": ask plain text, record result.

**If `--auto`:** Default to "None" unless PROJECT.md or STATE.md explicitly mentions active constraints.

## 7. Success Definition

How does "done" look from the outside?

Ask (plain text, NOT AskUserQuestion):

> "Finish this sentence: this milestone is a success when users can ___."

Wait for response. Parse 1-3 observable outcomes.

If the response is vague ("when everything works", "when it's polished"), prompt once:

> "What's a concrete user action that proves it — something you could demo?"

Capture outcomes. If they list more than 3, keep the 3 most concrete and user-observable.

**If `--auto`:** Derive success outcomes from scoped capabilities — "user can [primary action]" for each major in-scope cluster.

## 8. Open Questions

Surface anything that needs resolving in new-milestone or early research — not as a blocker, but as a signal.

Ask (plain text):

> "Any open questions or risks you want the planning session to address early?"

If they say "no" or give nothing: record "None — scope is clear."
If they give questions: capture them for the MILESTONE-CONTEXT.md open_questions section.

**If `--auto`:** Skip. Default to "None."

## 9. Write MILESTONE-CONTEXT.md

Write to `.planning/MILESTONE-CONTEXT.md`:

```markdown
# Milestone Context

**Gathered:** [ISO date]
**Status:** Ready for /gsd-new-milestone

<milestone_goal>
## Goal

[One sentence distilled from step 4 — what this milestone delivers for users]

</milestone_goal>

<scope>
## Scope

### In this milestone

[For each scoped-in capability, in priority order:]
- **[Capability name]**: [1-line description of what it means for users]

### Explicitly out of scope

[For each explicit exclusion:]
- **[Capability name]**: [reason — "deferred to next milestone", "separate product area", etc.]

[If no explicit exclusions: "No explicit exclusions — boundary is the in-scope list above"]

</scope>

<constraints>
## Constraints

[For each constraint from step 6:]
- [Constraint statement]

[If none: "None — unconstrained milestone"]

</constraints>

<success>
## Success Definition

This milestone is successful when:
- [Observable user outcome 1]
- [Observable user outcome 2]
[- [Observable user outcome 3] — only if genuinely distinct]

</success>

<open_questions>
## Open Questions for Planning

[Questions from step 8 that new-milestone or early research should address:]
- [Question]

[If none: "None — scope is clear"]

</open_questions>

---

*Milestone context gathered: [date]*
*Run /gsd-new-milestone to start planning*
```

## 10. Commit

```bash
pi-gsd-tools commit "docs: capture milestone context" --files .planning/MILESTONE-CONTEXT.md
```

If `commit_docs` is false: skip commit silently.

## 11. Present Summary and Next Steps

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 GSD ► MILESTONE CONTEXT CAPTURED ✓
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**Goal:** [one sentence from context file]

**In scope ([N] capabilities):**
[bullet list]

**Constraints:** [list or "none"]
**Success:** [first observable outcome]

───────────────────────────────────────────────────────────────

## ▶ Next Up

**Start milestone planning** — requirements, research, roadmap

`/gsd-new-milestone`

<sub>`/new` first → fresh context window</sub>

───────────────────────────────────────────────────────────────
```

## 12. Auto-Advance

1. Parse `--auto` flag from $ARGUMENTS.
2. Sync chain flag with intent — clear if not in `--auto` run:
   ```bash
   if [[ ! "$ARGUMENTS" =~ --auto ]]; then
     pi-gsd-tools config-set workflow._auto_chain_active false 2>/dev/null
   fi
   ```
3. Read chain flag and config:
   ```bash
   AUTO_CHAIN=$(pi-gsd-tools config-get workflow._auto_chain_active 2>/dev/null || echo "false")
   AUTO_CFG=$(pi-gsd-tools config-get workflow.auto_advance 2>/dev/null || echo "false")
   ```

**If `--auto` OR `AUTO_CHAIN` is true OR `AUTO_CFG` is true:**

Display:
```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 GSD ► AUTO-ADVANCING TO NEW-MILESTONE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Context captured. Launching new-milestone...
```

Launch:
```
Skill(skill="gsd-new-milestone", args="--auto ${GSD_WS}")
```

Handle return:
- **MILESTONE INITIALIZED** → Display success banner, done.
- **CHECKPOINT / BLOCKED** → Stop chain, show: `Continue: /gsd-new-milestone ${GSD_WS}`

**If not auto:** Step 11 already shown. Done.

</process>

<success_criteria>
- [ ] .planning/ exists (state_exists check)
- [ ] Existing MILESTONE-CONTEXT.md handled (update/view/skip)
- [ ] Last completed milestone surfaced for retrospective framing
- [ ] Milestone intent gathered via open conversation
- [ ] Scope in/out defined per capability cluster
- [ ] Hard constraints captured
- [ ] Success definition captured as observable user outcomes
- [ ] Open questions captured for planning session
- [ ] MILESTONE-CONTEXT.md written to .planning/
- [ ] Committed (if commit_docs)
- [ ] User knows next step: /gsd-new-milestone
</success_criteria>
