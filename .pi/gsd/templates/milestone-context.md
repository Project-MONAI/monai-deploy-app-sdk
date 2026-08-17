# Milestone Context Template

Template for `.planning/MILESTONE-CONTEXT.md` — captures product scope decisions for an upcoming milestone.

**Purpose:** Document what the milestone should deliver so `/gsd-new-milestone` can start with known intent rather than gathering it inline. Consumed and deleted by `new-milestone` after it generates requirements and a roadmap.

**Key principle:** Product-level only. WHAT users will be able to do — not HOW it will be implemented. Implementation decisions happen in `/gsd-discuss-phase` per phase.

**Downstream consumer:**
- `new-milestone` — reads `<scope>` for feature scoping, `<constraints>` for requirements boundaries, `<success>` to inform success criteria in ROADMAP.md

---

## File Template

```markdown
# Milestone Context

**Gathered:** [date]
**Status:** Ready for /gsd-new-milestone

<milestone_goal>
## Goal

[One sentence: what this milestone delivers for users]

</milestone_goal>

<scope>
## Scope

### In this milestone

- **[Capability name]**: [What users can do — one line]
- **[Capability name]**: [What users can do — one line]

### Explicitly out of scope

- **[Capability name]**: [Reason — "deferred to next milestone", "separate product area", etc.]

[If no explicit exclusions: "No explicit exclusions — boundary is the in-scope list above"]

</scope>

<constraints>
## Constraints

- [Hard constraint — e.g., "no breaking changes to existing API"]
- [Hard constraint — e.g., "must work with existing auth system"]

[If none: "None — unconstrained milestone"]

</constraints>

<success>
## Success Definition

This milestone is successful when:
- [Observable user outcome — something that can be demoed]
- [Observable user outcome]

</success>

<open_questions>
## Open Questions for Planning

- [Question to resolve early in new-milestone or research]

[If none: "None — scope is clear"]

</open_questions>

---

*Milestone context gathered: [date]*
*Run /gsd-new-milestone to start planning*
```

<good_examples>

**Example 1: SaaS product — adding collaboration**

```markdown
# Milestone Context

**Gathered:** 2025-03-15
**Status:** Ready for /gsd-new-milestone

<milestone_goal>
## Goal

Users can invite teammates and collaborate on projects in real time.

</milestone_goal>

<scope>
## Scope

### In this milestone

- **Invite by email**: User can send invites to teammates by email address
- **Role-based access**: Owner, editor, and viewer roles with clear permission boundaries
- **Shared project view**: Teammates see the same project state with live updates
- **Activity feed**: Users can see who changed what and when

### Explicitly out of scope

- **SSO / SAML**: Enterprise auth deferred to v2.0
- **Guest links**: Public sharing without accounts — separate product decision needed

</scope>

<constraints>
## Constraints

- No breaking changes to existing project data model — solo users must not need to migrate
- Invite emails must go through existing SendGrid integration (no new email provider)

</constraints>

<success>
## Success Definition

This milestone is successful when:
- A user can invite a colleague and both see the same project within 60 seconds
- A viewer cannot accidentally edit or delete content

</success>

<open_questions>
## Open Questions for Planning

- Should activity feed be real-time (websocket) or polling? Affects architecture phase ordering.
- What happens to a project if the owner deletes their account?

</open_questions>

---

*Milestone context gathered: 2025-03-15*
*Run /gsd-new-milestone to start planning*
```

**Example 2: CLI tool — v1.1 reliability release**

```markdown
# Milestone Context

**Gathered:** 2025-04-01
**Status:** Ready for /gsd-new-milestone

<milestone_goal>
## Goal

The backup CLI is reliable enough for unattended production use.

</milestone_goal>

<scope>
## Scope

### In this milestone

- **Retry with backoff**: Transient network failures retry automatically, not silently fail
- **Structured logging**: Machine-readable log output for monitoring integration
- **Config file support**: Users can set defaults in a config file, not just flags
- **Dry-run mode**: Users can preview what would be backed up before committing

### Explicitly out of scope

- **Restore command**: Planned for v1.2
- **S3 backend**: Deferred — local filesystem only for now

</scope>

<constraints>
## Constraints

- Must remain backwards compatible with v1.0 flag interface — existing scripts must not break
- No new runtime dependencies (Node built-ins only)

</constraints>

<success>
## Success Definition

This milestone is successful when:
- A backup job can run overnight on a cron without manual intervention
- A failed run produces a log entry that tells an ops engineer exactly what went wrong

</success>

<open_questions>
## Open Questions for Planning

- Should config file use TOML, JSON, or dotenv format? Research common CLI conventions.

</open_questions>

---

*Milestone context gathered: 2025-04-01*
*Run /gsd-new-milestone to start planning*
```

</good_examples>

<guidelines>
**What makes a good MILESTONE-CONTEXT.md:**

Good goal (specific, user-observable):
- "Users can invite teammates and collaborate on projects in real time."
- "The backup CLI is reliable enough for unattended production use."

Bad goal (too vague):
- "Improve collaboration features"
- "Make things more reliable"

Good scope item (user action):
- "User can invite colleagues by email address"
- "Dry-run mode previews changes before committing"

Bad scope item (implementation detail):
- "Add Redis pub/sub for real-time updates"
- "Refactor retry logic in backup module"

**After creation:**
- File lives at `.planning/MILESTONE-CONTEXT.md`
- `new-milestone` reads it in step 2, uses it for requirements scoping, then deletes it
- It does NOT persist — it's a handoff document, not a record
</guidelines>
