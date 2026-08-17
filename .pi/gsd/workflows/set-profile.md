<gsd-version v="2.1.4" />

<gsd-arguments>
  <settings>
    <keep-extra-args />
  </settings>
  <arg name="profile" type="string" optional />
</gsd-arguments>

<gsd-execute>
  <display msg="Loading model profile config..." />
  <shell command="pi-gsd-tools">
    <args>
      <arg string="config-get" />
      <arg string="workflow" />
      <arg string="--raw" />
    </args>
    <outs>
      <suppress-errors />
      <out type="string" name="workflow-config" />
    </outs>
  </shell>
  <json-parse src="workflow-config" out="current-profile" path="$.model_profile" />
</gsd-execute>

## Profile Context (pre-injected by WXP)

**Requested profile:** <gsd-paste name="profile" />

**Current profile:** <gsd-paste name="current-profile" />

**Full workflow config:**
<gsd-paste name="workflow-config" />

---

<purpose>
Set the active model profile, controlling which Claude model each GSD agent uses.
One command replaces the full /gsd-settings flow for the common case of switching profiles.

Profiles: `quality` | `balanced` | `budget` | `inherit`
</purpose>

<profile_reference>

| Profile | Planner | Executor | Researcher | Verifier | Use when |
|---------|---------|---------|------------|---------|---------|
| `quality` | opus | opus | opus | sonnet | Critical architecture, quota available |
| `balanced` | opus | sonnet | sonnet | sonnet | Normal development (default) |
| `budget` | sonnet | sonnet | haiku | haiku | Conserving quota, high-volume work |
| `inherit` | inherit | inherit | inherit | inherit | OpenRouter / local models / session switching |

</profile_reference>

<process>

<step name="resolve_profile">
<!-- Profile arg and current config pre-injected above via WXP -->

**If `profile` is empty (no argument provided):**

Show current profile and available options:

```
## Current Model Profile

Active: {current-profile || "balanced (default)"}

Available profiles:
  quality   - Opus everywhere (highest quality, highest cost)
  balanced  - Opus for planning, Sonnet for execution (recommended)
  budget    - Sonnet/Haiku mix (lowest cost)
  inherit   - Use the current session model for all agents

Usage: /gsd-set-profile <profile>

To configure individual agents and other settings: /gsd-settings
```

Exit (display only, no changes).
</step>

<step name="validate_profile">
Validate that the provided `profile` value is one of: `quality`, `balanced`, `budget`, `inherit`.

**If invalid:**
```
Error: Unknown profile '{profile}'.

Valid profiles: quality, balanced, budget, inherit

Example: /gsd-set-profile balanced
```
Exit.
</step>

<step name="apply_profile">
Apply the profile:

```bash
pi-gsd-tools config-set-model-profile {profile}
```

This updates `.planning/config.json` with the new model profile.
</step>

<step name="confirm">
```
✓ Model profile set to: {profile}

{profile description}

  Planner:     {model}
  Executor:    {model}
  Researcher:  {model}
  Verifier:    {model}

To configure more options: /gsd-settings
```

Where "profile description" maps:
- `quality` → "Maximum reasoning power. Opus for all decision-making agents."
- `balanced` → "Smart allocation. Opus for planning, Sonnet for execution and verification."
- `budget` → "Minimal Opus usage. Sonnet for writing, Haiku for research and verification."
- `inherit` → "Follow the current session model. Required for non-Anthropic providers."
</step>

</process>

<success_criteria>
- [ ] Profile validated against allowed values
- [ ] .planning/config.json updated with new profile
- [ ] Confirmation shows per-agent model assignments
- [ ] Link to /gsd-settings for advanced configuration
</success_criteria>
