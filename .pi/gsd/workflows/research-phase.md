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
      <arg string="roadmap" />
      <arg string="get-phase" />
    </args>
    <outs>
      <out type="string" name="roadmap-phase" />
    </outs>
  </shell>
  <shell command="pi-gsd-tools">
    <args>
      <arg string="agent-skills" />
      <arg string="gsd-phase-researcher" />
    </args>
    <outs>
      <suppress-errors />
      <out type="string" name="agent-skills-researcher" />
    </outs>
  </shell>
</gsd-execute>

## Context (pre-injected)

**Phase:** <gsd-paste name="phase" />

**Phase Data:**
<gsd-paste name="init" />

**Roadmap:**
<gsd-paste name="roadmap-phase" />

<purpose>
Research how to implement a phase. Spawns gsd-phase-researcher with phase context.

Standalone research command. For most workflows, use `/gsd-plan-phase` which integrates research automatically.
</purpose>

<available_agent_types>
Valid GSD subagent types (use exact names - do not fall back to 'general-purpose'):
- gsd-phase-researcher - Researches technical approaches for a phase
</available_agent_types>

<process>

## Step 0: Resolve Model Profile

@.pi/gsd/references/model-profile-resolution.md

Resolve model for:
- `gsd-phase-researcher`

## Step 1: Normalize and Validate Phase

@.pi/gsd/references/phase-argument-parsing.md

<!-- Context pre-injected above via WXP - variables available via <gsd-paste name="..."> -->

## Step 4: Spawn Researcher

```
Task(
  prompt="<objective>
Research implementation approach for Phase {phase}: {name}
</objective>

<files_to_read>
- {context_path} (USER DECISIONS from /gsd-discuss-phase)
- {requirements_path} (Project requirements)
- {state_path} (Project decisions and history)
</files_to_read>

${AGENT_SKILLS_RESEARCHER}

<additional_context>
Phase description: {description}
</additional_context>

<output>
Write to: .planning/phases/${PHASE}-{slug}/${PHASE}-RESEARCH.md
</output>",
  subagent_type="gsd-phase-researcher",
  model="{researcher_model}"
)
```

## Step 5: Handle Return

- `## RESEARCH COMPLETE` - Display summary, offer: Plan/Dig deeper/Review/Done
- `## CHECKPOINT REACHED` - Present to user, spawn continuation
- `## RESEARCH INCONCLUSIVE` - Show attempts, offer: Add context/Try different mode/Manual

</process>
