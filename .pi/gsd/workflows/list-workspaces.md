<gsd-version v="2.1.4" />

<gsd-arguments>
  <settings><keep-extra-args /></settings>
</gsd-arguments>

<gsd-execute>
  <shell command="pi-gsd-tools">
    <args>
      <arg string="init" />
      <arg string="list-workspaces" />
    </args>
    <outs>
      <out type="string" name="workspaces-data" />
    </outs>
  </shell>
  <if>
    <condition>
      <starts-with>
        <left name="workspaces-data" />
        <right type="string" value="@file:" />
      </starts-with>
    </condition>
    <then>
      <string-op op="split">
        <args>
          <arg name="workspaces-data" />
          <arg type="string" value="@file:" />
        </args>
        <outs>
          <out type="string" name="workspaces-data-file" />
        </outs>
      </string-op>
      <shell command="cat">
        <args>
          <arg name="workspaces-data-file" wrap='"' />
        </args>
        <outs>
          <out type="string" name="workspaces-data" />
        </outs>
      </shell>
    </then>
  </if>
</gsd-execute>

## Workspaces (pre-injected)

<gsd-paste name="workspaces-data" />

<purpose>
List all GSD workspaces found in ~/gsd-workspaces/ with their status.
</purpose>

<required_reading>
Read all files referenced by the invoking prompt's execution_context before starting.
</required_reading>

<process>

## 1. Setup

<!-- Context pre-injected above via WXP - variables available via <gsd-paste name="..."> -->

Parse JSON for: `workspace_base`, `workspaces`, `workspace_count`.

## 2. Display

**If `workspace_count` is 0:**

```
No workspaces found in ~/gsd-workspaces/

Create one with:
  /gsd-new-workspace --name my-workspace --repos repo1,repo2
```

Done.

**If workspaces exist:**

Display a table:

```
GSD Workspaces (~/gsd-workspaces/)

| Name      | Repos | Strategy | GSD Project |
| --------- | ----- | -------- | ----------- |
| feature-a | 3     | worktree | Yes         |
| feature-b | 2     | clone    | No          |

Manage:
  cd ~/gsd-workspaces/<name>     # Enter a workspace
  /gsd-remove-workspace <name>   # Remove a workspace
```

For each workspace, show:
- **Name** - directory name
- **Repos** - count from init data
- **Strategy** - from WORKSPACE.md
- **GSD Project** - whether `.planning/PROJECT.md` exists (Yes/No)

</process>
