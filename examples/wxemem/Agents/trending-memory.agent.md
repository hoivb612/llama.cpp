---
name: trending-memory
description: Generate overlaid Kennan WXE before-PnP, post-PnP, and sign-in physical-memory trends from daily wxemem JSON captures.
---

# Trending memory usage

Run `Generate-MemoryTrend.ps1` alongside this file to regenerate the report.
Default inputs are `..\daily`; default outputs are `output\memory-trend.*`.
This directory holds the agent instructions and implementation; custom-agent
discovery depends on the agent host's configuration.

```powershell
& 'C:\llama.cpp\wxe\Agents\Generate-MemoryTrend.ps1'
```

## Scope and metric

- Only versioned build folders matching `major.minor.yyMMdd-HHmm` under `daily`.
- If supplied, use `-BuildList` to restrict the report to the listed build IDs.
  Without it, include all versioned builds.
- Only `Kennan\WXE`, not other devices or operating systems.
- `beforepnp`, `prepnp*` (including `prepnpxboxapp`), and historical `nopnp`
  are equivalent baseline stages; Xbox app is installed by default.
- Overlay a second line for `postpnp*` (including `postpnpxboxapp` and
  `postpnpxboxpcapp`), `afterpnp`, and historical `pnp`.
- Overlay a third line for exactly `xboxappsignin`, `signin`, or `postsignin`.
  Do not include `presignin` or Steam sign-in stages.
- Only read wxemem JSON captures directly in those stage directories (`.json`
  or JSON-named `.txt` files). Do not recurse into nested folders, including
  Houseman and Omni. A stage with only nested captures remains a reported gap.
- Exclude Steam and other stages outside those patterns.
- Read exactly `physical.used_bytes` from wxemem JSON, not commit, private WS,
  or a sum of process working sets. Display GiB (bytes / 1,073,741,824).
- Order by the build timestamp encoded in the directory, not file modification
  time. The graph labels are **build dates**, not capture dates. Include every
  selected build as a shared category. Each line has independent gaps for
  missing captures, never zero or substitution from the other stage.
- If multiple eligible JSON captures exist for one build and stage, stop and ask which
  to select. Do not average them or silently choose the newest.
- Malformed JSON and invalid memory metrics are errors, not missing samples.
- Do not modify any source captures.

## Output and reporting

The PowerPoint contains one slide with three overlaid series on the same GiB axis:
blue circles for before-PnP, orange squares for post-PnP, green triangles for
sign-in, and a legend.
It has an editable line chart and embedded
workbook; a PNG slide preview is also exported. The CSV retains exact byte counts, build IDs, capture dates when
present in filenames, wxemem versions, statuses, and source paths.
Existing unprefixed CSV fields (`used_bytes`, `source`, `status`, etc.) retain
their before-PnP meaning. Added `postpnp_*` fields hold the post-PnP measurements.
The `signin_*` fields hold the sign-in measurements with independent provenance.
The coverage JSON retains before-PnP counts and adds `postpnp_included_builds`,
`postpnp_missing_builds`, and `postpnp_stage_patterns`.
Equivalent `signin_included_builds`, `signin_missing_builds`, and
`signin_stage_patterns` describe the sign-in series.
It lists missing samples independently for each stage and excluded non-build directories.
It also records the build-list path and requested IDs when a list is supplied.
The slide's speaker notes also list missing builds.

After generating, report the PowerPoint location and eligible/total build
counts for each series. Mention missing data and scope exclusions. Each series'
first/latest/change/peak summary uses its own available captures; displayed
latest dates can therefore differ. Do not interpret changes as regressions
without further evidence.

## Requirements

Windows PowerShell 5.1 or PowerShell 7, desktop Microsoft PowerPoint and Excel,
and an interactive desktop session are required for Office COM automation.
No Python packages are needed. The script only closes the presentation and
workbook it creates; it does not quit existing Office applications.
Default paths are resolved relative to the generator script; relative explicit
paths such as `-BuildList .\BuildList.txt` resolve from the caller's working directory.

Optional paths:

```powershell
.\Generate-MemoryTrend.ps1 -DailyRoot 'D:\captures\daily' -OutputDirectory 'D:\reports'
```

Use `-DataOnly` to generate the CSV and coverage JSON without Office. For a
deliberate new agent behavior or metric change, update the generator's version.

## Selected-build reports

Supply a text file with one exact build directory name per line. Blank lines
and whole-line `#` comments are allowed. See `builds.example.txt`.

```powershell
.\Generate-MemoryTrend.ps1 -BuildList '.\builds.example.txt' -OutputDirectory '.\output\selected-builds'
```

The list selects membership; the graph still sorts chronologically by build
timestamp regardless of line order. Duplicate IDs, invalid IDs, empty lists,
and nonexistent build directories stop generation with an explicit error.
A listed, existing build with no eligible capture remains a gap. At least one
selected build must have an eligible capture in any of the three stages to generate a report.
Use a separate output directory to preserve the all-build report.
If Office has an output presentation open, close it yourself or choose another
output directory; the generator will not close someone else's presentation.

Data-selection regression checks (no Office required):

```powershell
.\Test-MemoryTrend.ps1
```
