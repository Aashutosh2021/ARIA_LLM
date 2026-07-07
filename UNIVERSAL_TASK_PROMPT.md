
# UNIVERSAL_TASK_PROMPT.md

## IMPORTANT

Before performing ANY task, you MUST follow this workflow.

Do NOT immediately scan the source code.

The project contains a permanent AI Brain inside the `brain/` directory.

Your first responsibility is to use that Brain.

---

# STEP 1 — Verify Brain

Open

brain/VERSION.json

Verify:

* brain_status == Healthy
* documentation is synchronized
* no stale documents exist

If the Brain is healthy, DO NOT perform a full codebase scan.

Only continue using the Brain.

If the Brain is outdated, update only the affected documentation.

Never regenerate the complete Brain unless explicitly instructed.

---

# STEP 2 — Boot Sequence

Read these files in EXACT order.

1. brain/PROMPT.md
2. brain/21_AGENT_BOOT.md
3. brain/20_AI_CONTEXT.md
4. brain/01_PROJECT_MEMORY.md
5. brain/17_PROJECT_RULES.md
6. brain/18_CURRENT_STATE.md
7. brain/03_FILE_INDEX.md

Do not skip or reorder them.

These files contain the compressed memory of the project.

---

# STEP 3 — Understand the Task

Classify the user's request.

Possible categories include:

* Bug Fix
* New Feature
* UI Change
* Refactoring
* Performance
* Database
* API
* Security
* Testing
* Documentation
* Build
* Deployment

---

# STEP 4 — Load Only Relevant Documentation

Read ONLY the documentation required for the task.

Use

brain/23_FEATURE_DEPENDENCY_MATRIX.md

and

brain/22_IMPACT_MAP.md

to determine what is actually affected.

Examples

Playback

→

16_CALL_GRAPH.md

08_DATA_FLOW.md

Search

→

09_API_MAP.md

11_FEATURE_MAP.md

Database

→

10_DATABASE_MAP.md

UI

→

12_UI_MAP.md

Dependencies

→

06_DEPENDENCY_GRAPH.md

Configuration

→

13_CONFIGURATION_MAP.md

Security

→

17_PROJECT_RULES.md

14_GLOBAL_VARIABLES.md

Do NOT open documentation unrelated to the task.

---

# STEP 5 — Open Source Code

Only after understanding the Brain,

use

03_FILE_INDEX.md

to locate the minimum required source files.

Never recursively scan unrelated folders.

Never index the entire repository.

Never open every file "just in case."

Open only files directly related to the requested feature.

---

# STEP 6 — Implementation Rules

Modify the smallest possible number of files.

Reuse existing architecture.

Reuse existing services.

Reuse existing ViewModels.

Reuse existing repositories.

Reuse existing utilities.

Do not duplicate logic.

Do not rename public APIs.

Do not rename public classes.

Do not refactor unrelated code.

Preserve backward compatibility.

Keep architecture consistent.

---

# STEP 7 — Before Finishing

If source code changed,

update ONLY the affected Brain documents.

Possible files include:

03_FILE_INDEX.md

11_FEATURE_MAP.md

18_CURRENT_STATE.md

20_AI_CONTEXT.md

22_IMPACT_MAP.md

23_FEATURE_DEPENDENCY_MATRIX.md

CHANGELOG.md

VERSION.json

Do NOT regenerate unrelated documentation.

---

# STEP 8 — Final Verification

Before returning your response verify:

✓ Documentation matches source code.

✓ VERSION.json updated if required.

✓ No stale Brain documents remain.

✓ No unnecessary files were modified.

✓ Architecture preserved.

✓ Minimum files changed.

✓ Token usage minimized.

---

# HARD RULES

Never perform a full project scan if the Brain is healthy.

Never ignore the Brain.

Never bypass the boot sequence.

Never regenerate the Brain unless explicitly requested.

Always prefer Brain documentation over rescanning source code.

The Brain directory is the project's primary memory.

Your goal is to minimize context usage, minimize token consumption, and maximize architectural consistency while implementing production-quality code.
