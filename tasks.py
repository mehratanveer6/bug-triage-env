
import random

# ─────────────────────────────────────────
# BUG REPORT DATASET
# ─────────────────────────────────────────

BUG_REPORTS = [
    {
        "id": "BUG-001",
        "title": "Button not clickable on mobile",
        "description": "The submit button on the checkout page is unresponsive on iOS Safari. Users cannot complete purchases.",
        "severity": "high",
        "team": "frontend",
        "action": "hotfix"
    },
    {
        "id": "BUG-002",
        "title": "Slow database query on dashboard",
        "description": "The analytics dashboard takes 45 seconds to load due to an unoptimized SQL query joining 4 tables.",
        "severity": "medium",
        "team": "backend",
        "action": "optimize"
    },
    {
        "id": "BUG-003",
        "title": "Typo in footer text",
        "description": "The footer says Copywright instead of Copyright. Minor cosmetic issue.",
        "severity": "low",
        "team": "frontend",
        "action": "patch"
    },
    {
        "id": "BUG-004",
        "title": "Server crashes under high load",
        "description": "Production server goes down when more than 500 concurrent users are active. Memory usage hits 100%.",
        "severity": "high",
        "team": "infra",
        "action": "hotfix"
    },
    {
        "id": "BUG-005",
        "title": "Wrong tax calculation for EU users",
        "description": "VAT is being applied twice for users in Germany and France, causing overcharging.",
        "severity": "high",
        "team": "backend",
        "action": "hotfix"
    },
    {
        "id": "BUG-006",
        "title": "Dark mode colors look off",
        "description": "In dark mode, some text appears grey on grey background making it hard to read.",
        "severity": "low",
        "team": "frontend",
        "action": "patch"
    },
    {
        "id": "BUG-007",
        "title": "SSL certificate expiring soon",
        "description": "The SSL certificate for api.example.com expires in 3 days. Needs urgent renewal.",
        "severity": "medium",
        "team": "infra",
        "action": "schedule"
    },
    {
        "id": "BUG-008",
        "title": "User passwords stored in plain text",
        "description": "A code review revealed passwords are being logged to a plain text file on the server.",
        "severity": "high",
        "team": "backend",
        "action": "hotfix"
    },
    {
        "id": "BUG-009",
        "title": "Missing alt text on images",
        "description": "Product images on the shop page have no alt text, failing accessibility checks.",
        "severity": "low",
        "team": "frontend",
        "action": "patch"
    },
    {
        "id": "BUG-010",
        "title": "Disk space critically low on prod server",
        "description": "Production server at 98% disk usage. Logs not rotating correctly.",
        "severity": "medium",
        "team": "infra",
        "action": "optimize"
    },
]

VALID_SEVERITIES = ["low", "medium", "high"]
VALID_TEAMS = ["frontend", "backend", "infra"]
VALID_ACTIONS = ["hotfix", "patch", "optimize", "schedule", "close"]

# ─────────────────────────────────────────
# TASK 1: SEVERITY CLASSIFICATION (Easy)
# ─────────────────────────────────────────

def get_task1_prompt(bug: dict) -> str:
    return f"""You are a bug triage assistant.

Bug Report:
Title: {bug["title"]}
Description: {bug["description"]}

Classify the severity of this bug.
Reply with ONLY one word: low, medium, or high"""

def grade_task1(action: str, bug: dict) -> float:
    action = action.strip().lower()
    correct = bug["severity"]
    if action == correct:
        return 1.0
    # Partial credit: adjacent severity gets 0.5
    severity_order = ["low", "medium", "high"]
    if action in severity_order:
        diff = abs(severity_order.index(action) - severity_order.index(correct))
        if diff == 1:
            return 0.5
    return 0.0

# ─────────────────────────────────────────
# TASK 2: TEAM ROUTING (Medium)
# ─────────────────────────────────────────

def get_task2_prompt(bug: dict) -> str:
    return f"""You are a bug triage assistant.

Bug Report:
Title: {bug["title"]}
Description: {bug["description"]}
Severity: {bug["severity"]}

Which team should handle this bug?
Reply with ONLY one word: frontend, backend, or infra"""

def grade_task2(action: str, bug: dict) -> float:
    action = action.strip().lower()
    correct = bug["team"]
    if action == correct:
        return 1.0
    return 0.0

# ─────────────────────────────────────────
# TASK 3: ACTION SELECTION (Hard)
# ─────────────────────────────────────────

def get_task3_prompt(bug: dict) -> str:
    return f"""You are a senior engineering manager doing bug triage.

Bug Report:
Title: {bug["title"]}
Description: {bug["description"]}
Severity: {bug["severity"]}
Assigned Team: {bug["team"]}

What is the correct next action?
Options: hotfix, patch, optimize, schedule, close

- hotfix: deploy emergency fix immediately (for critical bugs affecting users now)
- patch: fix in next regular release (for minor issues)
- optimize: refactor or improve performance (for non-urgent slowness)
- schedule: plan for future sprint (for non-urgent infrastructure work)
- close: no action needed (for invalid or duplicate reports)

Reply with ONLY one word from the options above."""

def grade_task3(action: str, bug: dict) -> float:
    action = action.strip().lower()
    correct = bug["action"]
    if action == correct:
        return 1.0
    # Partial credit for reasonable but wrong answers
    partial_credit = {
        "hotfix": {"patch": 0.3},
        "patch": {"hotfix": 0.3, "schedule": 0.2},
        "optimize": {"schedule": 0.3},
        "schedule": {"optimize": 0.3, "patch": 0.2},
        "close": {}
    }
    if correct in partial_credit and action in partial_credit.get(correct, {}):
        return partial_credit[correct][action]
    return 0.0

# ─────────────────────────────────────────
# TASK REGISTRY
# ─────────────────────────────────────────

TASKS = {
    "severity-classification": {
        "name": "severity-classification",
        "description": "Classify the severity of a bug report as low, medium, or high",
        "difficulty": "easy",
        "get_prompt": get_task1_prompt,
        "grader": grade_task1,
        "valid_actions": VALID_SEVERITIES,
    },
    "team-routing": {
        "name": "team-routing",
        "description": "Route the bug report to the correct team: frontend, backend, or infra",
        "difficulty": "medium",
        "get_prompt": get_task2_prompt,
        "grader": grade_task2,
        "valid_actions": VALID_TEAMS,
    },
    "action-selection": {
        "name": "action-selection",
        "description": "Select the correct action: hotfix, patch, optimize, schedule, or close",
        "difficulty": "hard",
        "get_prompt": get_task3_prompt,
        "grader": grade_task3,
        "valid_actions": VALID_ACTIONS,
    },
}

def get_random_bug():
    return random.choice(BUG_REPORTS)
