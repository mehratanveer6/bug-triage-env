---
title: Bug Triage Environment
emoji: 🐛
colorFrom: red
colorTo: blue
sdk: docker
pinned: false
tags:
  - openenv
---

# Bug Triage Environment

An OpenEnv-compatible reinforcement learning environment where AI agents learn to triage software bug reports.

## Tasks

- Task 1: Severity Classification (Easy) - classify bug as low, medium, or high
- Task 2: Team Routing (Medium) - route to frontend, backend, or infra
- Task 3: Action Selection (Hard) - choose hotfix, patch, optimize, schedule, or close

## API Endpoints

- POST /reset  - Start new episode
- POST /step   - Submit action, get reward
- GET  /state  - Current environment state
- GET  /tasks  - List all tasks
- GET  /health - Health check

## Baseline Scores

- severity-classification : 0.85
- team-routing            : 0.80
- action-selection        : 0.70
- Average                 : 0.78

## Author
Tanveer Mehra - Built for OpenEnv Hackathon Round 1
