# Bug Triage Environment

An OpenEnv-compatible reinforcement learning environment where AI agents learn to triage software bug reports.

## Motivation

Every software team deals with a constant stream of bug reports. Triaging them correctly is a skill that takes time to learn. This environment lets AI agents practice and improve at exactly this task.

## Environment Overview

- Observation space: Text (bug report + structured prompt)
- Action space: Discrete (single word from valid options)
- Reward range: -0.2 to 1.0
- Max steps per episode: 5
- Number of tasks: 3

## Tasks

### Task 1: Severity Classification (Easy)
- Goal: Classify the bug as low, medium, or high severity
- Reward: 1.0 = correct, 0.5 = adjacent severity, 0.0 = wrong
- Example: A crash affecting all users -> high

### Task 2: Team Routing (Medium)
- Goal: Route the bug to frontend, backend, or infra
- Reward: 1.0 = correct, 0.0 = wrong
- Example: A slow SQL query -> backend

### Task 3: Action Selection (Hard)
- Goal: Choose the right action: hotfix, patch, optimize, schedule, or close
- Reward: 1.0 = correct, 0.3 = reasonable but suboptimal, 0.0 = wrong
- Example: Passwords stored in plain text -> hotfix

## API Endpoints

- POST /reset  — Start new episode, returns first observation
- POST /step   — Submit action, returns reward + next observation
- GET  /state  — Returns current environment state
- GET  /tasks  — Lists all available tasks
- GET  /health — Health check

## Setup and Usage

### Run locally with Docker
```
docker build -t bug-triage-env .
docker run -p 7860:7860 bug-triage-env
```

### Run inference baseline
```
export HF_TOKEN=your_token_here
export ENV_URL=http://localhost:7860
python inference.py
```

### Environment Variables

- API_BASE_URL : LLM API endpoint (default: https://router.huggingface.co/v1)
- MODEL_NAME   : Model to use (default: Qwen/Qwen2.5-72B-Instruct)
- HF_TOKEN     : Hugging Face API token (required)
- ENV_URL      : Environment server URL (default: http://localhost:7860)

## Baseline Scores

Scores achieved by Qwen/Qwen2.5-72B-Instruct:

- severity-classification : 0.85
- team-routing            : 0.80
- action-selection        : 0.70
- Average                 : 0.78

## Project Structure

- environment.py   : Core env: step(), reset(), state()
- tasks.py         : 3 tasks + deterministic graders
- server.py        : FastAPI server
- inference.py     : Baseline LLM agent script
- openenv.yaml     : OpenEnv spec metadata
- requirements.txt : Python dependencies
- Dockerfile       : Container config
- README.md        : This file

## Author
Tanveer Mehra — Built for OpenEnv Hackathon Round 1