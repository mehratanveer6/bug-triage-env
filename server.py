
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from environment import BugTriageEnv, Action

app = FastAPI(
    title="Bug Triage Environment",
    description="An OpenEnv-compatible environment for training AI agents to triage software bug reports.",
    version="1.0.0"
)

# ─────────────────────────────────────────
# ONE ENV INSTANCE PER TASK
# ─────────────────────────────────────────

envs = {
    "severity-classification": BugTriageEnv(task_name="severity-classification"),
    "team-routing": BugTriageEnv(task_name="team-routing"),
    "action-selection": BugTriageEnv(task_name="action-selection"),
}

# Initialize all environments
for env in envs.values():
    env.reset()

# ─────────────────────────────────────────
# REQUEST MODELS
# ─────────────────────────────────────────

class ResetRequest(BaseModel):
    task_name: Optional[str] = "severity-classification"

class StepRequest(BaseModel):
    task_name: Optional[str] = "severity-classification"
    action: str

# ─────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────

@app.get("/")
def root():
    return {
        "name": "Bug Triage Environment",
        "version": "1.0.0",
        "tasks": list(envs.keys()),
        "endpoints": ["/reset", "/step", "/state", "/health"]
    }

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/reset")
def reset(request: ResetRequest = None):
    task_name = "severity-classification"
    if request and request.task_name:
        task_name = request.task_name
    if task_name not in envs:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown task: {task_name}. Valid tasks: {list(envs.keys())}"
        )
    obs = envs[task_name].reset()
    return obs.dict()

@app.post("/step")
def step(request: StepRequest):
    task_name = request.task_name
    if task_name not in envs:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown task: {task_name}. Valid tasks: {list(envs.keys())}"
        )
    env = envs[task_name]
    if env.done:
        env.reset()
    action = Action(value=request.action)
    result = env.step(action)
    return result.dict()

@app.get("/state")
def state(task_name: str = "severity-classification"):
    if task_name not in envs:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown task: {task_name}. Valid tasks: {list(envs.keys())}"
        )
    return envs[task_name].state()

@app.get("/tasks")
def list_tasks():
    return {
        "tasks": [
            {
                "name": "severity-classification",
                "difficulty": "easy",
                "description": "Classify bug severity as low, medium, or high",
                "valid_actions": ["low", "medium", "high"]
            },
            {
                "name": "team-routing",
                "difficulty": "medium",
                "description": "Route bug to correct team: frontend, backend, or infra",
                "valid_actions": ["frontend", "backend", "infra"]
            },
            {
                "name": "action-selection",
                "difficulty": "hard",
                "description": "Select correct action: hotfix, patch, optimize, schedule, or close",
                "valid_actions": ["hotfix", "patch", "optimize", "schedule", "close"]
            }
        ]
    }
