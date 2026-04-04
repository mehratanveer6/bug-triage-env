
import random
from typing import Any, Optional
from pydantic import BaseModel
from tasks import TASKS, get_random_bug, VALID_SEVERITIES, VALID_TEAMS, VALID_ACTIONS

# ─────────────────────────────────────────
# PYDANTIC MODELS (OpenEnv Spec Required)
# ─────────────────────────────────────────

class Observation(BaseModel):
    task_name: str
    bug_id: str
    bug_title: str
    bug_description: str
    prompt: str
    valid_actions: list[str]
    step_number: int

class Action(BaseModel):
    value: str

class Reward(BaseModel):
    score: float
    reason: str

class StepResult(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: dict

# ─────────────────────────────────────────
# MAIN ENVIRONMENT CLASS
# ─────────────────────────────────────────

class BugTriageEnv:
    """
    Bug Report Triage Environment
    An AI agent learns to triage software bug reports by classifying
    severity, routing to teams, and selecting correct actions.
    """

    MAX_STEPS = 5  # Max attempts per episode

    def __init__(self, task_name: str = "severity-classification"):
        if task_name not in TASKS:
            raise ValueError(f"Unknown task: {task_name}. Choose from: {list(TASKS.keys())}")
        self.task_name = task_name
        self.task = TASKS[task_name]
        self.current_bug = None
        self.step_number = 0
        self.done = False
        self.total_reward = 0.0
        self.history = []

    def _make_observation(self) -> Observation:
        return Observation(
            task_name=self.task_name,
            bug_id=self.current_bug["id"],
            bug_title=self.current_bug["title"],
            bug_description=self.current_bug["description"],
            prompt=self.task["get_prompt"](self.current_bug),
            valid_actions=self.task["valid_actions"],
            step_number=self.step_number,
        )

    def reset(self) -> Observation:
        """Reset environment to start a new episode."""
        self.current_bug = get_random_bug()
        self.step_number = 0
        self.done = False
        self.total_reward = 0.0
        self.history = []
        return self._make_observation()

    def step(self, action: Action) -> StepResult:
        """Take one step: agent submits an action, gets reward + next observation."""
        if self.done:
            raise RuntimeError("Episode is done. Call reset() to start a new one.")

        self.step_number += 1
        action_value = action.value.strip().lower()

        # ── Grade the action ──
        raw_score = self.task["grader"](action_value, self.current_bug)

        # ── Reward shaping ──
        # Penalize invalid actions
        if action_value not in self.task["valid_actions"]:
            reward = -0.2
            reason = f"Invalid action '{action_value}'. Valid options: {self.task['valid_actions']}"
        elif raw_score == 1.0:
            reward = 1.0
            reason = f"Correct! '{action_value}' is the right answer."
        elif raw_score > 0:
            reward = raw_score
            reason = f"Partially correct. '{action_value}' is close but not ideal."
        else:
            reward = 0.0
            reason = f"Incorrect. '{action_value}' is wrong for this bug report."

        # ── Penalize too many steps (encourage efficiency) ──
        if self.step_number >= self.MAX_STEPS:
            reward -= 0.1
            self.done = True

        # ── Episode ends on correct answer ──
        if raw_score == 1.0:
            self.done = True

        self.total_reward += reward
        self.history.append({
            "step": self.step_number,
            "action": action_value,
            "reward": reward,
            "done": self.done,
        })

        # ── Next observation (same bug, agent can retry if not done) ──
        next_obs = self._make_observation()

        return StepResult(
            observation=next_obs,
            reward=round(reward, 2),
            done=self.done,
            info={
                "correct_answer": self.current_bug[self._get_answer_key()],
                "reason": reason,
                "total_reward": round(self.total_reward, 2),
                "steps_taken": self.step_number,
            }
        )

    def state(self) -> dict:
        """Return current full state of the environment."""
        return {
            "task_name": self.task_name,
            "bug_id": self.current_bug["id"] if self.current_bug else None,
            "step_number": self.step_number,
            "done": self.done,
            "total_reward": round(self.total_reward, 2),
            "history": self.history,
        }

    def _get_answer_key(self) -> str:
        mapping = {
            "severity-classification": "severity",
            "team-routing": "team",
            "action-selection": "action",
        }
        return mapping[self.task_name]
