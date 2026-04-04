
import asyncio
import os
import requests
from openai import OpenAI

# ─────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────

API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
ENV_URL      = os.getenv("ENV_URL", "http://localhost:7860")
BENCHMARK    = "bug-triage-env"
MAX_STEPS    = 5
TEMPERATURE  = 0.0
MAX_TOKENS   = 50

TASKS = [
    "severity-classification",
    "team-routing",
    "action-selection",
]

# ─────────────────────────────────────────
# LOGGING (MANDATORY FORMAT)
# ─────────────────────────────────────────

def log_start(task, model):
    print(f"[START] task={task} env={BENCHMARK} model={model}", flush=True)

def log_step(step, action, reward, done, error=None):
    error_str = error if error else "null"
    done_str  = "true" if done else "false"
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_str} error={error_str}", flush=True)

def log_end(success, steps, score, rewards):
    rewards_str = ",".join([f"{r:.2f}" for r in rewards])
    success_str = "true" if success else "false"
    print(f"[END] success={success_str} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)

# ─────────────────────────────────────────
# AGENT LOGIC
# ─────────────────────────────────────────

def ask_llm(client, prompt, valid_actions):
    """Ask the LLM and extract a valid action from its response."""
    system = (
        "You are a precise bug triage assistant. "
        "Always reply with exactly ONE word from the valid options. "
        "No explanation, no punctuation, just the single word."
    )
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": prompt}
            ],
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
        )
        raw = response.choices[0].message.content.strip().lower()
        # Extract first valid action found in response
        for word in raw.split():
            clean = word.strip(".,!?;:")
            if clean in valid_actions:
                return clean
        # If no valid action found, return raw (will get penalized by grader)
        return raw.split()[0] if raw else "unknown"
    except Exception as e:
        return f"error:{str(e)[:30]}"

def run_task(client, task_name):
    """Run one full episode for a task."""
    rewards   = []
    steps_taken = 0
    success   = False
    score     = 0.0
    last_error = None

    log_start(task=task_name, model=MODEL_NAME)

    try:
        # ── Reset environment ──
        reset_resp = requests.post(
            f"{ENV_URL}/reset",
            json={"task_name": task_name},
            timeout=30
        )
        reset_resp.raise_for_status()
        obs = reset_resp.json()

        for step_num in range(1, MAX_STEPS + 1):
            prompt       = obs.get("prompt", "")
            valid_actions = obs.get("valid_actions", [])

            # ── Get action from LLM ──
            action = ask_llm(client, prompt, valid_actions)

            # ── Send action to environment ──
            step_resp = requests.post(
                f"{ENV_URL}/step",
                json={"task_name": task_name, "action": action},
                timeout=30
            )
            step_resp.raise_for_status()
            result = step_resp.json()

            reward     = result.get("reward", 0.0)
            done       = result.get("done", False)
            info       = result.get("info", {})
            last_error = info.get("reason", None)

            rewards.append(reward)
            steps_taken = step_num

            log_step(
                step=step_num,
                action=action,
                reward=reward,
                done=done,
                error=None
            )

            if done:
                # Score is best reward achieved in episode
                score   = max(rewards)
                success = score >= 0.8
                break

            obs = result.get("observation", obs)

        if not rewards:
            score   = 0.0
            success = False

        score = max(rewards) if rewards else 0.0
        success = score >= 0.8

    except Exception as e:
        last_error = str(e)
        log_step(step=steps_taken+1, action="null", reward=0.0, done=True, error=last_error[:50])
        score   = 0.0
        success = False

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score

# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────

async def main():
    client = OpenAI(
        api_key=API_KEY,
        base_url=API_BASE_URL,
    )

    print(f"# Running Bug Triage Environment Baseline", flush=True)
    print(f"# Model: {MODEL_NAME}", flush=True)
    print(f"# Tasks: {TASKS}", flush=True)
    print(f"# Env URL: {ENV_URL}", flush=True)
    print("", flush=True)

    all_scores = {}
    for task in TASKS:
        score = run_task(client, task)
        all_scores[task] = score
        print("", flush=True)

    print("# ─── FINAL SCORES ───", flush=True)
    for task, score in all_scores.items():
        print(f"# {task}: {score:.2f}", flush=True)
    avg = sum(all_scores.values()) / len(all_scores)
    print(f"# Average score: {avg:.2f}", flush=True)

if __name__ == "__main__":
    asyncio.run(main())
