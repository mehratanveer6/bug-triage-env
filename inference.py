import asyncio
import os
import sys
import requests
from openai import OpenAI

API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or "dummy-key"
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
ENV_URL      = os.getenv("ENV_URL", "https://hoelanderrr-bug-triage-env.hf.space")
BENCHMARK    = "bug-triage-env"
MAX_STEPS    = 5
MAX_TOKENS   = 50
TEMPERATURE  = 0.0
TASKS        = ["severity-classification", "team-routing", "action-selection"]

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

def ask_llm(client, prompt, valid_actions):
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "Reply with exactly ONE word from the valid options. No explanation."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
        )
        raw = response.choices[0].message.content.strip().lower()
        for word in raw.split():
            clean = word.strip(".,!?;:")
            if clean in valid_actions:
                return clean
        return raw.split()[0] if raw else "unknown"
    except Exception:
        return valid_actions[0] if valid_actions else "unknown"

def run_task(client, task_name):
    rewards, steps_taken, success, score = [], 0, False, 0.0
    log_start(task=task_name, model=MODEL_NAME)
    try:
        try:
            reset_resp = requests.post(f"{ENV_URL}/reset", json={"task_name": task_name}, timeout=30)
            reset_resp.raise_for_status()
            obs = reset_resp.json()
        except Exception as e:
            log_step(step=1, action="null", reward=0.0, done=True, error=f"reset_failed:{str(e)[:30]}")
            log_end(success=False, steps=0, score=0.0, rewards=[0.0])
            return 0.0

        for step_num in range(1, MAX_STEPS + 1):
            try:
                prompt        = obs.get("prompt", "")
                valid_actions = obs.get("valid_actions", [])
                action        = ask_llm(client, prompt, valid_actions)
                step_resp     = requests.post(f"{ENV_URL}/step", json={"task_name": task_name, "action": action}, timeout=30)
                step_resp.raise_for_status()
                result        = step_resp.json()
                reward        = float(result.get("reward", 0.0))
                done          = bool(result.get("done", False))
                rewards.append(reward)
                steps_taken   = step_num
                log_step(step=step_num, action=action, reward=reward, done=done)
                if done:
                    break
                obs = result.get("observation", obs)
            except Exception as e:
                log_step(step=step_num, action="null", reward=0.0, done=True, error=str(e)[:30])
                break

        score   = max(rewards) if rewards else 0.0
        success = score >= 0.8
    except Exception:
        score, success = 0.0, False
    finally:
        if not rewards:
            rewards = [0.0]
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
    return score

async def main():
    try:
        client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)
    except Exception as e:
        print(f"# OpenAI client error: {e}", flush=True)
        client = None

    print(f"# Running Bug Triage Environment Baseline", flush=True)
    print(f"# Model: {MODEL_NAME}", flush=True)
    print(f"# Env URL: {ENV_URL}", flush=True)
    print("", flush=True)

    all_scores = {}
    for task in TASKS:
        try:
            score = run_task(client, task)
        except Exception as e:
            print(f"# Task {task} failed: {e}", flush=True)
            score = 0.0
            log_end(success=False, steps=0, score=0.0, rewards=[0.0])
        all_scores[task] = score
        print("", flush=True)

    print("# FINAL SCORES", flush=True)
    for task, score in all_scores.items():
        print(f"# {task}: {score:.2f}", flush=True)
    avg = sum(all_scores.values()) / len(all_scores)
    print(f"# Average: {avg:.2f}", flush=True)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"# Fatal error: {e}", flush=True)
        sys.exit(0)
