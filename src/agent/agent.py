"""The agent loop: the chat model decides which tool to call, we run it,
feed the result back, and repeat until the model writes its final answer.

Run from the project root:
    python src/agent/agent.py "Assess customer 9237-HQITU and draft a Sinhala SMS."
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv  # noqa: E402
from openai import AzureOpenAI  # noqa: E402

from schemas import SYSTEM, TOOLS  # noqa: E402
from tools import FUNCS  # noqa: E402

load_dotenv()

CHAT_DEPLOYMENT = "chat"
client = AzureOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_key=os.environ["AZURE_OPENAI_KEY"],
    api_version=os.environ["AZURE_OPENAI_API_VERSION"],
)


def run_agent(user_msg: str, history: list | None = None, max_steps: int = 8):
    """Return (final answer, trace). The trace lists every tool call for transparency."""
    messages = [{"role": "system", "content": SYSTEM}] + (history or []) + [{"role": "user", "content": user_msg}]
    trace = []
    for _ in range(max_steps):
        resp = client.chat.completions.create(
            model=CHAT_DEPLOYMENT, messages=messages, tools=TOOLS, temperature=0.2
        )
        msg = resp.choices[0].message
        if not msg.tool_calls:          # no more tools wanted: this is the answer
            return msg.content, trace
        messages.append(msg)
        for call in msg.tool_calls:
            args = json.loads(call.function.arguments)
            try:
                result = FUNCS[call.function.name](**args)
            except Exception as e:      # let the model see the error and explain it
                result = {"error": str(e)}
            trace.append({"tool": call.function.name, "args": args, "result": result})
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": json.dumps(result, ensure_ascii=False),
            })
    return "Stopped: too many steps.", trace


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "Assess customer 9237-HQITU and draft a Sinhala SMS."
    answer, trace = run_agent(question)
    for step in trace:
        print(f"[tool] {step['tool']}({step['args']})")
    print()
    print(answer)
