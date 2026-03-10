"""Demo: Using Claude Code subscription in a Python script.

Prereq: Run `claude login` once to authenticate via OAuth.
"""

import json
import subprocess
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent


def ask_claude(prompt: str, input_text: str = "", output_json: bool = False) -> str:
    cmd = ["claude", "-p", prompt]
    if output_json:
        cmd.extend(["--output-format", "json"])

    result = subprocess.run(
        cmd,
        input=input_text or None,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(f"Claude failed: {result.stderr}")

    if output_json:
        return json.loads(result.stdout)
    return result.stdout.strip()


if __name__ == "__main__":
    print("=== Test 1: Ask Claude what 2+2 is ===")
    answer = ask_claude("What is 2+2? Reply with just the number.")
    print(answer)

    print("\n=== Test 2: Ask Claude to read and summarize a sentence ===")
    sentence = (SCRIPT_DIR / "sentence.txt").read_text()
    answer = ask_claude("Read this sentence and tell me what animal it mentions:", input_text=sentence)
    print(answer)

    print("\n=== Test 3: JSON output ===")
    answer = ask_claude("What is 2+2? Reply with just the number.", output_json=True)
    print(json.dumps(answer, indent=2))
