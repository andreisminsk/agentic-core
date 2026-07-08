"""Example: custom confirmation callback with agentic_core.

Demonstrates overriding the built-in confirmation with a custom policy
that logs every action and applies selective approval rules.
"""

from agentic_core import AgenticSession, AgenticInterrupted


def custom_confirm(action_type, details):
    """Custom confirmation callback with selective rules.

    The callback receives full context about each action, allowing you to
    implement any approval policy: logging, selective auto-approve, diff
    preview, rate limiting, role-based access control, etc.

    Args:
        action_type: One of 'write', 'edit', 'run', 'tool', 'skill', 'file'.
            - 'file' is always auto-approved by the executor before the
              callback is even invoked (read-only, no side effects).
            - 'write'/'edit' reach the callback only when auto_approve_safe=False.
            - 'run' reaches the callback for non-safe commands (safe ones are
              auto-approved when auto_approve_safe=True).
            - 'tool'/'skill' reach the callback when the tool/skill has
              auto_approve=False (e.g. web_search, http_request).

        details: dict with keys depending on action_type:
            - run: {
                'command': str,       # full shell command
                'level': str,         # 'safe' | 'warning' | 'chain' | 'blocked'
                'message': str,       # human-readable safety message
              }
              Note: 'blocked' commands never reach the callback — they're
              rejected internally. You'll see 'safe' only if auto_approve_safe=False.

            - write: {
                'path': str,          # file path being written
              }

            - edit: {
                'path': str,          # file path being edited
                'diff': str | None,   # unified diff preview of the changes,
                                      # or None if file doesn't exist yet
              }
              The diff is generated BEFORE execution — it's a preview of
              what WILL happen, not what already happened.

            - tool: {
                'name': str,          # tool name (e.g. 'web_search')
                'params': dict,       # parsed parameters from the JSON block
              }

            - skill: {
                'name': str,          # skill name
                'params': dict,       # parsed parameters
              }

    Returns:
        True to approve the action, False to reject it.
        Rejected actions produce an observation like '[run skipped: not
        approved by user]' that is fed back to the model.

    Side effects you can perform in the callback:
        - Print to stdout for user display
        - Read files (e.g. inspect the target before approving WRITE)
        - Log to a file or database for audit trails
        - Update counters for rate limiting
        - Call input() for interactive prompts
        - Raise an exception to abort the entire action batch
    """
    import sys

    # Auto-approve read-only actions
    if action_type == "file":
        return True

    # Auto-approve safe tools (read_file, list_files, calculator, time_now, etc.)
    if action_type in ("tool", "skill"):
        name = details.get("name", "")
        safe_names = {"read_file", "list_files", "find_file", "search_in_files",
                      "peek_file", "time_now", "calculator", "mkdir"}
        if name in safe_names:
            print(f"  [auto-approved {action_type}: {name}]")
            return True

    # For RUN: show command and safety level
    if action_type == "run":
        cmd = details.get("command", "")
        level = details.get("level", "safe")
        print(f"  [RUN ({level})] {cmd[:120]}")
        if level == "safe":
            return True  # auto-approve safe commands
        # Warn for destructive commands
        if level in ("warning", "chain"):
            print(f"  ⚠ {details.get('message', '')}")

    # For WRITE/EDIT: show path and optional diff
    if action_type in ("write", "edit"):
        path = details.get("path", "")
        print(f"  [{action_type.upper()}] {path}")
        diff = details.get("diff")
        if diff and diff != "(no changes)":
            print("  --- diff preview ---")
            for line in diff.splitlines()[:20]:
                if line.startswith("+"):
                    print(f"  \033[32m{line}\033[0m")
                elif line.startswith("-"):
                    print(f"  \033[31m{line}\033[0m")
                else:
                    print(f"  {line}")
            if len(diff.splitlines()) > 20:
                print(f"  ... ({len(diff.splitlines()) - 20} more lines)")
            print("  --- end diff ---")

    # Prompt for remaining actions
    try:
        return input("  Allow? [y/N] ").strip().lower().startswith("y")
    except (EOFError, KeyboardInterrupt):
        return False


def main():
    from ollama import Client
    client = Client(host="http://localhost:11434")

    session = AgenticSession(
        client=client,
        model="glm-5.1:cloud",
        workdir=".",
        tools=True,
        skills=True,
        name="myapp",
        auto_approve_safe=False,
        confirm_callback=custom_confirm,
        max_feedback_rounds=5,
        config_filename=".myapp/agentic-config.json",
    )

    print("Agentic session ready (custom callback). Type 'quit' to exit.\n")
    while True:
        try:
            user_input = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_input or user_input.lower() in ("quit", "exit"):
            break
        try:
            response = session.run(user_input)
        except AgenticInterrupted as e:
            print(f"\n[interrupted during {e.phase}]")
            if e.partial_text:
                print(f"  partial: {e.partial_text[:200]}")
            continue
        print(f"\nassistant> {response}\n")


if __name__ == "__main__":
    main()
