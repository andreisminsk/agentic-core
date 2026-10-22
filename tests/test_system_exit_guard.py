"""SystemExit guardrail (port of upstream 65c5579).

Tools/skills that exec model-supplied code may call sys.exit() —
SystemExit is a BaseException and must not escape into the session loop.
"""

import sys
import unittest

from agentic_core.actions import ActionExecutor
from agentic_core.session import AgenticSession
from agentic_core.tools import Tool, register as register_tool
from agentic_core.skills import Skill, register as register_skill


class _ExitTool(Tool):
    name = "_test_exit_tool"
    description = "raises SystemExit"
    auto_approve = True

    def execute(self, params, workdir=None):
        sys.exit(3)


class _ExitSkill(Skill):
    name = "_test_exit_skill"
    description = "raises SystemExit"
    auto_approve = True

    def execute(self, params, workdir=None):
        sys.exit(7)


register_tool(_ExitTool())
register_skill(_ExitSkill())


class TestSystemExitGuard(unittest.TestCase):
    def test_tool_systemexit_converted(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            ex = ActionExecutor(workdir=td, confirm_callback=lambda a, d: True)
            obs = ex.execute_actions(
                [{"type": "tool", "path": "_test_exit_tool", "params": {}}])
            self.assertEqual(len(obs), 1)
            self.assertIn("SystemExit(3)", obs[0])
            self.assertIn("session continues", obs[0])

    def test_skill_systemexit_converted(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            ex = ActionExecutor(workdir=td, confirm_callback=lambda a, d: True)
            obs = ex.execute_actions(
                [{"type": "skill", "path": "_test_exit_skill", "params": {}}])
            self.assertEqual(len(obs), 1)
            self.assertIn("SystemExit(7)", obs[0])

    def test_signature_params_differ(self):
        sig1 = AgenticSession._action_signature(
            [{"type": "tool", "path": "read_file",
              "params": {"path": "a.py"}}])
        sig2 = AgenticSession._action_signature(
            [{"type": "tool", "path": "read_file",
              "params": {"path": "b.py"}}])
        self.assertNotEqual(sig1, sig2)


if __name__ == "__main__":
    unittest.main()
