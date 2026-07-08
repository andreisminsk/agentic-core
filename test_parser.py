"""Quick parser tests for the agentic core."""

from agentic_core.parser import parse_actions, parse_edit_content, extract_text_outside_blocks


def test_write_block():
    text = (
        "I'll create the file.\n\n"
        "**WRITE:`hello.py`**\n"
        "```python\n"
        "print('hello')\n"
        "```\n"
        "**EOF:`hello.py`**\n"
    )
    actions = parse_actions(text)
    assert len(actions) == 1
    assert actions[0]["type"] == "write"
    assert actions[0]["path"] == "hello.py"
    assert "print('hello')" in actions[0]["code"]
    assert actions[0]["closed"] is True
    print("test_write_block: PASS")


def test_edit_block():
    text = (
        "**EDIT:`app.py`**\n"
        "```search-replace\n"
        "<<<<<<< SEARCH\n"
        "old = 1\n"
        "=======\n"
        "new = 2\n"
        ">>>>>>> REPLACE\n"
        "```\n"
        "**EOF:`app.py`**\n"
    )
    actions = parse_actions(text)
    assert len(actions) == 1
    assert actions[0]["type"] == "edit"
    blocks = parse_edit_content(actions[0]["code"])
    assert len(blocks) == 1
    assert blocks[0][0] == "old = 1"
    assert blocks[0][1] == "new = 2"
    print("test_edit_block: PASS")


def test_run_block():
    text = (
        "Let me run it.\n\n"
        "**RUN:**\n"
        "```bash\n"
        "python hello.py\n"
        "```\n"
    )
    actions = parse_actions(text)
    assert len(actions) == 1
    assert actions[0]["type"] == "run"
    assert actions[0]["lang"] == "bash"
    assert "python hello.py" in actions[0]["code"]
    print("test_run_block: PASS")


def test_tool_block():
    text = (
        "**TOOL:`read_file`**\n"
        "```json\n"
        '{"path": "src/app.py"}\n'
        "```\n"
        "**EOF:`read_file`**\n"
    )
    actions = parse_actions(text)
    assert len(actions) == 1
    assert actions[0]["type"] == "tool"
    assert actions[0]["path"] == "read_file"
    assert actions[0]["params"]["path"] == "src/app.py"
    print("test_tool_block: PASS")


def test_multiple_blocks():
    text = (
        "Here's what I'll do.\n\n"
        "**WRITE:`a.py`**\n"
        "```python\na=1\n```\n"
        "**EOF:`a.py`**\n\n"
        "**WRITE:`b.py`**\n"
        "```python\nb=2\n```\n"
        "**EOF:`b.py`**\n"
    )
    actions = parse_actions(text)
    assert len(actions) == 2
    assert actions[0]["path"] == "a.py"
    assert actions[1]["path"] == "b.py"
    print("test_multiple_blocks: PASS")


def test_extract_prose():
    text = (
        "Let me explain.\n\n"
        "**WRITE:`x.py`**\n"
        "```python\nx=1\n```\n"
        "**EOF:`x.py`**\n\n"
        "Done!"
    )
    prose = extract_text_outside_blocks(text)
    assert "Let me explain" in prose
    assert "Done" in prose
    print("test_extract_prose: PASS")


if __name__ == "__main__":
    test_write_block()
    test_edit_block()
    test_run_block()
    test_tool_block()
    test_multiple_blocks()
    test_extract_prose()
    print("\nAll tests passed!")
