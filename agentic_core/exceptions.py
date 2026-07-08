"""Exceptions for the agentic core."""


class AgenticInterrupted(Exception):
    """Raised when the user interrupts the agentic loop (Ctrl+C).

    The core catches KeyboardInterrupt at three boundaries (streaming,
    action execution, confirmation) and raises this clean exception
    instead, ensuring conversation history remains consistent.

    Attributes:
        partial_text: Text collected so far (partial model response or
            observations from actions executed before the interrupt).
        phase: Where the interrupt occurred — 'streaming', 'actions',
            or 'confirmation'.
    """
    def __init__(self, partial_text="", phase="unknown"):
        self.partial_text = partial_text
        self.phase = phase
        super().__init__(f"Interrupted during {phase}: {partial_text[:100]}")
