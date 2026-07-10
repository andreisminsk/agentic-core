"""Fuzzy and exact line-based search/match for EDIT operations.

Adapted from ollama-chat-agentic — layered matching with CRLF normalization.
Returns char positions for direct string replacement in actions.py.
"""

import difflib


def _dedent_lines(lines):
    """Remove common leading whitespace from lines."""
    min_indent = float('inf')
    for line in lines:
        stripped = line.lstrip()
        if stripped:
            min_indent = min(min_indent, len(line) - len(stripped))
    if min_indent == float('inf'):
        min_indent = 0
    return [line[min_indent:] if len(line) >= min_indent and line.strip() else line
            for line in lines]


def _search_lines_exact(haystack, needle):
    n = len(needle)
    if n == 0:
        return None
    for i in range(len(haystack) - n + 1):
        if haystack[i:i + n] == needle:
            return (i, i + n)
    return None


def _search_lines_stripped(haystack, needle):
    n = len(needle)
    if n == 0:
        return None
    sh = [l.rstrip() for l in haystack]
    sn = [l.rstrip() for l in needle]
    for i in range(len(sh) - n + 1):
        if sh[i:i + n] == sn:
            return (i, i + n)
    return None


def _search_lines_dedented(haystack, needle):
    n = len(needle)
    if n == 0:
        return None
    dedented_needle = _dedent_lines(needle)
    for i in range(len(haystack) - n + 1):
        if _dedent_lines(haystack[i:i + n]) == dedented_needle:
            return (i, i + n)
    return None


def _search_lines_fuzzy(haystack, needle, threshold=0.8):
    n = len(needle)
    if n == 0:
        return None
    needle_text = '\n'.join(needle)
    best_ratio = 0.0
    best_pos = None
    for i in range(max(1, len(haystack) - n + 1)):
        candidate_text = '\n'.join(haystack[i:i + n])
        ratio = difflib.SequenceMatcher(None, candidate_text, needle_text).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_pos = i
    if best_ratio >= threshold and best_pos is not None:
        return (best_pos, best_pos + n)
    return None


def _lines_to_char_pos(lines, start_line, end_line):
    """Convert (start_line, end_line) to char positions in the original content."""
    start_char = sum(len(l) + 1 for l in lines[:start_line])
    end_char = start_char + sum(len(l) + 1 for l in lines[start_line:end_line])
    return (start_char, end_char)


def find_match_in_content(file_content, search_text):
    """Find search_text in file_content using layered fuzzy matching.

    Returns (start_char, end_char, match_quality) or None.
    match_quality: 'exact', 'whitespace', 'dedented', or 'fuzzy'.

    Normalizes CRLF/CR to LF before matching (critical on Windows).
    """
    # Normalize line endings
    normalized_content = file_content.replace('\r\n', '\n').replace('\r', '\n')
    normalized_search = search_text.replace('\r\n', '\n').replace('\r', '\n')

    file_lines = normalized_content.split('\n')
    search_lines = normalized_search.split('\n')

    # Strip trailing empty strings from trailing newlines
    while file_lines and file_lines[-1] == '':
        file_lines.pop()
    while search_lines and search_lines[-1] == '':
        search_lines.pop()

    if not search_lines:
        return None

    # Try each matching strategy in order of precision
    for matcher, quality in [
        (_search_lines_exact, 'exact'),
        (_search_lines_stripped, 'whitespace'),
        (_search_lines_dedented, 'dedented'),
        (_search_lines_fuzzy, 'fuzzy'),
    ]:
        result = matcher(file_lines, search_lines)
        if result:
            start_line, end_line = result
            # Convert line positions to char positions in the ORIGINAL content
            # (not normalized — so replacement works on the actual file)
            start_char, end_char = _lines_to_char_pos(
                normalized_content.split('\n'), start_line, end_line
            )
            return (start_char, end_char, quality)

    return None
