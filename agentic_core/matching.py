"""Fuzzy and exact matching for EDIT search/replace blocks."""

import difflib


def find_match_in_content(content, search_text, min_ratio=0.85):
    """Find search_text in content, with fuzzy fallback.

    Returns (start, end, quality) or None.
    quality is 'exact' or 'fuzzy'.
    """
    # Try exact match first
    idx = content.find(search_text)
    if idx != -1:
        return (idx, idx + len(search_text), 'exact')

    # Fuzzy: find the best matching window
    content_lines = content.split('\n')
    search_lines = search_text.split('\n')
    n = len(search_lines)
    if n == 0:
        return None

    best_ratio = 0
    best_start = None
    best_end = None

    for i in range(len(content_lines) - n + 1):
        window = content_lines[i:i + n]
        ratio = difflib.SequenceMatcher(None, window, search_lines).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_start = i
            best_end = i + n

    if best_ratio >= min_ratio and best_start is not None:
        # Convert line positions to char positions
        start_char = sum(len(l) + 1 for l in content_lines[:best_start])
        end_char = start_char + sum(len(l) + 1 for l in content_lines[best_start:best_end])
        return (start_char, end_char, 'fuzzy')

    return None
