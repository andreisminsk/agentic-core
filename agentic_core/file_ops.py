"""File operation handlers for WRITE, EDIT, FILE action blocks.

Extracted from actions.py — handles file creation, editing with
search/replace, and file reading.
"""

import os
import shutil

from .constants import MAX_CONSOLE_DISPLAY_CHARS
from .edit_utils import make_edit_summary, make_unified_diff
from .parser import parse_edit_content
from .matching import find_match_in_content
from .utils import agent_print, tool_print


class FileOperations:
    """Mixin providing file operation handlers for ActionExecutor."""

    def _save_pre_edit(self, path, content):
        """Save file content before modification for potential rollback."""
        if not hasattr(self, '_pre_edit_snapshots'):
            self._pre_edit_snapshots = {}
        if path not in self._pre_edit_snapshots:
            self._pre_edit_snapshots[path] = content

    def _rollback_edits(self, modified, created):
        """Restore files to their pre-edit state and remove newly created files."""
        for path in modified:
            if path in self._pre_edit_snapshots:
                full_path = os.path.join(self.workdir, path)
                os.makedirs(os.path.dirname(full_path) or ".", exist_ok=True)
                with open(full_path, "w", encoding="utf-8") as f:
                    f.write(self._pre_edit_snapshots[path])
        for path in created:
            full_path = os.path.join(self.workdir, path)
            if os.path.isfile(full_path):
                try:
                    os.remove(full_path)
                except OSError:
                    pass

    def _do_write(self, action):
        path = action["path"]
        code = action["code"]
        full = os.path.join(self.workdir, path)
        # Cache previous version
        if os.path.isfile(full):
            cache_dir = os.path.join(self.workdir, ".uhu", ".cache")
            os.makedirs(cache_dir, exist_ok=True)
            base, ext = os.path.splitext(path)
            for i in range(1, 999):
                cache_name = f"{base}.{i}{ext}"
                cache_path = os.path.join(cache_dir, cache_name)
                if not os.path.exists(cache_path):
                    try:
                        shutil.copy2(full, cache_path)
                    except Exception:
                        pass
                    break
        # Create parent dirs
        parent = os.path.dirname(full)
        if parent:
            os.makedirs(parent, exist_ok=True)
        try:
            with open(full, "w", encoding="utf-8") as f:
                f.write(code)
            lines = code.count('\n') + (1 if code and not code.endswith('\n') else 0)
            agent_print(f"[Wrote: {path} ({len(code)} bytes, {lines} lines)]")
            return f"[Wrote: {path} ({len(code)} bytes, {lines} lines)]"
        except Exception as e:
            return f"Error writing {path}: {e}"

    def _do_edit(self, action):
        path = action["path"]
        code = action["code"]
        full = os.path.join(self.workdir, path)
        if not os.path.isfile(full):
            return f"Error: file not found for edit: {path}"
        try:
            with open(full, "r", encoding="utf-8", errors="replace") as f:
                original = f.read()
        except Exception as e:
            return f"Error reading {path}: {e}"

        blocks = parse_edit_content(code)
        if not blocks:
            return f"Error: no search/replace blocks found in EDIT for {path}"

        content = original
        edits_applied = []
        failures = []
        for search_text, replace_text in blocks:
            if not search_text:
                # Insert at end
                content = content + replace_text
                edits_applied.append((len(content.split('\n')) - 1, len(content.split('\n')), 'insert', search_text, replace_text))
                continue
            match_info = find_match_in_content(content, search_text)
            if match_info is None:
                failures.append(search_text)
                continue
            start, end, quality = match_info
            content = content[:start] + replace_text + content[end:]
            edits_applied.append((start, end, quality, search_text, replace_text))

        if not edits_applied and failures:
            # All search blocks failed — show file content snippet to help model
            snippet_lines = original.split('\n')[:20]
            snippet = '\n'.join(snippet_lines)
            if len(original.split('\n')) > 20:
                snippet += f"\n... ({len(original.split(chr(10)))} lines total)"
            return (
                f"[EDIT FAILED: {path} — search text not found]\n"
                f"File content:\n{snippet}"
            )

        if failures:
            # Some blocks failed — apply what we can, warn about the rest
            failed_previews = [f[:80].replace('\n', '\\n') for f in failures]

        try:
            os.makedirs(os.path.dirname(full) or ".", exist_ok=True)
            with open(full, "w", encoding="utf-8") as f:
                f.write(content)
        except Exception as e:
            return f"Error writing {path}: {e}"

        summary = make_edit_summary(path, edits_applied)
        diff = make_unified_diff(path, original, content)
        agent_print(summary)
        if failures:
            return (
                f"{summary}\n{diff}\n"
                f"[WARNING: {len(failures)} search block(s) not found: "
                f"{'; '.join(failed_previews)}]"
            )
        return f"{summary}\n{diff}"

    def _do_file(self, action):
        path = action["path"]
        full = os.path.join(self.workdir, path)
        if not os.path.isfile(full):
            return f"Error: file not found: {path}"
        try:
            with open(full, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
            lines_count = content.count('\n') + (1 if content and not content.endswith('\n') else 0)
            header = f"[File: {path} | {lines_count} lines]"
            tool_print(header)
            tool_print(content[:MAX_CONSOLE_DISPLAY_CHARS])
            return f"{header}\n{content}"
        except Exception as e:
            return f"Error reading {path}: {e}"

    def _preview_edit_diff(self, action):
        """Generate a diff preview for an EDIT action before execution."""
        path = action["path"]
        full = os.path.join(self.workdir, path)
        if not os.path.isfile(full):
            return None
        try:
            with open(full, "r", encoding="utf-8") as f:
                original = f.read()
        except Exception:
            return None
        blocks = parse_edit_content(action["code"])
        if not blocks:
            return None
        content = original
        for search_text, replace_text in blocks:
            if not search_text:
                content = content + replace_text
                continue
            match_info = find_match_in_content(content, search_text)
            if match_info is None:
                continue
            start, end, _ = match_info
            content = content[:start] + replace_text + content[end:]
        return make_unified_diff(path, original, content)
