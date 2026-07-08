"""Essential file system tools for the agentic core."""

import os
import re
import shutil
from pathlib import Path


class ReadFileTool:
    name = "read_file"
    description = "Read a file's contents."
    system_prompt = (
        "## read_file\n"
        "Read a file's contents. Cross-platform replacement for cat/type.\n"
        "Parameters: path (string, required), offset (int, optional, default 1), "
        "limit (int, optional, default 2000)."
    )
    auto_approve = True

    def execute(self, params, workdir=None):
        path = params.get("path", "")
        offset = int(params.get("offset", 1))
        limit = int(params.get("limit", 2000))
        if not path:
            return "Error: path is required"
        full = os.path.join(workdir or ".", path)
        if not os.path.isfile(full):
            return f"Error: file not found: {path}"
        try:
            with open(full, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
            total = len(lines)
            start = max(0, offset - 1)
            end = min(start + limit, total)
            content = "".join(lines[start:end])
            header = f"[File: {path} | Lines {start+1}-{end} of {total}]\n"
            return header + content
        except Exception as e:
            return f"Error reading file: {e}"


class WriteFileTool:
    name = "write_file"
    description = "Create or overwrite a file."
    system_prompt = (
        "## write_file\n"
        "Create or overwrite a file with the given content. Auto-creates parent directories.\n"
        "Parameters: path (string, required), content (string, required), "
        "append (bool, optional, default false)."
    )
    auto_approve = True

    def execute(self, params, workdir=None):
        path = params.get("path", "")
        content = params.get("content", "")
        append = params.get("append", False)
        if not path:
            return "Error: path is required"
        full = os.path.join(workdir or ".", path)
        os.makedirs(os.path.dirname(full), exist_ok=True) if os.path.dirname(full) else None
        try:
            mode = "a" if append else "w"
            with open(full, mode, encoding="utf-8") as f:
                f.write(content)
            return f"[Wrote: {path} ({len(content)} bytes)]"
        except Exception as e:
            return f"Error writing file: {e}"


class ReplaceInFileTool:
    name = "replace_in_file"
    description = "Replace exact string matches in a file."
    system_prompt = (
        "## replace_in_file\n"
        "Replace one or more exact string matches in a file.\n"
        "Parameters: path (string, required), replacements (array, required) — "
        "each object has 'search' and 'replace' string fields."
    )
    auto_approve = True

    def execute(self, params, workdir=None):
        path = params.get("path", "")
        replacements = params.get("replacements", [])
        if not path:
            return "Error: path is required"
        if not replacements:
            return "Error: replacements array is required"
        full = os.path.join(workdir or ".", path)
        if not os.path.isfile(full):
            return f"Error: file not found: {path}"
        try:
            with open(full, "r", encoding="utf-8") as f:
                content = f.read()
            count = 0
            for r in replacements:
                search = r.get("search", "")
                replace = r.get("replace", "")
                if search and search in content:
                    content = content.replace(search, replace, 1)
                    count += 1
            with open(full, "w", encoding="utf-8") as f:
                f.write(content)
            return f"[Edited: {path} ({count} replacement(s))]"
        except Exception as e:
            return f"Error editing file: {e}"


class ListFilesTool:
    name = "list_files"
    description = "List files and directories."
    system_prompt = (
        "## list_files\n"
        "List files and directories. Cross-platform replacement for dir/ls/tree.\n"
        "Parameters: path (string, optional, default '.'), recursive (bool, optional, default false), "
        "glob (string, optional, default '*')."
    )
    auto_approve = True

    def execute(self, params, workdir=None):
        path = params.get("path", ".")
        recursive = params.get("recursive", False)
        glob_pat = params.get("glob", "*")
        base = os.path.join(workdir or ".", path)
        if not os.path.isdir(base):
            return f"Error: directory not found: {path}"
        from fnmatch import fnmatch
        results = []
        if recursive:
            for root, dirs, files in os.walk(base):
                for f in files:
                    if fnmatch(f, glob_pat):
                        rel = os.path.relpath(os.path.join(root, f), base)
                        results.append(rel.replace("\\", "/"))
        else:
            for entry in sorted(os.listdir(base)):
                full = os.path.join(base, entry)
                if os.path.isdir(full):
                    results.append(entry + "/")
                else:
                    if fnmatch(entry, glob_pat):
                        results.append(entry)
        if not results:
            return f"[Empty directory: {path}]"
        return f"[{path}: {len(results)} item(s)]\n" + "\n".join(results)


class FindFileTool:
    name = "find_file"
    description = "Find files by name pattern."
    system_prompt = (
        "## find_file\n"
        "Find files by name pattern. Cross-platform replacement for find/dir /s /b.\n"
        "Parameters: pattern (string, required), path (string, optional, default '.'), "
        "max_results (int, optional, default 50)."
    )
    auto_approve = True

    def execute(self, params, workdir=None):
        pattern = params.get("pattern", "*")
        path = params.get("path", ".")
        max_results = int(params.get("max_results", 50))
        base = os.path.join(workdir or ".", path)
        if not os.path.isdir(base):
            return f"Error: directory not found: {path}"
        from fnmatch import fnmatch
        results = []
        for root, dirs, files in os.walk(base):
            for f in files:
                if fnmatch(f, pattern):
                    rel = os.path.relpath(os.path.join(root, f), base)
                    results.append(rel.replace("\\", "/"))
                    if len(results) >= max_results:
                        break
            if len(results) >= max_results:
                break
        if not results:
            return f"[No files matching '{pattern}' in {path}]"
        return f"[Found {len(results)} file(s) matching '{pattern}']\n" + "\n".join(results)


class SearchInFilesTool:
    name = "search_in_files"
    description = "Search for a regex pattern across files."
    system_prompt = (
        "## search_in_files\n"
        "Search for a regex pattern across files. Cross-platform replacement for grep/findstr.\n"
        "Parameters: pattern (string, required), path (string, optional, default '.'), "
        "glob (string, optional, default '*'), max_results (int, optional, default 50)."
    )
    auto_approve = True

    def execute(self, params, workdir=None):
        pattern = params.get("pattern", "")
        path = params.get("path", ".")
        glob_pat = params.get("glob", "*")
        max_results = int(params.get("max_results", 50))
        if not pattern:
            return "Error: pattern is required"
        base = os.path.join(workdir or ".", path)
        from fnmatch import fnmatch
        try:
            regex = re.compile(pattern)
        except re.error as e:
            return f"Error: invalid regex: {e}"
        results = []
        for root, dirs, files in os.walk(base):
            for f in files:
                if not fnmatch(f, glob_pat):
                    continue
                full = os.path.join(root, f)
                try:
                    with open(full, "r", encoding="utf-8", errors="replace") as fh:
                        for lineno, line in enumerate(fh, 1):
                            if regex.search(line):
                                rel = os.path.relpath(full, base)
                                results.append(f"{rel}:{lineno}: {line.rstrip()}")
                                if len(results) >= max_results:
                                    return f"[Found {len(results)} match(es)]\n" + "\n".join(results)
                except Exception:
                    pass
        if not results:
            return f"[No matches for '{pattern}']"
        return f"[Found {len(results)} match(es)]\n" + "\n".join(results)


class PeekFileTool:
    name = "peek_file"
    description = "Show the beginning and end of a file."
    system_prompt = (
        "## peek_file\n"
        "Show the beginning and end of a file. Cross-platform replacement for head/tail.\n"
        "Parameters: path (string, required), head (int, optional, default 20), "
        "tail (int, optional, default 20)."
    )
    auto_approve = True

    def execute(self, params, workdir=None):
        path = params.get("path", "")
        head = int(params.get("head", 20))
        tail = int(params.get("tail", 20))
        if not path:
            return "Error: path is required"
        full = os.path.join(workdir or ".", path)
        if not os.path.isfile(full):
            return f"Error: file not found: {path}"
        try:
            with open(full, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
            total = len(lines)
            head_lines = lines[:head]
            tail_lines = lines[-tail:] if tail < total else []
            parts = [f"[File: {path} | {total} lines]"]
            parts.append("--- head ---")
            parts.extend(l.rstrip("\n") for l in head_lines)
            if tail_lines:
                parts.append("--- tail ---")
                parts.extend(l.rstrip("\n") for l in tail_lines)
            return "\n".join(parts)
        except Exception as e:
            return f"Error reading file: {e}"


class MkdirTool:
    name = "mkdir"
    description = "Create a directory."
    system_prompt = (
        "## mkdir\n"
        "Create a directory, including any missing parent directories.\n"
        "Parameters: path (string, required)."
    )
    auto_approve = True

    def execute(self, params, workdir=None):
        path = params.get("path", "")
        if not path:
            return "Error: path is required"
        full = os.path.join(workdir or ".", path)
        os.makedirs(full, exist_ok=True)
        return f"[Created directory: {path}]"


class CopyFileTool:
    name = "copy_file"
    description = "Copy a file or directory."
    system_prompt = (
        "## copy_file\n"
        "Copy a file or directory. Creates destination parent directories.\n"
        "Parameters: source (string, required), destination (string, required), "
        "overwrite (bool, optional, default false)."
    )
    auto_approve = True

    def execute(self, params, workdir=None):
        src = params.get("source", "")
        dst = params.get("destination", "")
        overwrite = params.get("overwrite", False)
        if not src or not dst:
            return "Error: source and destination are required"
        wd = workdir or "."
        src_full = os.path.join(wd, src)
        dst_full = os.path.join(wd, dst)
        if not os.path.exists(src_full):
            return f"Error: source not found: {src}"
        if os.path.exists(dst_full) and not overwrite:
            return f"Error: destination exists: {dst} (use overwrite=true)"
        os.makedirs(os.path.dirname(dst_full), exist_ok=True) if os.path.dirname(dst_full) else None
        try:
            if os.path.isdir(src_full):
                shutil.copytree(src_full, dst_full, dirs_exist_ok=overwrite)
            else:
                shutil.copy2(src_full, dst_full)
            return f"[Copied: {src} → {dst}]"
        except Exception as e:
            return f"Error copying: {e}"


class MoveFileTool:
    name = "move_file"
    description = "Move or rename a file or directory."
    system_prompt = (
        "## move_file\n"
        "Move or rename a file or directory. Creates destination parent directories.\n"
        "Parameters: source (string, required), destination (string, required), "
        "overwrite (bool, optional, default false)."
    )
    auto_approve = True

    def execute(self, params, workdir=None):
        src = params.get("source", "")
        dst = params.get("destination", "")
        overwrite = params.get("overwrite", False)
        if not src or not dst:
            return "Error: source and destination are required"
        wd = workdir or "."
        src_full = os.path.join(wd, src)
        dst_full = os.path.join(wd, dst)
        if not os.path.exists(src_full):
            return f"Error: source not found: {src}"
        if os.path.exists(dst_full) and not overwrite:
            return f"Error: destination exists: {dst} (use overwrite=true)"
        os.makedirs(os.path.dirname(dst_full), exist_ok=True) if os.path.dirname(dst_full) else None
        try:
            shutil.move(src_full, dst_full)
            return f"[Moved: {src} → {dst}]"
        except Exception as e:
            return f"Error moving: {e}"
