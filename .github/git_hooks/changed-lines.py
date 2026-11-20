"""
Hooks that format (black) or lint (flake8, all rules) only the changed lines of Python files

Legacy files do not follow these rules, so formatting or linting whole files is too noisy.

Usage: ``changed-lines.py {format,lint} [files]``. Without files, all changed ``.py`` files are processed.

Changed lines are those between HEAD and the working tree (in a pre-commit run, unstaged changes are stashed:
the staged lines plus any edits of the format hook). With PRE_COMMIT_FROM_REF and PRE_COMMIT_TO_REF set, e.g. by
``pre-commit run --from-ref origin/develop --to-ref HEAD``, they are those since the merge base of both refs
(PRE_COMMIT_TO_REF must be HEAD). Both hooks do nothing while a merge is in progress.
"""

import argparse
import collections
import os
import re
import subprocess
import sys
from typing import NoReturn

# Explicit output formats: user configs (e.g. diff.noprefix, flake8 format=) must not change what is parsed.
# Differs from the CI's full flake8 job: E203, E701 conflict with black (see black's docs, "Using Black with other
# tools"); no --max-complexity, as C901 is reported per function, not per changed line.
FLAKE8_ARGS = ["--max-line-length=127", "--extend-ignore=E203,E701", "--format=default"]
GIT_DIFF_ARGS = ["-U0", "--no-color", "--no-ext-diff", "--no-relative", "--src-prefix=a/", "--dst-prefix=b/"]


def fail(message) -> NoReturn:
    """Exit with an error: unexpected output must block the commit, not pass silently."""
    print(f"changed-lines: {message}")
    sys.exit(1)


def git(*args):
    # errors="replace": changed files may not be UTF-8 (only paths and hunk headers are parsed)
    return subprocess.check_output(["git"] + list(args)).decode(errors="replace")


def diff_refs():
    """Return the `git diff` arguments selecting the changes to process."""
    from_ref, to_ref = os.environ.get("PRE_COMMIT_FROM_REF"), os.environ.get("PRE_COMMIT_TO_REF")
    if not (from_ref and to_ref):
        return ["HEAD"]
    if git("rev-parse", to_ref) != git("rev-parse", "HEAD"):
        fail(f"--to-ref {to_ref} is not the checked-out commit (HEAD), but the hooks process the working tree")
    # compare to the working tree, so that line numbers include the edits of the format hook
    return ["--merge-base", from_ref]


def changed_hunks():
    """Return {file: list of (first line, number of lines)} of the added/changed lines (new file version)."""
    diff = git("diff", *GIT_DIFF_ARGS, *diff_refs())

    hunks = collections.defaultdict(list)
    filename = None
    for line in diff.splitlines():
        if line.startswith("+++ "):
            if line == "+++ /dev/null":  # deleted file
                filename = None
            elif line.startswith("+++ b/"):
                filename = line[len("+++ b/") :].rstrip("\t")  # git appends a tab to paths with spaces
            else:  # e.g. a quoted path with special characters
                fail(f"can not parse the diff line {line!r}")
        elif line.startswith("@@") and filename:
            match = re.match(r"@@ -\S+ \+(\d+)(?:,(\d+))? @@", line)
            if match is None:
                fail(f"can not parse the diff line {line!r}")
            hunks[filename].append((int(match.group(1)), int(match.group(2) or 1)))

    return hunks


def format_files(filenames, hunks):
    """Run black on the changed lines; black takes line ranges only for a single file per call."""
    returncode = 0
    for filename in filenames:
        ranges = [f"--line-ranges={start}-{start + count - 1}" for start, count in hunks[filename] if count]
        if ranges:
            returncode |= subprocess.run([sys.executable, "-m", "black"] + ranges + [filename]).returncode

    return returncode


def lint_files(filenames, hunks):
    """Run flake8 and print the violations on changed lines as warnings (exit code 0, like the CI)."""
    lines = collections.defaultdict(set)
    for filename in filenames:
        for start, count in hunks[filename]:
            # +1: blank-line errors (E30x) are reported at the statement after the inserted/removed lines
            lines[filename].update(range(start, start + count + 1))

    result = subprocess.run([sys.executable, "-m", "flake8"] + FLAKE8_ARGS + filenames, capture_output=True, text=True)
    if result.returncode and not result.stdout:  # flake8 failed (e.g. not installed): do not pass silently
        print(result.stderr)
        return result.returncode

    violations = []
    for line in result.stdout.splitlines():  # <file>:<line>:<column>: <code> <message>
        match = re.match(r"(.+?):(\d+):(\d+): (\S+) (.*)", line)
        if match is None:
            fail(f"can not parse the flake8 output line {line!r}")
        filename, lineno, column, code, message = match.groups()
        if int(lineno) in lines[filename]:
            if os.environ.get("GITHUB_ACTIONS") == "true":  # annotation shown in the PR diff
                line = f"::warning file={filename},line={lineno},col={column},title=flake8 {code}::{message}"
            violations.append(line)

    if violations:
        print("\n".join(violations) + f"\n\nWarning: {len(violations)} flake8 violation(s) on changed lines.")

    return 0


def main():
    parser = argparse.ArgumentParser(description="Format (black) or lint (flake8) only the changed lines of Python files")
    parser.add_argument("mode", choices=["format", "lint"])
    parser.add_argument("files", nargs="*")
    args = parser.parse_args()

    if subprocess.run(["git", "rev-parse", "-q", "--verify", "MERGE_HEAD"], capture_output=True).returncode == 0:
        # the diff to HEAD contains all lines of the merged branch, not only those changed by the user
        print("changed-lines: skipped while a merge is in progress")
        return 0

    hunks = changed_hunks()
    filenames = args.files or [filename for filename in hunks if filename.endswith(".py")]
    filenames = [filename for filename in filenames if filename in hunks]
    if not filenames:
        return 0

    if args.mode == "format":
        return format_files(filenames, hunks)

    return lint_files(filenames, hunks)


if __name__ == "__main__":
    sys.exit(main())
