"""
Hook that runs the full flake8 rule set, but reports only violations on changed lines

Most legacy files do not pass all flake8 rules, so linting whole files is too noisy to block on.
Uses the same settings as the (non-blocking) "unimportant" flake8 job of the CI.

Changed lines are the staged ones (`git diff --cached`), or those between
PRE_COMMIT_FROM_REF and PRE_COMMIT_TO_REF, e.g. for
``pre-commit run flake8-changed-lines --from-ref origin/develop --to-ref HEAD``.
"""
import collections
import os
import re
import subprocess
import sys

FLAKE8_ARGS = ["--max-line-length=127"]


def changed_lines():
    """ Return {file: set of changed line numbers} (new file version). """
    from_ref, to_ref = os.environ.get("PRE_COMMIT_FROM_REF"), os.environ.get("PRE_COMMIT_TO_REF")
    refs = [f"{from_ref}...{to_ref}"] if from_ref and to_ref else ["--cached"]
    diff = subprocess.check_output(["git", "diff", "-U0", "--no-color"] + refs).decode()

    lines = collections.defaultdict(set)
    filename = None
    for line in diff.splitlines():
        if line.startswith("+++ "):
            filename = line[len("+++ b/"):] if line.startswith("+++ b/") else None
        elif line.startswith("@@") and filename:
            start, count = re.match(r"@@ -\S+ \+(\d+)(?:,(\d+))? @@", line).groups()
            start, count = int(start), int(count or 1)
            # +1: blank-line errors (E30x) are reported at the statement after the inserted lines
            lines[filename].update(range(start, start + count + 1))

    return lines


def main(filenames):
    lines = changed_lines()
    filenames = [filename for filename in filenames if filename in lines]
    if not filenames:
        return 0

    result = subprocess.run(
        [sys.executable, "-m", "flake8"] + FLAKE8_ARGS + filenames, capture_output=True, text=True)
    if result.returncode and not result.stdout:  # flake8 failed (e.g. not installed): do not pass silently
        print(result.stderr)
        return result.returncode

    violations = []
    for line in result.stdout.splitlines():  # <file>:<line>:<column>: <code> <message>
        match = re.match(r"(.+?):(\d+):\d+: ", line)
        if match and int(match.group(2)) in lines[match.group(1)]:
            violations.append(line)

    if violations:
        print("\n".join(violations) + f"\n\n{len(violations)} flake8 violation(s) on changed lines.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
