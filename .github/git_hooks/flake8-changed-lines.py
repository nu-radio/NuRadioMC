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

# Explicit output formats: user configs (e.g. diff.noprefix, flake8 format=) must not change what is parsed
FLAKE8_ARGS = ["--max-line-length=127", "--format=default"]
GIT_DIFF_ARGS = ["-U0", "--no-color", "--no-ext-diff", "--src-prefix=a/", "--dst-prefix=b/"]


def fail(message):
    """ Exit with an error: unexpected output must block the commit, not pass silently. """
    print(f"flake8-changed-lines: {message}")
    sys.exit(1)


def changed_lines():
    """ Return {file: set of changed line numbers} (new file version). """
    from_ref, to_ref = os.environ.get("PRE_COMMIT_FROM_REF"), os.environ.get("PRE_COMMIT_TO_REF")
    refs = [f"{from_ref}...{to_ref}"] if from_ref and to_ref else ["--cached"]
    diff = subprocess.check_output(["git", "diff"] + GIT_DIFF_ARGS + refs).decode()

    lines = collections.defaultdict(set)
    filename = None
    for line in diff.splitlines():
        if line.startswith("+++ "):
            if line == "+++ /dev/null":  # deleted file
                filename = None
            elif line.startswith("+++ b/"):
                filename = line[len("+++ b/"):]
            else:  # e.g. a quoted path with special characters
                fail(f"can not parse the diff line {line!r}")
        elif line.startswith("@@") and filename:
            match = re.match(r"@@ -\S+ \+(\d+)(?:,(\d+))? @@", line)
            if match is None:
                fail(f"can not parse the diff line {line!r}")
            start, count = int(match.group(1)), int(match.group(2) or 1)
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
        if match is None:
            fail(f"can not parse the flake8 output line {line!r}")
        if int(match.group(2)) in lines[match.group(1)]:
            violations.append(line)

    if violations:
        print("\n".join(violations) + f"\n\n{len(violations)} flake8 violation(s) on changed lines.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
