
"""
Hook that checks if .ipynb files have been stripped of outputs before committing

This prevents constantly adding large changes to the repository.
If the user tries to commit notebooks with output, the commit
is rejected and a helpful error message is raised.

Inspired by / adapted from https://stackoverflow.com/a/74753885
"""
import subprocess
import json

filenames = subprocess.check_output('git diff --name-only --cached'.split())
filenames = filenames.decode().split('\n')

bad_files = []

for filename in filenames:
    if filename.endswith('.ipynb'):
        with open(filename, 'r') as f:
            ipyjson = json.load(f)
            for cell in ipyjson['cells']:
                if cell.get('outputs'): # cell contains non-zero output
                    bad_files.append(filename)
                    break

if len(bad_files):
    print(
        "One or more .ipynb files you are trying to commit still contain outputs:\n\n"
        + '\n'.join(["\t" + filename for filename in bad_files])
        + "\n\nPlease fix this by running\n\n"
        + "\t" + "jupyter nbconvert --clear-output --inplace <file>\n\n"
        + "for each file before adding them to your commit."
    )
    exit(1)
