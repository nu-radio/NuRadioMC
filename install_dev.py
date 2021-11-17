import os
import sys
import shutil
import subprocess
import logging
import tempfile

logger = logging.getLogger('NuRadioMC-install-dev')
logger.setLevel(logging.INFO)

top_dir = os.path.dirname(os.path.realpath(__file__))#.split('.github')[0] #TODO - finalize path
os.chdir(top_dir) # change CWD to repository root

def yesno_input(msg=''):
    while True:
        arg = input(msg+'\n[Y]es / [N]o\n')
        if arg in ["Y", "y"]:
            return True
        elif arg in ["N", "n"]:
            return False
        else:
            print("Invalid command entered")

def convert_version_number(version, to_string=False):
    if version == '*':
        return ''
    elif '^' in version or '~' in version:
        logger.warning(
            'Dependency specification using "^" or "~" not allowed in pip install, replacing by ">="'
        )
        version = version.replace('^', '>=').replace('~', '>=')
    elif version.lstrip('><= \t') == version:
        version = '=={}'.format(version)
    if to_string and version != '':
        version = (
            ''.join([char for char in version if char in '<>='])
            + '"{}"'.format(version.lstrip('<>= \t'))
        )
    return version

if '.git' in os.listdir(top_dir): # check that we are in a git repository
    ### Install dependencies
    install_dependencies = yesno_input("Install NuRadioMC dependencies? (Warning: this may take several minutes)")
    if install_dependencies:
        # print("Installing poetry...")
        # subprocess.call([sys.executable, "-m", "pip", "install", "poetry"])
        # subprocess.call(["poetry", "install", "--no-root"])
        subprocess.call([sys.executable, '-m', 'pip', 'install', 'toml']) # we need toml to read pyproject.toml
        import toml
        toml_dict = toml.load(os.path.join(top_dir, 'pyproject.toml'))
        reqs = toml_dict['tool']['poetry']['dependencies']
        reqs_pip = []
        for req in reqs: # we convert the requirements to the pip requirements.txt format
            cond = ''
            if req == 'python':
                continue
            version = reqs[req]
            if not type(version) == str:
                if 'python' in version:
                    cond = '; python_version {}'.format(
                        convert_version_number(version['python'], to_string=True)
                    )
                version = version['version']
            version = convert_version_number(version)
            reqs_pip.append(''.join([req, version, cond]))
        with tempfile.NamedTemporaryFile(mode='w+t') as req_txt: # make a temporary requirements.txt
            req_txt.writelines('\n'.join(reqs_pip))
            req_txt.seek(0)
            subprocess.call([sys.executable, '-m', 'pip', 'install', '-r', req_txt.name])

    ### Add NuRadioMC to PYTHONPATH in .bashrc, if not already available
    try:
        current_pythonpath = os.environ["PYTHONPATH"].split(":")
    except KeyError:
        current_pythonpath = []
    if top_dir not in [os.path.realpath(j) for j in current_pythonpath]:
        add_NuRadioMC_to_pythonpath = yesno_input("{} not yet in PYTHONPATH. Add to user .bashrc?".format(top_dir))
        if add_NuRadioMC_to_pythonpath:
            try:
                bashrc_path = os.path.expanduser("~/.bashrc")
                with open(bashrc_path, 'a') as bashrc_file:
                    bashrc_file.write("export PYTHONPATH=$PYTHONPATH:{}".format(top_dir))
            except OSError as e:
                print("Failed to add {} to .bashrc. Please manually add it to your PYTHONPATH.".format(top_dir))

    ### Write pre-commit hook
    write_pre_commit_hook = yesno_input("Install pre-commit hook (recommended for developers)?")
    if write_pre_commit_hook:
        old_file = os.path.join(top_dir,'.github/git_hooks/pre-commit')
        new_file = os.path.join(top_dir,'.git/hooks/pre-commit')
        if os.path.exists(new_file): # if user has a pre-commit hook already, confirm before overwriting
            write_pre_commit_hook = yesno_input("Custom pre-commit file already present at {}. Overwrite?".format(new_file))
    if write_pre_commit_hook:
        shutil.copy(old_file, new_file)
        subprocess.call(['chmod', '+x', new_file])
        print('Successfully installed pre-commit hook at {}'.format(new_file))
    # else:
    #     print('Pre-commit hook installation aborted. No hooks installed.')
else:
    msg = (
        'No git repository detected. If this is incorrect, and you are using '
        'the developer version please follow the manual installation '
        'instructions at (...)'
    )
    print(msg)