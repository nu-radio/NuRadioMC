import os
import sys
import shutil
import subprocess
import logging
import tempfile
import argparse
logging.basicConfig()

logger = logging.getLogger('NuRadioMC-install-dev')
logger.setLevel(logging.INFO)

__doc__ = """
An interactive script to install NuRadioMC for developers.

It can be used to install all core and most optional dependencies,
add the NuRadioMC path to your PYTHONPATH, and install a git hook
to prevent accidental commits of large files to the NuRadioMC repository.

Invoking ``python install_dev.py`` without options will launch the 
installer in interactive mode - the user will be prompted before every
potential modification. If you want to run the script non-interactively
in the command line, ``python install_dev.py --help`` gives an overview
of the different options; 
``python install_dev.py [--options] --no-interactive`` will run only
the options specified by --options and skip everything else without 
prompts.

"""


def yesno_input(msg='', skip=None):
    """Interactive yes/no input

    If skip is not None, no input is requested and
    skip is returned instead.
    """
    if not skip == None:
        print(msg + " (selected: {})".format(skip))
        return skip
    while True:
        arg = input(msg+'\n[Y]es / [N]o\n')
        if arg.upper() in ["Y", "YES"]:
            return True
        elif arg.upper() in ["N", "NO"]:
            return False
        else:
            print("Invalid command entered")


def convert_version_number(version, to_string=False):
    """Convert poetry version requirement to a pip-compatible string"""
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

def convert_poetry_to_pip(reqs):
    """Converts a list of poetry-style requirements to pip-compatible ones"""
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
            if 'git' in version:
                req = 'git+{}'.format(version['git'])
                if 'branch' in version:
                    req+='@{}'.format(version['branch'])
                elif 'rev' in version:
                    req+='@{}'.format(version['rev'])
                elif 'tag' in version:
                    req+='@{}'.format(version['tag'])
            if 'version' in version:
                version = version['version']
            else:
                version = '*'
        version = convert_version_number(version)
        reqs_pip.append(''.join([req, version, cond]))
    return reqs_pip

if __name__ == "__main__":
    # By default, this script will run interactively. One can also run parts 
    # non-interactively by using the following command line arguments
    argparser = argparse.ArgumentParser(
        prog = "NuRadioMC.install_dev.py", 
        description="Script to install NuRadioMC dependencies for developers"
    )
    argparser.add_argument(
        "--install", action="store_true", default=None, 
        help="Install NuRadioMC core dependencies.")
    argparser.add_argument("--no-install", action="store_false", dest="install")

    argparser.add_argument(
        "--dev", default=None, nargs='*',
        help=(
            "Install NuRadioMC optional / dev dependencies. " 
            "A list of optional features, as defined in pyproject.toml"
            " under [tool.poetry.extras], separated by spaces. To "
            "install all dev dependencies, use \"--dev all\" ")
    )
    argparser.add_argument("--no-dev", action="store_false", dest="dev")

    argparser.add_argument("--pythonpath", action="store_true", default=None, 
        help="Add NuRadioMC to the PYTHONPATH in .bashrc")
    argparser.add_argument("--no-pythonpath", action="store_false", dest="pythonpath")

    argparser.add_argument(
        "--git-hook", action="store_true", default=None, 
        help="Install git pre-commit hook to prevent pushing large files to the repository on accident.")
    argparser.add_argument("--no-git-hook", action="store_false", dest="git_hook")

    argparser.add_argument(
        "-U", "--user", help="Append '--user' to pip installs (relevant only if not installing to a virtual env)",
        action="store_true")

    argparser.add_argument(
        "--interactive", action="store_true", default=True,
        help="Use interactive installer if no command line options are given (this is true by default).")
    argparser.add_argument("--no-interactive", action="store_false", dest="interactive")
    args = vars(argparser.parse_args())
    
    if args["interactive"] == False: # never use interactive installer
       for key in args.keys():
           if args[key] == None:
               args[key] = False

    top_dir = os.path.dirname(os.path.realpath(__file__))
    os.chdir(top_dir) # change CWD to repository root
    retcode = 0

    if '.git' in os.listdir(top_dir): # check that we are in a git repository
        ### Install dependencies
        if args["user"]:
            pip_install_as_user = ["--user"]
        else:
            pip_install_as_user = []
        install_dependencies = yesno_input("Install NuRadioMC dependencies? (Warning: this may take several minutes)", skip=args['install'])
        if install_dependencies:
            # print("Installing poetry...")
            # subprocess.call([sys.executable, "-m", "pip", "install", "poetry"])
            # subprocess.call(["poetry", "install", "--no-root"])
            retcode |= subprocess.call([sys.executable, '-m', 'pip', 'install', 'toml']+ pip_install_as_user) # we need toml to read pyproject.toml
            import toml
            toml_dict = toml.load(os.path.join(top_dir, 'pyproject.toml'))
            reqs = toml_dict['tool']['poetry']['dependencies']
            reqs_pip = convert_poetry_to_pip(reqs)

            # install the requirements using pip
            with tempfile.NamedTemporaryFile(mode='w+t') as req_txt: # make a temporary requirements.txt
                req_txt.writelines('\n'.join(reqs_pip))
                req_txt.seek(0)
                retcode |= subprocess.call([sys.executable, '-m', 'pip', 'install', '-r', req_txt.name] + pip_install_as_user)
        
        ### Install optional / dev dependencies
        install_dev_dependencies = yesno_input(
            "Install optional/dev dependencies?\n(If yes, shows an interactive list of available dependencies that can be installed)",
             skip=args['dev'])
        if install_dev_dependencies:
            try:
                import toml
            except ImportError:
                retcode |= subprocess.call([sys.executable, '-m', 'pip', 'install', 'toml'] + pip_install_as_user) # we need toml to read pyproject.toml
                import toml
            toml_dict = toml.load(os.path.join(top_dir, 'pyproject.toml'))
            reqs = toml_dict['tool']['poetry']['dev-dependencies']
            extras = toml_dict['tool']['poetry']['extras']
            header = "{:4s}|{:12s}|{:16s}|{}\n".format("id", "Install?", "extra", "modules")
            str_format = "{:4s}|{:12s}|{:16s}|{}\n"
            selected_for_install = []
            header = str_format.format("id", "Install?", "feature", "modules")
            footer = (
                "\nTo (un)select modules for installation, enter the module ids, "
                "separated by spaces.\n To install, include -i. To select all, include -all. "
                "To cancel without installing any modules, include -c\n\n"
                "E.g. \"1 2 4 -i\" selects modules 1, 2 and 4 and installs them.\n"
            )

            while (install_dev_dependencies == True): # this is skipped if --dev [modules] was given in the command line
                install_table = [header]
                for i,key in enumerate(extras.keys(), start=1):
                    install_table.append(
                        str_format.format(str(i), ["No", "Yes"][str(i) in selected_for_install], key, extras[key])
                    )
                install_table.append(footer)
                print(''.join(install_table))
                user_input = input()
                user_input = user_input.split()
                for i in user_input:
                    if i.upper() in selected_for_install:
                        selected_for_install.remove(i.upper())
                    else:
                        selected_for_install.append(i.upper())
                if ("-C" in selected_for_install) or ("C" in selected_for_install):
                    install_dev_dependencies = []
                if ("-ALL" in selected_for_install):
                    selected_for_install += [str(i) for i in range(1, len(extras)+1)]
                    selected_for_install.remove("-ALL")
                if ("-I" in selected_for_install) or ("I" in selected_for_install):
                    install_dev_dependencies = selected_for_install
        
            install_modules = []
            install_extras = []
            for i,key in enumerate(extras.keys(), start=1):
                install_extra = any([
                    (str(i) in install_dev_dependencies),
                    (key in install_dev_dependencies),
                    ('ALL' in [j.upper() for j in install_dev_dependencies])
                ])
                if install_extra:
                    install_modules += extras[key]
                    install_extras.append(key)
            
            reqs = dict([(req, reqs[req]) for req in reqs if req in install_modules])
            reqs_pip = convert_poetry_to_pip(reqs)

            # install the requirements using pip
            print("Installing the following features: {}".format(install_extras))
            with tempfile.NamedTemporaryFile(mode='w+t') as req_txt: # make a temporary requirements.txt
                req_txt.writelines('\n'.join(reqs_pip))
                req_txt.seek(0)
                retcode |= subprocess.call([sys.executable, '-m', 'pip', 'install', '-r', req_txt.name] + pip_install_as_user)

        ### Add NuRadioMC to PYTHONPATH in .bashrc, if not already available
        try:
            current_pythonpath = os.environ["PYTHONPATH"]
        except KeyError:
            current_pythonpath = ""
        check_pythonpath = [path for path in current_pythonpath.split(':') if len(path)>0] # the current directory doesn't count!
        if top_dir not in [os.path.realpath(j) for j in check_pythonpath]:
            add_NuRadioMC_to_pythonpath = yesno_input("{} not yet in PYTHONPATH. Add to user .bashrc?".format(top_dir), skip=args['pythonpath'])
            if add_NuRadioMC_to_pythonpath:
                try:
                    bashrc_path = os.path.expanduser("~/.bashrc")
                    with open(bashrc_path, 'a+') as bashrc_file:
                        exportline = "export PYTHONPATH=$PYTHONPATH:{}".format(top_dir)
                        bashrc_file.seek(0)
                        line_in_bashrc = any([ # check if we / the user has already updated .bashrc
                            ('export' in k) & ('PYTHONPATH' in k) & (top_dir in k) 
                            for k in bashrc_file.readlines()
                        ])
                        if line_in_bashrc:
                            print("PYTHONPATH already updated in .bashrc. Try relaunching the shell.")
                        else:
                            bashrc_file.write("export PYTHONPATH=$PYTHONPATH:{}".format(top_dir))
                except OSError as e:
                    print("Failed to add {} to .bashrc. Please manually add it to your PYTHONPATH.".format(top_dir))

        ### Write pre-commit hook
        write_pre_commit_hook = yesno_input(
            (
                "Install pre-commit hook (recommended for developers)?\n"
                "This prevents large files being accidentally committed to the repository."
            ), skip=args['git_hook'])
        if write_pre_commit_hook:
            old_file = os.path.join(top_dir,'.github/git_hooks/pre-commit')
            new_file = os.path.join(top_dir,'.git/hooks/pre-commit')
            if os.path.exists(new_file): # if user has a pre-commit hook already, confirm before overwriting
                write_pre_commit_hook = yesno_input("Custom pre-commit file already present at {}. Overwrite?".format(new_file), skip=args['git_hook'])
        if write_pre_commit_hook:
            shutil.copy(old_file, new_file)
            retcode |= subprocess.call(['chmod', '+x', new_file])
            print('Successfully installed pre-commit hook at {}'.format(new_file))
    else:
        msg = (
            'No git repository detected. If this is incorrect, and you are using '
            'the developer version please follow the manual installation '
            'instructions at https://nu-radio.github.io/NuRadioMC/Introduction/pages/installation.html#manual-installation'
        )
        print(msg)

    sys.exit(retcode)