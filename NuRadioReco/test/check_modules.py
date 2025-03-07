#! /usr/bin/env python3


# This script will walk the module directory, load everything that looks like a
# module and try to import (possibly failing if for example, you don't have all
# the optional things for each module)

# Then it will find all classes defined in the module with a run method, and
# complain if the time attribute is not present (this is an attribute added by
# the register_run decorator) We also check for classes that have run defined but no begining/end

# It prints out a bunch of chatty output, then at the end, prints a list of
# modules that couldn't be imported, classes that have run but no decorator,
# and classes that have run but no begin/end.
#




import os
import pathlib
import inspect
import importlib
import argparse


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description='Simple script to check NuRadioReco classes in /modules conform to the convention.'
        )
    argparser.add_argument('-r', '--run', action='store_true', help="Raise error on missing/broken `run` method.")
    argparser.add_argument('-b', '--begin', action='store_true', help="Raise error on missing/broken `begin` method.")
    argparser.add_argument('-e', '--end', action='store_true', help="Raise error on missing/broken `end` method.")
    argparser.add_argument('-i', '--import',action='store_true', help="Raise error if module failed to import.", dest='broken') # args.import is not a valid name

    args = argparser.parse_args()

    broken = []
    unregistered_runs = []
    missing_begin = []
    missing_end = []

    # switch to the NuRadioMC parent directory
    os.chdir(pathlib.Path(__file__).parents[2])

    for dirpath,folders,files in os.walk(os.path.join('NuRadioReco','modules')):
        for f in files:
            if not f.endswith(".py") or f == "__init__.py":
                continue
            try:
                mname = dirpath.replace('/','.')+'.'+f[:-3]
                print("Trying ", mname)

                m = importlib.import_module(mname)

                for name,obj in inspect.getmembers(m, lambda member: inspect.isclass(member) and member.__module__ == mname):
                    print("Found class ",name, obj)
                    if hasattr(obj,'run') and not hasattr(obj.run,'time'):
                        print('Has run method but not registered properly! Public flogging will be scheduled.')
                        unregistered_runs.append(mname + '.' +name)

                    if hasattr(obj,'run') and not hasattr(obj,'begin'):
                        print ('Has run but no begin...')
                        missing_begin.append(mname + '.' + name)
                    if hasattr(obj,'run') and not hasattr(obj,'end'):
                        print ('Has run but no end...')
                        missing_end.append(mname + '.' + name)


            except Exception as e:
                print("Couldn't load module... maybe it's broken, oh well. Exception below:")
                print('\t', e)
                broken.append(mname)



    print("\n\n\n.........................................\n")
    print ("Broken modules:\n\t" + '\n\t'.join(broken))
    print()
    print ("Unregistered runs:\n\t" + '\n\t'.join(unregistered_runs))
    print()
    print ("Missing end:\n\t" + '\n\t'.join(missing_end))
    print()
    print ("Missing begin:\n\t" + '\n\t'.join(missing_begin))

    exit_code = (
        bool(args.run and len(unregistered_runs))
        + 2 * bool(args.begin and len(missing_begin))
        + 4 * bool(args.end and len(missing_end))
        + 8 * bool(args.broken and len(broken))
    )

    if exit_code: # it seems sys.exit(0) is sometimes still treated as an error, so we avoid calling it if there was no error.
        print('\n\n' + 80*'!' + '\n' + f"One or more problems found, exiting with code {exit_code}")
        exit(exit_code)











