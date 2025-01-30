#! /usr/bin/env python3



# module and try to import (possibly failing if for example, you don't have all
# the optional things for each module)

# Then it will find all classes defined in
# the module with a run method, and complain if the time attribute is not
# present (this is an attribute added by the register_run decorator)

import os
import inspect
import importlib


broken = []
unregistered_runs = []
missing_begin = []
missing_end = []

for dirpath,folders,files in os.walk('NuRadioReco/modules'):
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
            print(e)
            print("Couldn't load module... maybe it's broken, oh well")
            broken.append(mname)



print("\n\n\n.........................................\n")
print ("Broken modules: ", broken)
print()
print ("Unregistered runs: ", unregistered_runs)
print()
print ("Missing end: ", missing_end)
print()
print ("Missing begin: ", missing_begin)









