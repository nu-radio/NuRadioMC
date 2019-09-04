from functools import wraps
from timeit import default_timer as timer

def run_decorator(run):
    @wraps(run)
    
    def register_run_method(self, evt, station, det, **kwargs):
        evt.register_module(self, self.__class__.__name__, kwargs)
        start = timer()
        res =  run(self, evt, station, det, **kwargs)
        end = timer()
        if not self in register_run_method.time:
            register_run_method.time[self] = 0
        register_run_method.time[self] += (end - start)
        return res
    register_run_method.time = {}
    return register_run_method
       
        


# class ModuleDecorator(object):
#     def __init__(self, klas):
#         self.klas = klas
#         self.org_run = self.klas.run
#         self.klas.run = self.run
# 
#     def __call__(self, *arg, **kwargs):
#         print("calling __call__")
#         return self.klas.__call__(*arg, **kwargs)
# 
#     def run(self, *args, **kwargs):
#         print("registry")
#         args[0].register_module(self.klas.__class__.__name__, kwargs)
#         start = timer()
#         print(args)
#         self.org_run(self.klas, *args, **kwargs)
#         end = timer()
#         print(end - start)