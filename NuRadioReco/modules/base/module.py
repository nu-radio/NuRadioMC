from functools import wraps
from timeit import default_timer as timer

def register_run(level):
    if level not in ["station", "event"]:
        raise NotImplementedError("The level needs to be either 'station' or 'event'")
    def run_decorator(run):
        @wraps(run)
        def register_run_method(self, evt, station, det, **kwargs):
            if(level == "event"):
                evt.register_module(self, self.__class__.__name__, kwargs)
            elif(level == "station"):
                station.register_module(self, self.__class__.__name__, kwargs)
            start = timer()
            res =  run(self, evt, station, det, **kwargs)
            end = timer()
            if not self in register_run_method.time:
                register_run_method.time[self] = 0
            register_run_method.time[self] += (end - start)
            return res
        register_run_method.time = {}
        
        return register_run_method
    return run_decorator


       


def time_this(original_function):      
    print("decorating")
    def new_function(*args,**kwargs):
        print("starting timer")
        import datetime                 
        before = datetime.datetime.now()                     
        x = original_function(*args,**kwargs)                
        after = datetime.datetime.now()                      
        print("Elapsed Time = {0}".format(after-before))
        return x                                             
    return new_function  

def ModuleDecorator(Cls):
    class NewCls(object):
        def __init__(self,*args,**kwargs):
            self.oInstance = Cls(*args,**kwargs)
        def __getattribute__(self,s):
            """
            this is called whenever any attribute of a NewCls object is accessed. This function first tries to 
            get the attribute off NewCls. If it fails then it tries to fetch the attribute from self.oInstance (an
            instance of the decorated class). If it manages to fetch the attribute from self.oInstance, and 
            the attribute is an instance method then `time_this` is applied.
            """
            try:    
                x = super(NewCls,self).__getattribute__(s)
            except AttributeError:      
                pass
            else:
                return x
            print(s)
            x = self.oInstance.__getattribute__(s)
            print(x)
            if type(x) == type(self.__init__): # it is an instance method
                return run_decorator(x)                 # this is equivalent of just decorating the method with time_this
            else:
                return x
    return NewCls


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