import numpy as np
from NuRadioMC.simulation import simulation
import os
import multiprocessing
from multiprocessing import Queue
import time


class NuRadioMCRunner(object):
    """
    The purpose of this classe is to run NuRadioMC on a cluster using a full node submission, i.e., a full node with X cores is reserved. 
    As NuRadioMC is a single-core process, this class will start multiple NuRadioMC simulations and will distribute them on the X cores. As
    soon as one job is finished, a new one is started so that all X cores are kept busy. 
    The job ends after a fixed time, or after enough triggers are acquired, or after X crashes occurred.
    """

    def __init__(self, n_worker, task, output_path, kwargs={}, max_runtime=3600 * 24 * 8, n_triggers_max=1e6, max_crashes=10):
        self.q = Queue()
        self.task = task
        self.n_worker = n_worker
        self.i_task = 0
        self.worker = []
        self.n_triggers = 0
        self.iworker = 0
        self.i_skipped = 0
        self.stop_processing = False
        self.max_runtime = max_runtime
        self.n_triggers_max = n_triggers_max
        self.output_path = output_path
        self.kwargs = kwargs
        self.start_time = time.time()
        self.crashed_counter = 0
        self.crashed_counter = 0
        self.max_crashes = max_crashes

    def get_outputfilename(self):
        """
        define how output files are named
        """
        return os.path.join(self.output_path, f"{np.log10(self.kwargs['nu_energy']):.2f}_{self.i_task:06d}.hdf5")

    def run(self):

        while True:
            outputfilename = self.get_outputfilename()
            self.kwargs["output_filename"] = outputfilename
            if(os.path.exists(outputfilename)):
                print(f"outputfile {outputfilename} for task {self.i_task} already exists", flush=True)
            elif(os.path.exists(outputfilename + ".nur")):
                print(f"outputfile {outputfilename}.nur for task {self.i_task} already exists", flush=True)
            else:
                print(f"starting job {self.i_task}", flush=True)
                n = multiprocessing.Process(name=f'worker-{self.i_task}', target=self.task, args=(self.q, self.i_task), kwargs=self.kwargs)
                n.start()
                self.worker.append(n)
                self.iworker += 1
            self.i_task += 1
            if(self.iworker >= self.n_worker):
                break

        stop_processing = False
        while True:
            # check on all processes
            for iN, n in enumerate(self.worker):
                if not n.is_alive():
                    print(f"job is not alive, getting results {n}, exitcode = {n.exitcode}", flush=True)
                    n_trig = 0
                    if(n.exitcode != 0):
                        self.crashed_counter += 1
                    else:
                        if(not self.q.empty()):
                            n_trig = self.q.get_nowait()
                            self.n_triggers += n_trig
                    print(f"{iN} has finished with {n_trig} events, total number of triggered events is {self.n_triggers}", flush=True)
                    outputfilename = self.get_outputfilename()
                    self.kwargs["output_filename"] = outputfilename
                    if(os.path.exists(outputfilename)):
                        print(f"outputfile {outputfilename} for task {self.i_task} already exists", flush=True)
                    else:
                        n = multiprocessing.Process(name=f'worker-{self.i_task}', target=self.task, args=(self.q, self.i_task), kwargs=self.kwargs)
                        n.start()
                        self.worker[iN] = n
                    self.i_task += 1
            time.sleep(10)
            if(self.crashed_counter > self.max_crashes):
                print(f"more than {self.max_crashes} jobs crashed. Exiting...", flush=True)
                stop_processing = True
            if(stop_processing or self.n_triggers > self.n_triggers_max or ((time.time() - self.start_time) > self.max_runtime)):
                if(self.n_triggers > self.n_triggers_max):
                    print(f"more than {self.n_triggers_max} triggers, waiting for workers to stop \n\n\n\n\n\n", flush=True)
                else:
                    print(f"{simulation.pretty_time_delta(time.time()-self.start_time)} passed. No more jobs will be submitted.", flush=True)
                for iN, n in enumerate(self.worker):
                    if(not self.q.empty()):
                        n_trig = self.q.get()
                        self.n_triggers += n_trig
                    n.join()
                    print(f"{iN} has finished with {n_trig} events, total number of triggered events is {self.n_triggers}", flush=True)
                break
