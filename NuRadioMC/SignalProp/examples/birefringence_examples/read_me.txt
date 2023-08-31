Example scripts for simulations including birefringence.

starting_pulses:    
                    Data measured for the SPice emitter in an annecoic chamber.
                    The data was taken from https://github.com/ggaswint/ARIANNAanalysis/tree/master/data/AnechoicChamberData/EFieldData
                    The measurements were made for different launch angles (careful, different definition as in NuRadioMC).
                    There were 10 measurements of the electric field per launch angles.
                    Each file is a numpy array with shape (3, trace_length). [0] corresponds to the time stamps. [1] corresponds to the theta component. [2] corresponds to the phi component. 

01_simple_propagation.py :





02_path_info.py 
03_ARA_SPice.py 
04_ARIANNA_SPice.py 
05_Veff_comparison.py