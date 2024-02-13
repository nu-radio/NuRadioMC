Example scripts for simulations including birefringence.


01_simple_propagation.py :
    Using a simple pulse model, this example provides the opportunity to use and compare the birefringence model for different ray-tracers and ice models.
    Some options only work with radiopropa installed.

02_path_info.py :
    Using a simple pulse model, this example provides the opportunity to see all the relevant birefringent properties during propagation. 
    This includes the propagation path the pulse form with and without birefringence the refractive indices, 
    the effective refractive indices as well as the polarization eigenvectors.

03_ARA_SPice.py :
    Using pulses measured for the SPice drop, this example propagates pulses from different emitter depths and compares their amplitude to measured data from ARA.

04_ARIANNA_SPice.py :
    Using pulses measured for the SPice drop, this example propagates pulses from different emitter depths and compares their polarization to measured data from ARIANNA.

05_RNOG_DISC.py:
    Calculate the approximate time delay at RNOG stations compared to the DISC hole for different ice models and directions of the ice flow.

06_Veff_comparison.py:
    This example runs a full NuRadioMC simulation and calculates the effective volume for a non-birefringent and a birefringent medium.

07_SPIice_simulation_ARIANNA.py:
    This example runs a full NuRadioMC simulation for the ARIANNA-51 detector geometry using the implementation in emitter.py.

08_SPIice_simulation_ARA.py:
    This example runs a full NuRadioMC simulation for the ARA-A5 detector geometry using the implementation in emitter.py.


extra_files:

    SPice_pulses.npy (Used in example 03, 04, 05, 06, 07):    
            Data measured for the SPice emitter in an annecoic chamber (corrected for a refractive index of 1.78).
            The data was taken from https://github.com/ggaswint/ARIANNAanalysis/tree/master/data/AnechoicChamberData/EFieldData
            The measurements were made for various launch angles (careful, different definition as in NuRadioMC).
            There were 10 measurements (iN) of the electric field per launch angles (launch_angle).
            All downsampled pulses are saved in extra_files/SPice_pulses.xz in a directory:
                Sampling rate:  directory['sampling_rate']
                Pulses:         directory['efields'][launch_angle][iN] -> returns a 2d-array e_field
                                e_field[0] -> theta component of the pulse
                                e_field[1] -> phi component of the pulse

    example_pulse.npy (Used in example 01, 02):    
            Simple neutrino pulse generated with NuRadioMC to see the effect of different ray propagation modules. 

    ARA_data.npy (Used in example 03):
            Amplitude from the ARA collaboration for a SPice drop. Published here: https://iopscience.iop.org/article/10.1088/1475-7516/2020/12/009/pdf

    ARIANNA_data.npy & ARIANNA_systematics.npy (Used in example 04):
            Amplitude from the ARIANNA collaboration for a SPice drop. Published here: https://iopscience.iop.org/article/10.1088/1748-0221/15/09/P09039/pdf

