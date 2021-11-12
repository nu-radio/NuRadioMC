This directory contains all necessary files to do a cosmic ray analysis for a specific detector,
including trigger settings and air shower reconstruction.

1. 'create_config_file.py'
Create a configuration file with all relevant parameters.

2. 'calculate_trigger_threshold_for_trigger_rate.py'
    or
    'calculate_trigger_rate_for_thresholds.py' and 'merge_output theshold_calulcations.py'
In this step, the threshold for a specific antenna set is calculated. One can choose between a brut force
script which increases the threshold incrementally until the target trigger rate is obtained or a
smarter and faster script which estimates the threshold range and calculates the trigger rate for the given
threshold range. The advantage is, that the hugh number of iterations can be divided into several job and
will be merged with the script 'merge_output theshold_calulcations.py' Even if you only do one job with
'calculate_trigger_rate_for_thresholds.py ' you have to merge the one file afterwards!

3. 'plot_trigger_rate_vs_threshold.py'
This script plots the slope of the trigger rate for different thresholds obtained
by calculate_trigger_threshold_for_trigger_rate.py
or 'calculate_trigger_rate_for_thresholds.py' and 'merge_output theshold_calulcations.py'

4. 'air_shower_reconstruction.py'
This script evaluates the obtained trigger settings on air shower simulations done with corsika and
stores the results in nur files.

5. 'anaylze_air_shower_reco.py'
To analyze the performancec of the detector, the results of the air shower reco have to be divided into
different energy, zenith and distance bins. this is done by 'anaylze_air_shower_reco.py'

