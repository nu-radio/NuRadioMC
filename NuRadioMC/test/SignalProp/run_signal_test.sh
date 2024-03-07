set -e
cd NuRadioMC/test/SignalProp/
python3 T01test_python_vs_cpp.py
python3 T02test_analytic_D_T.py
python3 T04MooresBay.py
python3 T05unit_test_C0_SP.py
python3 T06unit_test_C0_mooresbay.py
python3 T07test_birefringence.py

cd emitter_sim_test
python3 T01_sim_events.py
python3 T02_run_simulation.py

cd ..
python3 T08test_emitter_birefringence.py
rm emitter_sim_test/test_input.hdf5
rm emitter_sim_test/test_output.hdf5
rm emitter_sim_test/test_output.nur

cd ../../../SignalProp/examples
python3 example_3d.py
python3 A01IceCubePulserToARA.py
cd birefringence_examples

python3 01_simple_propagation.py
rm 01_simple_propagation_plot.png

python3 02_path_info.py  
rm 02_path_info_plot.png  

python3 03_ARA_SPice.py
rm 03_ARA_simple_plot.png

python3 04_ARIANNA_SPice.py
rm 04_ARIANNA_simple_plot.png

python3 05_RNOG_DISC.py
rm DISC_greenland.png

cd 06_Veff_comparison  
python3 06_A_generate_event_list.py
python3 06_B_Veff_comparison.py
rm 1e19_n1e3.hdf5
rm 06_Veff_comp_plot.png
rm Veff_19_birefringence.hdf5
rm Veff_19_no_birefringence.hdf5
