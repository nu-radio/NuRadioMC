set -e
cd NuRadioMC/test/SignalProp/
python3 T01test_python_vs_cpp.py
python3 T02test_analytic_D_T.py
python3 T04MooresBay.py
python3 T05unit_test_C0_SP.py
python3 T06unit_test_C0_mooresbay.py

cd ../../SignalProp/examples
python3 example_3d.py
python3 A01IceCubePulserToARA.py
python3 ray_tracing_modules.py
