set -e
cd NuRadioMC/test/SignalProp/
python T01test_python_vs_cpp.py
python T02test_analytic_D_T.py
python T04MooresBay.py
python T05unit_test_C0_SP.py
python T06unit_test_C0_mooresbay.py
