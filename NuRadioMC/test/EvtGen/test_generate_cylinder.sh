set -e
FILE=NuRadioMC/test/EvtGen/test.hdf5
if test -f "$FILE"; then
	rm $FILE
fi
python NuRadioMC/EvtGen/generate_cylinder.py $FILE 1000 1e18 1e18 0 3000 -2700 0