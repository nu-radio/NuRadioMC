set -e
FILE=NuRadioMC/test/EvtGen/test_proposal.hdf5
if test -f "$FILE"; then
	rm $FILE
fi
python NuRadioMC/EvtGen/generate_cylinder.py $FILE 100 1e18 1e18 0 3000 -2700 0 --proposal --proposal_config NuRadioMC/test/EvtGen/config_PROPOSAL_greenland.json