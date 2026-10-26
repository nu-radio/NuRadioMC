#!/bin/bash
cd "$(dirname "$0")"
cd CPPAnalyticRayTracing

# check if old versions exist, and delete them
if [ -f wrapper.so ]; then
	rm wrapper.so  # first delete already existing module
fi
oldfiles=$(find . -maxdepth 1 -name "wrapper.*.so")
if [ -n "$oldfiles" ]; then
	rm wrapper.*.so #also the versions on mac
fi

# try to set $GSLDIR if it isn't already defined
if [[ -z "${GSLDIR}" ]]; then
	echo "GSLDIR environment variable undefined"
	echo "setting GSLDIR to " $(gsl-config --prefix)
	export GSLDIR=$(gsl-config --prefix)
fi
python setup.py build_ext --inplace