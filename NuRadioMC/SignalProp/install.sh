#!/bin/bash
cd "$(dirname "$0")"
cd CPPAnalyticRayTracing
rm wrapper.so  # first delete already existing module
if [[ -z "${GSLDIR}" ]]; then
	echo "GSLDIR environment variable undefined" 
	echo "setting GSLDIR to " $(gsl-config --prefix)
	export GSLDIR=$(gsl-config --prefix)
fi
python setup.py build_ext --inplace