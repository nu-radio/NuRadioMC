#!/bin/bash
cd "$(dirname "$0")"
cd CPPAnalyticRayTracing

# Remove old wrapper files, do not report errors if they do not exist
rm -v wrapper.so 2> /dev/null
rm -v wrapper.*.so 2> /dev/null

if [[ -z "${GSLDIR}" ]]; then
	echo "GSLDIR environment variable undefined"
	echo "setting GSLDIR to " $(gsl-config --prefix)
	export GSLDIR=$(gsl-config --prefix)
fi
python setup.py build_ext --inplace