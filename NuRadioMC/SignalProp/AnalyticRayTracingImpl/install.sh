#!/bin/bash
# Build the Python wrapper of the C++ ray tracer for one Python interpreter: $PYTHON, else `python` from PATH.
#
# Each interpreter gets its own file (wrapper<EXT_SUFFIX>, e.g. wrapper.cpython-312-x86_64-linux-gnu.so). Only the
# running interpreter's file is replaced, so builds for several Python versions can coexist in one checkout.
#
# Usage: install.sh [--clean-all]
#   --clean-all  first remove the wrappers of all interpreters and the build directory
usage="Usage: $0 [--clean-all]"
clean_all=false
for arg in "$@"; do
	case "$arg" in
		--clean-all) clean_all=true ;;
		-h|--help) echo "$usage"; exit 0 ;;
		*) echo "$usage" >&2; exit 2 ;;
	esac
done
python_exe="${PYTHON:-python}"  # $PYTHON is set when called from python (sys.executable)

# stop if this fails: the rm calls below must run only in CPPAnalyticRayTracing
cd "$(dirname "$0")/CPPAnalyticRayTracing" || exit 1

if $clean_all; then
	rm -f wrapper.so wrapper.*.so
	rm -rf build
else
	ext_suffix=$("$python_exe" -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))') || exit 1
	rm -f "wrapper${ext_suffix}"
fi

# try to set $GSLDIR if it isn't already defined
if [[ -z "${GSLDIR}" ]]; then
	echo "GSLDIR environment variable undefined"
	echo "setting GSLDIR to " $(gsl-config --prefix)
	export GSLDIR=$(gsl-config --prefix)
fi
"$python_exe" setup.py build_ext --inplace
