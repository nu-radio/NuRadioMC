#!/bin/bash
# Build the Python wrapper of the C++ ray tracer for one Python interpreter: $PYTHON, else `python` from PATH.
# Each Python version gets its own file (wrapper<EXT_SUFFIX>, e.g. wrapper.cpython-312-x86_64-linux-gnu.so), so builds
# for several Python versions can coexist.
usage="Usage: $0 [--clean-all]  (--clean-all: first remove the builds of all Python versions and the build directory)"
clean_all=false
for arg in "$@"; do
	case "$arg" in
		--clean-all) clean_all=true ;;
		-h|--help) echo "$usage"; exit 0 ;;
		*) echo "$usage" >&2; exit 2 ;;
	esac
done

# stop if this fails: the rm calls below must run only in CPPAnalyticRayTracing
cd "$(dirname "$0")/CPPAnalyticRayTracing" || exit 1

rm -f wrapper.so  # legacy untagged build: any Python version would import it if its own build is missing
if $clean_all; then
	rm -f wrapper.*.so
	rm -rf build
fi

# try to set $GSLDIR if it isn't already defined
if [[ -z "${GSLDIR}" ]]; then
	echo "GSLDIR environment variable undefined"
	echo "setting GSLDIR to " $(gsl-config --prefix)
	export GSLDIR=$(gsl-config --prefix)
fi
# $PYTHON is set when called from python (sys.executable). --force: rebuild even if the existing file looks up to date
# (it is called when importing that file failed)
"${PYTHON:-python}" setup.py build_ext --inplace --force
