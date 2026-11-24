# AnalyticRayTracing
C++ code for analytic ray tracing.

## Use in NuRadioMC
NuRadioMC compiles the Python wrapper of this code itself. Prerequisites (GSL) and how to build it by hand:
[GSL speed-boost](https://nu-radio.github.io/NuRadioMC/NuRadioMC/pages/Manuals/signal_propagation.html#gsl-speed-boost)
(source: `documentation/source/NuRadioMC/pages/Manuals/signal_propagation.rst`).

## As standalone package
You will need a functioning installation of [GSL](https://www.gnu.org/software/gsl/):
- Set the environment variable `GSLDIR` to your local installation of GSL, e.g. `export GSLDIR=$(gsl-config --prefix)`.
- On Linux, if GSL is not installed in a system library directory, add `$GSLDIR/lib` to your `LD_LIBRARY_PATH`.

Getting going is easy. Just:
- Make it: `make analytic_raytracing`
- Run it: `./analytic_raytracing`
- The main is at the bottom of the code, which you can modify to your liking.
  - You can find if, and how many, solutions exist to the ray tracing problem.
  - Find the ray tracing path, path length, time of flight, and attenuation factor of the ray.
  - Find the launch and receive angles for the ray.
