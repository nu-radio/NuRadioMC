# AnalyticRayTracing
C++ code for analytic ray tracing

## Prerequisites
You will need to have a functioning installation of [GSL](https://www.gnu.org/software/gsl/) ([1.16](https://ftp.gnu.org/gnu/gsl/gsl-1.16.tar.gz) is verified to work).
- You will need to set the enviromnent variable `GSLDIR` to your local installation of GSL.
- You will also need to have `GSLDIR` in your `LD_LIBRARY_PATH`.
- For Mac users: you can locate your GSL installation via `gsl-config --prefix`, i.e. `export GSLDIR=$(gsl-config --prefix)`

## Install instructions

### As part of NuRadioMC
To create python wrapper around the relevant C function, so that the C code can be used directly from the ray tracer
class of NuRadioMC just execute
`python setup.py build_ext --inplace` 

### As standalone package
Getting going is easy. Just:
- Make it: `make analytic_raytracing`
- Run it: `./analytic_raytracing`
- The main is at the bottom of the code, which you can modify to your liking.
  - You can find if, and how many, solutions exist to the ray tracing problem.
  - Find the ray tracing path, path length, time of flight, and attenuation factor of the ray.
  - Find the launch and receive angles for the ray.

