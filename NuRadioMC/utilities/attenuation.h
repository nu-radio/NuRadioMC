#include <math.h>
#include <units.h>

using namespace std;

double fit_GL1(double z, double frequency){
	// Model for Greenland. Taken from DOI: https://doi.org/10.3189/2015JoG15J057
	// Returns the attenuation length at 75 MHz as a function of depth
	double fit_values[] = {1.16052586e+03, 6.87257150e-02, -9.82378264e-05,
									-3.50628312e-07, -2.21040482e-10, -3.63912864e-14};

	double att_length = 0;
	for (int power = 0; power < 6; power++){
		att_length += fit_values[power] * pow(z, power);
	}

	double att_length_f = att_length - 0.55*utl::m * (frequency/utl::MHz - 75);

	const double min_length = 100 * utl::m;
	if ( att_length_f < min_length ){ att_length_f = min_length; }

	return att_length_f;
}

double get_temperature(double z){
	//return temperature as a function of depth
	// from https://icecube.wisc.edu/~araproject/radio/#icetemperature
	double z2 = abs(z/utl::m);
	return 1.83415e-09*z2*z2*z2 + (-1.59061e-08*z2*z2) + 0.00267687*z2 + (-51.0696 );
}

double get_attenuation_length(double z, double frequency, int model){
	if(model == 1) {
		double t = get_temperature(z);
		double f0 = 0.0001;
		double f2 = 3.16;
		double w0 = log(f0);
		double w1 = 0.0;
		double w2 = log(f2);
		double w = log(frequency / utl::GHz);
		double b0 = -6.74890 + t * (0.026709 - t * 0.000884);
		double b1 = -6.22121 - t * (0.070927 + t * 0.001773);
		double b2 = -4.09468 - t * (0.002213 + t * 0.000332);
		double a, bb;
		if(frequency<1. * utl::GHz){
			a = (b1 * w0 - b0 * w1) / (w0 - w1);
			bb = (b1 - b0) / (w1 - w0);
		} else{
			a = (b2 * w1 - b1 * w2) / (w1 - w2);
			bb = (b2 - b1) / (w2 - w1);
		}
		return 1./exp(a +bb*w);
	} else if (model == 2) {

		return fit_GL1(z, frequency);

	} else if (model == 3) {
		double R = 0.82;
		double d_ice = 576 * utl::m;
		double att_length = 460 * utl::m - 180 * utl::m /utl::GHz * frequency;
		att_length *= 1./(1 + att_length / (2 * d_ice) * log(R));  // additional correction for reflection coefficient being less than 1.

		double d = -z * 420. * utl::m / d_ice;
        double L = (1250.*0.08886 * exp(-0.048827 * (225.6746 - 86.517596 * log10(848.870 - (d)))));
        // this differs from the equation published in F. Wu PhD thesis UCI.
        // 262m is supposed to be the depth averaged attenuation length but the
        // integral (int(1/L, 420, 0)/420) ^ -1 = 231.21m and NOT 262m.
        att_length *= L / (231.21 * utl::m);
		return att_length;
	} else {
		std::cout << "attenuation length model " << model << " unknown" << std::endl;
		throw 0;
	}
}
