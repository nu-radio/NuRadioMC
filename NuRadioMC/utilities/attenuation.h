#include <math.h>
#include <units.h>

using namespace std;

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
		}
		else{
			a = (b2 * w1 - b1 * w2) / (w1 - w2);
			bb = (b2 - b1) / (w2 - w1);
		}
		return 1./exp(a +bb*w);
	} else {
		std::cout << "attenuation length model " << model << " unknown" << std::endl;
		throw 0;
	}
}
