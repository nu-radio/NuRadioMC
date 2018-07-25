#include "createAsk.h"
#include "Askaryan.h"
#include <cstdlib>
#include <fstream>
#include <ctime>
#include <iostream>
#include <complex>
#include <cmath>

#include <units.h>

using namespace std;

void getFrequencySpectrum2(double*& spectrumRealR, double*& spectrumImagR,
		double*& spectrumRealTheta, double*& spectrumImagTheta,
		double*& spectrumRealPhi, double*& spectrumImagPhi, int& size,
		const double energy, const double theta, double* freqs, int size_f,
		const bool isEMShower, const double n, const double R, const bool LPM,
		const double a) {
	// we transform the frequency array to the base units of the Askaryan module which is GHz
	std::vector<float> freqs2;
	for (int i = 0; i < size_f; ++i) {
		freqs2.push_back(freqs[i] / utl::GHz); // Askaryan module uses GHz internally
	}

	Askaryan *h = new Askaryan();
	h->setFormScale(1 / (sqrt(2.0 * 3.14159) * 0.03));
	h->setAskFreq(&freqs2);
	if (isEMShower) {
		h->emShower(energy / utl::GeV); // Askaryan module uses GeV internally
		if (LPM) {
			h->lpmEffect();
		}
	} else {
		h->hadShower(energy / utl::GeV); // Askaryan module uses GeV internally
	}
	h->setAskR(R);
	h->setIndex(n);
	h->setAskTheta(theta);
	if (a > 0) {
		h->setAskDepthA(a / utl::m);  // Askaryan module uses m internally
	}
	vector<vector<cf> > *Eshow = new vector<vector<cf> >;
	Eshow = h->E_omega();
	vector<cf> eR = Eshow->at(0);
	vector<cf> eTheta = Eshow->at(1);
	vector<cf> ePhi = Eshow->at(2);
	delete Eshow;
	delete h;

	size = eTheta.size();
	spectrumRealR = new double[size];
	spectrumImagR = new double[size];
	spectrumRealTheta = new double[size];
	spectrumImagTheta = new double[size];
	spectrumRealPhi = new double[size];
	spectrumImagPhi = new double[size];
	for (int j = 0; j < size; ++j) {
		spectrumRealR[j] = eR.at(j).real() * utl::V / utl::m / utl::MHz;
		spectrumImagR[j] = eR.at(j).imag() * utl::V / utl::m / utl::MHz;
		spectrumRealTheta[j] = eTheta.at(j).real() * utl::V / utl::m / utl::MHz;
		spectrumImagTheta[j] = eTheta.at(j).imag() * utl::V / utl::m / utl::MHz;
		spectrumRealPhi[j] = ePhi.at(j).real() * utl::V / utl::m / utl::MHz;
		spectrumImagPhi[j] = ePhi.at(j).imag() * utl::V / utl::m / utl::MHz;
	}
}
