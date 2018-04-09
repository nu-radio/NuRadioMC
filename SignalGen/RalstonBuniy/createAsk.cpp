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

std::vector<std::complex<float>> getFrequencySpectrum(const double energy,
		const double theta, std::vector<float> &freqs, const bool isEMShower) {

	// we transform the frequency array to the base units of the Askaryan module which is GHz
	for (int i = 0; i < freqs.size(); ++i) {
		freqs[i] /= utl::GHz;
	}

	Askaryan *h = new Askaryan();
	h->setFormScale(1 / (sqrt(2.0 * 3.14159) * 0.03));
	h->setAskFreq(&freqs);
	if (isEMShower)
		h->emShower(energy / utl::GeV); // Askaryan module uses GHz internally
	else
		h->hadShower(energy / utl::GeV);
	h->setAskDepthA(1.5);
	h->setAskR(1000.0);
	h->setAskTheta(theta);
	vector<vector<cf> > *Eshow = new vector<vector<cf> >;
	Eshow = h->E_omega();
	vector<cf> eTheta = Eshow->at(1);
	delete Eshow;
	delete h;
	std::vector<std::complex<float>> result;
	result = eTheta;
	// convert to default units
	for (int j = 0; j < result.size(); ++j) {
		result[j] *= utl::V / utl::m / utl::MHz;
	}
	return result;
}

std::pair<std::vector<float>, std::vector<std::vector<float> > > getTimeTrace(
		const double energy, const double theta, const double fmin,
		const double fmax, const double df, const bool isEMShower) {

	Askaryan *h = new Askaryan();
	vector<vector<float> > *Eshow = new vector<vector<float> >;
	vector<float> *freqs = new vector<float>;
	vector<float> e;
	vector<float> *t = new vector<float>;
	float R = 0.0;

	for (float i = fmin; i < fmax; i += df) {
		freqs->push_back(i / utl::GHz);
	}
	h->setAskFreq(freqs);
	h->setAskTheta(theta);
	if (isEMShower)
		h->emShower(energy / utl::GeV);
	else
		h->hadShower(energy / utl::GeV);
	h->setFormScale(7.8);
	h->lpmEffect();
	Eshow = h->E_t();
	t = h->time();
	R = h->getAskR();

	std::pair<std::vector<float>, std::vector<std::vector<float> > > result;
	result.first = *t;
	result.second = *Eshow;
	// convert to default units and normalize to signal at 1m away from shower
	for (int j = 0; j < result.first.size(); ++j) {
		result.first[j] *= utl::ns;
		result.second[0][j] *= utl::V / utl::m * R;
		result.second[1][j] *= utl::V / utl::m * R;
		result.second[2][j] *= utl::V / utl::m * R;
	}

	return result;
}

int main(int argc, char **argv) {
	const double emEnergy = atof(argv[1]) * utl::eV;
	const float theta = atof(argv[2]) * utl::degree;
	std::pair<std::vector<float>, std::vector<std::complex<float>>>result;

	vector<float> freqs;
	float df = 0.75;
	for (float f1 = 1.0; f1 < 10.0; f1 = f1 + df)
		freqs.push_back(f1 * 1e-3);
	for (float f1 = 1.0; f1 < 10.0; f1 = f1 + df)
		freqs.push_back(f1 * 1e-2);
	for (float f1 = 1.0; f1 < 10.0; f1 = f1 + df)
		freqs.push_back(f1 * 1e-1);
	for (float f1 = 1.0; f1 < 10.0; f1 = f1 + df)
		freqs.push_back(f1 * 1e+0);
	for (float f1 = 1.0; f1 < 10.0; f1 = f1 + df)
		freqs.push_back(f1 * 1e+1);

	result = getFrequencySpectrum(emEnergy, theta, freqs);
	char title[100];
	sprintf(title, "shower_%s.dat", argv[3]);
	ofstream out(title);
	for (int j = 0; j < result.second.size(); ++j) {
		out << result.first[j] << " " << abs(result.second[j]) << " "
				<< real(result.second[j]) << " " << imag(result.second[j])
				<< std::endl;
	}
	out.close();

	// get time trace
	std::pair<std::vector<float>, std::vector<std::vector<float> > > timeTrace;
	timeTrace = getTimeTrace(1 * utl::EeV, theta, 0 * utl::MHz, 5 * utl::GHz,
			10 * utl::MHz, true);
	char title2[100];
	sprintf(title2, "shower_time_%s.dat", argv[3]);
	ofstream out2(title2);
	for (int j = 0; j < timeTrace.first.size(); ++j) {
		out2 << timeTrace.first[j] << " " << timeTrace.second[0][j] << " "
				<< timeTrace.second[1][j] << " " << timeTrace.second[2][j]
				<< std::endl;
	}
	out2.close();
	return 0;
}
