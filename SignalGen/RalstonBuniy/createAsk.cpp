#include "Askaryan.h"
#include <cstdlib>
#include <fstream>
#include <ctime>
#include <iostream>
#include <complex>
#include <cmath>


using namespace std;

std::pair<std::vector<float>, std::vector<std::complex<float>>> createAsk(const double emEnergy, const double theta) {
	vector<float> *freqs = new vector<float>;
	float df = 0.75;
	for(float f1 = 1.0; f1<10.0; f1=f1+df) freqs->push_back(f1*1e-3);
	for(float f1 = 1.0; f1<10.0; f1=f1+df) freqs->push_back(f1*1e-2);
	for(float f1 = 1.0; f1<10.0; f1=f1+df) freqs->push_back(f1*1e-1);
	for(float f1 = 1.0; f1<10.0; f1=f1+df) freqs->push_back(f1*1e+0);
	for(float f1 = 1.0; f1<10.0; f1=f1+df) freqs->push_back(f1*1e+1);
	Askaryan *h = new Askaryan();
	h->setFormScale(1/(sqrt(2.0*3.14159)*0.03));
	h->setAskFreq(freqs);
	h->emShower(emEnergy);
	h->setAskDepthA(1.5);
	h->setAskR(1000.0);
	h->setAskTheta(theta*PI/180.0);
	vector<vector<cf> > *Eshow = new vector<vector<cf> >;
	Eshow = h->E_omega();
	vector<cf> eTheta = Eshow->at(1);
	delete Eshow;
	delete h;
	std::pair<std::vector<float>, std::vector<std::complex<float>>> result;
	result.first = *freqs;
	result.second = eTheta;
	delete freqs;
	return result;
}

int main(int argc, char **argv) {
	const double emEnergy = atof(argv[1]);
	const float theta = atof(argv[2]);
	std::pair<std::vector<float>, std::vector<std::complex<float>>> result;
	result = createAsk(emEnergy, theta);
	char title[100];
	sprintf(title, "shower_%s.dat", argv[3]);
	ofstream out(title);
	for (int j = 0; j < result.second.size(); ++j) {
		out << result.first[j] << " " << abs(result.second[j]) << std::endl;
	}
	out.close();
	return 0;
}
