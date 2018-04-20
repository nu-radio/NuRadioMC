#include "Askaryan.h"
#include <cstdlib>
#include <fstream>
#include <ctime>
#include <iostream>
#include <complex>
#include <cmath>
using namespace std;

int main(int argc, char **argv){
    char title[100];
    vector<float> *freqs = new vector<float>;
    float df = 0.75;
	for(float f1 = 1.0; f1<10.0; f1=f1+df) freqs->push_back(f1*1e-3);
	for(float f1 = 1.0; f1<10.0; f1=f1+df) freqs->push_back(f1*1e-2);
	for(float f1 = 1.0; f1<10.0; f1=f1+df) freqs->push_back(f1*1e-1);
	for(float f1 = 1.0; f1<10.0; f1=f1+df) freqs->push_back(f1*1e+0);
	for(float f1 = 1.0; f1<10.0; f1=f1+df) freqs->push_back(f1*1e+1);
    Askaryan *h = new Askaryan();
    h->setFormScale(1/(sqrt(2.0*3.14159)*atof(argv[4])));
    h->setAskFreq(freqs);
    h->setAskR(1200.0);
    h->emShower(atof(argv[1]));
    //~ h->toggleLowFreqLimit();
	h->lpmEffect();
    float theta = atof(argv[2]);
    sprintf(title,"shower_%s.dat",argv[3]);
    ofstream out(title);
    h->setAskTheta(theta*PI/180.0);
    vector<vector<cf> > *Eshow = new vector<vector<cf> >;
    Eshow = h->E_omega();
    vector<cf> eTheta = Eshow->at(1);
    delete Eshow;
    //Scale the field by R/E_C (m/TeV)
    for(int j=0;j<eTheta.size();++j) out<<freqs->at(j)<<" "<<h->getAskR()*abs(eTheta[j])/(h->getAskE()/1.0e3)<<endl;
    out.close();
    delete h;
    delete freqs;
    return 0;
}
