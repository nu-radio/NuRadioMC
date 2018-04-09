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
    h->setFormScale(1/(sqrt(2.0*3.14159)*0.03));
    h->setAskFreq(freqs);
    h->emShower(atof(argv[1]));
    h->setAskDepthA(1.5);
    h->setAskR(1000.0);
    float theta = atof(argv[2]);
    sprintf(title,"shower_%s.dat",argv[3]);
    ofstream out(title);
    h->setAskTheta(theta*PI/180.0);
    vector<vector<cf> > *Eshow = new vector<vector<cf> >;
    Eshow = h->E_omega();
    vector<cf> eTheta = Eshow->at(1);
    delete Eshow;
    for(int j=0;j<eTheta.size();++j) out<<freqs->at(j)<<" "<<abs(eTheta[j])<<endl;
    out.close();
    delete h;
    delete freqs;
    return 0;
}
