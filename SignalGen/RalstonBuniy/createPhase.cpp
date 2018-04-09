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
    float df = 1e-3;
    for(float f=df;f<10.0;f=f+df) freqs->push_back(f);
    Askaryan *h = new Askaryan();
    h->setAskFreq(freqs);
    h->emShower(atof(argv[1]));
    h->lpmEffect();
    h->setFormScale(1.0/(sqrt(2.0*3.14159)*0.1));
    float theta = atof(argv[2]);
    sprintf(title,"shower_%s_F.dat",argv[3]);
    ofstream out(title);
    h->setAskTheta(theta*PI/180.0);
    vector<vector<float> > *Eshow = new vector<vector<float> >;
    Eshow = h->E_t();
    vector<float> *t = h->time();
    vector<float> eTheta = Eshow->at(1);
    delete Eshow;
    for(int j=0;j<eTheta.size();++j) out<<t->at(j)<<" "<<eTheta[j]<<endl;
    out.close();
    delete h;
    delete freqs;
    return 0;
}
