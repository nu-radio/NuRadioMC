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
    float df = 0.001;
    float fc = 10.0;
    for(float x=df;x<fc;x=x+df) freqs->push_back(x);
    Askaryan *h = new Askaryan();
    h->setFormScale(1.0/(sqrt(2.0*3.14159)*0.05));
    h->setAskFreq(freqs);
    h->emShower(atof(argv[1]));
    h->lpmEffect();
    vector<float> *t = new vector<float>;
    t = h->time();
    sprintf(title,"shower_%s.dat",argv[2]);
    ofstream out(title);
    //for(float theta=45.8;theta<65.8;theta=theta+0.01)
    //{
    h->setAskTheta(57.0*PI/180.0);
    vector<vector<float> > *Eshow = new vector<vector<float> >;
    Eshow = h->E_t();
    vector<float> eTheta = Eshow->at(1);
    delete Eshow;
    for(int j=0;j<eTheta.size();++j) out<<eTheta[j]<<" "<<endl;
    //for(int j=0;j<eTheta.size();++j) out<<eTheta[j]<<" ";
    //out<<endl;
	//}
    out.close();
    delete h;
    delete freqs;
    delete t;
	return 0;
}
