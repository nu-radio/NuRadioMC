#include "Askaryan.h"
#include <cstdlib>
#include <fstream>
#include <ctime>
#include <iostream>
#include <complex>
using namespace std;

int main(int argc, char **argv){
    char title[100];
    vector<float> *freqs = new vector<float>;
    Askaryan *h = new Askaryan();
    h->setAskR(atof(argv[6]));
    sprintf(title,"shower_%s.dat",argv[2]);
    freqs->push_back(atof(argv[3]));
    h->setAskFreq(freqs);
    if(atof(argv[5])==1.0)
    {
		h->emShower(atof(argv[1]));
        h->toggleLowFreqLimit();
        h->lpmEffect();
        h->setFormScale(1.0/(sqrt(2.0*3.14159)*atof(argv[4])));
        std::cout<<"EM: Energy: "<<atof(argv[1])<<" Frequency: "<<atof(argv[3])<<" Distance: "<<atof(argv[6])<<" Eta: "<<h->getAskEta(atof(argv[3]))<<std::endl;
	}
	else
	{
		h->hadShower(atof(argv[1]));
        h->toggleLowFreqLimit();
        h->lpmEffect();
        h->setFormScale(1.0/(sqrt(2.0*3.14159)*atof(argv[4])));
        std::cout<<"Had: Energy: "<<atof(argv[1])<<" Frequency: "<<atof(argv[3])<<" Distance: "<<atof(argv[6])<<" Eta: "<<h->getAskEta(atof(argv[3]))<<std::endl;
	}
    ofstream out(title);
    float dtheta = 0.1;
    for(float theta=30.8;theta<80.8;theta=theta+dtheta)
    {
		h->setAskTheta(theta*PI/180.0);
		vector<vector<cf> > *Eshow = new vector<vector<cf> >;
		Eshow = h->E_omega();
		vector<cf> eTheta = Eshow->at(1);
		delete Eshow;
		for(int j=0;j<eTheta.size();++j) out<<theta<<" "<<(h->getAskR())*abs(eTheta[j])/(h->getAskE()/1000.0)<<endl;
	}
    out.close();
    delete h;
    delete freqs;
    return 0;
}
