//Askaryan.cpp

#include "Askaryan.h"
#include <fftw3.h>
#include <algorithm>
#include <numeric>
#include <iostream>

std::vector<float> Askaryan::k()
{
    std::vector<float> result;
	std::vector<float>::iterator j=_askaryanFreq->begin();
    for(j=_askaryanFreq->begin();j!=_askaryanFreq->end();++j)
        result.push_back(2.0*PI*(*j)/(LIGHT_SPEED/INDEX));
	return result;
}

std::vector<float> Askaryan::eta(){
	std::vector<float> result;
	std::vector<float> K = k();
	std::vector<float>::iterator i;
	for(i=K.begin();i!=K.end();++i)
		result.push_back((*i)*(_askaryanDepthA*_askaryanDepthA)/_askaryanR*(sin(_askaryanTheta)*sin(_askaryanTheta)));
	return result;
}

void Askaryan::setAskTheta(float x)
{
	_askaryanTheta = x;
}

void Askaryan::setAskFreq(std::vector<float> *x)
{
	_askaryanFreq = x;
}

void Askaryan::setAskR(float x)
{
	_askaryanR = x;
}

void Askaryan::setAskDepthA(float x)
{
	_askaryanDepthA = x;
}

void Askaryan::setNmax(float x)
{
	_Nmax = x;
}

void Askaryan::setAskE(float x)
{
	_E = x;
}

float Askaryan::getAskE()
{
    return _E;
}

float Askaryan::getAskDepthA()
{
    return _askaryanDepthA;
}

std::vector<cf> Askaryan::I_ff()
{
//Equation 17 from Buniy and Ralston, sans pre-factor to fit into Eq. 19, the general form.
    std::vector<float> K = k();
	std::vector<float> Eta = eta();
	std::vector<cf> result; result.clear();
	std::vector<float>::iterator i; i=K.begin();
	std::vector<float>::iterator j; j=Eta.begin();
	float re_p=0;
	float im_p=0;
	float re_d=0;
	float im_d=0;
	for(i=K.begin(),j=Eta.begin();i!=K.end(),j!=Eta.end();++i,++j)
	{
		re_d = 1-3*((*j)*(*j))*cos(_askaryanTheta)/(sin(_askaryanTheta)*sin(_askaryanTheta))*(cos(_askaryanTheta)-COS_THETA_C)/(1+((*j)*(*j)));
		im_d = -(*j)-3*((*j)*(*j)*(*j))*cos(_askaryanTheta)/(sin(_askaryanTheta)*sin(_askaryanTheta))*(cos(_askaryanTheta)-COS_THETA_C)/(1+((*j)*(*j)));
		cf denom(re_d,im_d);
		re_p = -0.5*((*i)*_askaryanDepthA*(*i)*_askaryanDepthA)*((cos(_askaryanTheta)-COS_THETA_C)*(cos(_askaryanTheta)-COS_THETA_C))/(1+((*j)*(*j)));
		im_p = -(*j)*0.5*((*i)*_askaryanDepthA*(*i)*_askaryanDepthA)*((cos(_askaryanTheta)-COS_THETA_C)*(cos(_askaryanTheta)-COS_THETA_C))/(1+((*j)*(*j)));
		cf power(re_p,im_p);
exp(power);
		result.push_back(exp(power)/sqrt(denom));
	}
	return result;
}

std::vector<std::vector<cf> > Askaryan::E_omega()
{
	//Electric field in the general case, eq. 19 in the paper.
	std::vector<float> K = k();
	std::vector<float> Eta = eta();
	std::vector<cf> I_FF = I_ff();
	std::vector<cf> rComp; rComp.clear();
	std::vector<cf> thetaComp; thetaComp.clear();
	std::vector<cf> phiComp; phiComp.clear();
	std::vector<float>::iterator q=K.begin();
	std::vector<float>::iterator j=Eta.begin();
	std::vector<cf>::iterator i=I_FF.begin();
	for(i=I_FF.begin(),j=Eta.begin(),q=K.begin();q!=K.end();++i,++j,++q)
	{
		//Overall normalization: a(m), nu(GHz), Nmax(1000), nu(GHz)...checked JCH March 8th, 2016
		float nu = LIGHT_SPEED*(*q)/(2.0*PI);
		cf norm(2.52e-7*_askaryanDepthA*_Nmax*nu/_askaryanR/NORM,0.0);
		//Kinematic factor, psi...checked JCH March 8th, 2016...fixed missing sin(theta)
		cf psi(sin(_askaryanTheta)*sin((*q)*_askaryanR),-sin(_askaryanTheta)*cos((*q)*_askaryanR));
		//radial component (imaginary part is zero)...checked JCH March 8th, 2016
		cf rComp_num(-(cos(_askaryanTheta)-COS_THETA_C)/sin(_askaryanTheta),0.0);
		rComp.push_back((*i)*norm*psi*rComp_num);
		//theta component (has real and imaginary parts)...checked JCH March 8th, 2016
		cf thetaComp_num(1+pow((*j),2)/pow((1+(*j)),2)*COS_THETA_C/pow(sin(_askaryanTheta),2)*(cos(_askaryanTheta)-COS_THETA_C),
			-(*j)/pow((1+(*j)),2)*COS_THETA_C/pow(sin(_askaryanTheta),2)*(cos(_askaryanTheta)-COS_THETA_C));
		thetaComp.push_back((*i)*norm*psi*thetaComp_num);
		//phi component (is zero)...checked JCH March 8th, 2016
		cf phiComp_num(0,0);
		phiComp.push_back(phiComp_num);
	}

	if(_useFormFactor)
	{
		std::vector<float>::iterator k;
		std::vector<cf>::iterator Er=rComp.begin();
		std::vector<cf>::iterator Etheta=thetaComp.begin();
		std::vector<cf>::iterator Ephi=phiComp.begin();
		for (k=K.begin();k!=K.end();++k)
		{
			float a = (*k)/_rho0;
			float b = sin(_askaryanTheta)/sqrt(2.0*PI);
			float atten = pow(1+pow(a,2)*pow(b,2),-1.5);
			(*Er)*=atten;
			(*Etheta)*=atten;
			(*Ephi)*=atten;
			++Er;
			++Etheta;
			++Ephi;
		}
	}

	// //Electric field: r, theta, phi
	std::vector<std::vector<cf> > result;
	result.push_back(rComp);
	result.push_back(thetaComp);
	result.push_back(phiComp);
	return result;
}

float Askaryan::criticalF()
{
		return *max_element(_askaryanFreq->begin(),_askaryanFreq->end());
}

std::vector<std::vector<float> > Askaryan::E_t()
{
	std::vector<std::vector<cf> > e = E_omega();
	std::vector<cf> e_r = e.at(0);
	std::vector<cf> e_theta = e.at(1);
	std::vector<cf> e_phi = e.at(2);
	float df = criticalF()/(float(e_r.size()));
	df*=1000.0; //now in MHz.
	int n = e_r.size()*2;
	fftw_complex *in1,*in2,*in3,*out1,*out2,*out3;
	in1 = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*n);
	in2 = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*n);
	in3 = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*n);
	out1 = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*n);
	out2 = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*n);
	out3 = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*n);
	fftw_plan p1=0;
	fftw_plan p2=0;
	fftw_plan p3=0;
	if(FFTW_CHOICE=="FFTW_BACKWARD")
	{
		p1 = fftw_plan_dft_1d(n,in1,out1,1,FFTW_ESTIMATE);
		p2 = fftw_plan_dft_1d(n,in2,out2,1,FFTW_ESTIMATE);
		p3 = fftw_plan_dft_1d(n,in3,out3,1,FFTW_ESTIMATE);
	}
	else
	{
		p1 = fftw_plan_dft_1d(n,in1,out1,-1,FFTW_ESTIMATE);
		p2 = fftw_plan_dft_1d(n,in2,out2,-1,FFTW_ESTIMATE);
		p3 = fftw_plan_dft_1d(n,in3,out3,-1,FFTW_ESTIMATE);
	}
	// Proper assignment to input transforms
	for(int i=0;i<n;++i)
	{
		if(i<n/2){
			in1[i][0] = real(e_r[i]);
			in2[i][0] = real(e_theta[i]);
			in3[i][0] = real(e_phi[i]);
			in1[i][1] = imag(e_r[i]);
			in2[i][1] = imag(e_theta[i]);
			in3[i][1] = imag(e_phi[i]);
		}
		else{
			in1[i][0] = real(e_r[n-i-1]);
			in2[i][0] = real(e_theta[n-i-1]);
			in3[i][0] = real(e_phi[n-i-1]);
			in1[i][1] = -1.0*imag(e_r[n-i-1]);
			in2[i][1] = -1.0*imag(e_theta[n-i-1]);
			in3[i][1] = -1.0*imag(e_phi[n-i-1]);
		}
	}
	fftw_execute(p1);
	fftw_execute(p2);
	fftw_execute(p3);
	std::vector<std::vector<float> > result;
	std::vector<float> Er_t;
	std::vector<float> Etheta_t;
	std::vector<float> Ephi_t;
	for(int i=0;i<n;++i)
	{
		//Output the real part.  It has been verified that the imaginary
		//part is zero, meaning all the power is present in Re{E(t)}.
		//We must multiply the result by df in MHz, because the IFFT is
		//computed discretely, without the frequency measure.  This
		//changes the final units from V/m/MHz to V/m vs. time.
		Er_t.push_back(out1[i][0]*df);
		Etheta_t.push_back(out2[i][0]*df);
		Ephi_t.push_back(out3[i][0]*df);
	}
	//Note: The choice of sign in the Fourier transform convention should not determine physical
	//properties of the output.  The following code ensures the correct physical timing, according
	//to the RB paper, and that the either choice of convention produces the same answer.
	if(FFTW_CHOICE=="FFTW_BACKWARD"){
		std::reverse(Er_t.begin(),Er_t.end());
		std::reverse(Etheta_t.begin(),Etheta_t.end());
		std::reverse(Ephi_t.begin(),Ephi_t.end());
	}
	result.push_back(Er_t);
	result.push_back(Etheta_t);
	result.push_back(Ephi_t);
	fftw_destroy_plan(p1);
	fftw_destroy_plan(p2);
	fftw_destroy_plan(p3);
	fftw_free(in1);
	fftw_free(in2);
	fftw_free(in3);
	fftw_free(out1);
	fftw_free(out2);
	fftw_free(out3);
	fftw_cleanup();
	//Check Parseval's theorem.
	//1. Obtain inner products.
	float sum_sq_r_t = inner_product(Er_t.begin(),Er_t.end(),Er_t.begin(),0.0);
	float sum_sq_theta_t = inner_product(Etheta_t.begin(),Etheta_t.end(),Etheta_t.begin(),0.0);
	float sum_sq_phi_t = inner_product(Ephi_t.begin(),Ephi_t.end(),Ephi_t.begin(),0.0);
	cf sum_sq_r_f = inner_product(e_r.begin(),e_r.end(),e_r.begin(),cf(0,0));
	cf sum_sq_theta_f = inner_product(e_theta.begin(),e_theta.end(),e_theta.begin(),cf(0,0));
	cf sum_sq_phi_f = inner_product(e_phi.begin(),e_phi.end(),e_phi.begin(),cf(0,0));
	std::cout<<(sum_sq_theta_t - sum_sq_theta_f)/sum_sq_theta_t<<std::endl;
	return result;
}

std::vector<float> Askaryan::time()
{
	float fc = criticalF();
	float dt = 1.0/(2.0*fc);
	int n = 2*_askaryanFreq->size();
	std::vector<float> result;
	for(int i=0;i<n;++i) result.push_back(float(i)*dt);
	return result;
}

void Askaryan::emShower(float E=0.0)
{
	this->setAskE(E);
	this->_isEM = 1;
	float E_CRIT = 0.073; //GeV
	float max_x = 50.0; //maximum number of radiation lengths
	float dx = 0.01; //small enough bin in depth for our purposes.
	float x_start = 0.01; //starting radiation length
	//Greissen EM shower profile from Energy E in GeV.
	std::vector<float> *nx = new std::vector<float>();
	nx->clear();
	float a=0.0;
	float b=0.0;
	float c=0.0;
	float d=0.0;
	for(float x=x_start;x<max_x;x+=dx){
        a = 0.31/sqrt(log(E/E_CRIT));
        b = x;
        c = 1.5*x;
        d = log((3*x)/(x+2*log(E/E_CRIT)));
		nx->push_back(a*exp(b-c*d));
	}
	//find location of maximum, and charge excess from Fig. 5.9, compare in cm not m.
	std::vector<float>::iterator n_max=max_element(nx->begin(),nx->end());
	float excess=0.09+dx*(std::distance(nx->begin(),n_max))*ICE_RAD_LENGTH/ICE_DENSITY*1.0e-4;
	this->setNmax(excess*(*n_max)/1000.0);
	//find depth, which is really the FWHM of this Greisen formula.
	std::vector<float>::iterator i=nx->begin();
	for(i=nx->begin();i!=nx->end();++i)
	{
		if((*i)/(*n_max)>0.606531) break;
	}
	std::vector<float>::iterator j=i;
	for(j=i;j!=nx->end();++j)
	{
		if((*j)/(*n_max)<0.606531) break;
	}
	this->setAskDepthA(dx*std::distance(i,j)/ICE_DENSITY*ICE_RAD_LENGTH/100.0); //meters
	delete nx;
}

void Askaryan::hadShower(float E=0.0)
{
	this->setAskE(E);
    this->_isHAD = 1;
    //Gaisser-Hillas hadronic shower parameterization
    std::vector<float> *nx = new std::vector<float>;
    float max_x = 2000.0; //maximum depth in g/cm^2
	float dx = 1.0; //small enough bin in depth for our purposes.
	float x_start = dx; //depth in g/cm^2
    float S0 = 0.11842;
    float X0 = 39.562; //g/cm^2
    float lambda = 113.03; //g/cm^2
    float Ec = 0.17006; //GeV
    float Xmax = X0*log(E/Ec);
    float a=0.0;
    float b=0.0;
    float c=0.0;
	for(float x=x_start;x<max_x;x+=dx){
		a = S0*E/Ec*(Xmax-lambda)/Xmax*exp(Xmax/lambda-1);
		b = pow(x/(Xmax-lambda),Xmax/lambda);
		c = exp(-x/lambda);
		nx->push_back(a*b*c);
	}
    //find location of maximum, and charge excess from Fig. 5.9, compare in cm not m.
    std::vector<float>::iterator n_max = max_element(nx->begin(),nx->end());
    float excess=0.09+dx*(std::distance(nx->begin(),n_max))/ICE_DENSITY*1.0e-4;
	this->setNmax(excess*(*n_max)/1000.0);
	//find depth, which is really the FWHM of this Gaisser-Hillas 
	//formula.  I chose the 1-sigma width to better represent the gaussian.
	std::vector<float>::iterator i=nx->begin();
	for(i=nx->begin();i!=nx->end();++i){
		if((*i)/(*n_max)>0.606531) break;
	}
	std::vector<float>::iterator j=i;
	for(j=i;j!=nx->end();++j){
		if((*j)/(*n_max)<0.606531) break;
	}
	this->setAskDepthA(dx*std::distance(i,j)/ICE_DENSITY/100.0); //meters
	delete nx;
}

void Askaryan::lpmEffect()
{
    //"Accounts" for the "LPM effect," by "stretching" the shower profile, according to
    //Klein and Gerhardt (2010). Polynomial fit to figure 9 for EM and Hadronic.
    float prior_a = this->getAskDepthA();
    
    if(_isEM){
        //EM fit parameters
        float p1 = -2.8564e2;
        float p2 = 7.8140e1;
        float p3 = -8.3893;
        float p4 = 4.4175e-1;
        float p5 = -1.1382e-2;
        float p6 = 1.1493e-4;
        float e = log10(_E)+9.0; //log_10 of Energy in eV
        float log10_shower_depth = p1+p2*pow(e,1)+p3*pow(e,2)+p4*pow(e,3)+p5*pow(e,4)+p6*pow(e,5);
        float a = pow(10.0,log10_shower_depth);
        this->setAskDepthA(a);
        //Right here, record the reduction in n_max that I don't believe in.
        if(_strictLowFreqLimit)
        {
		this->setNmax(_Nmax/(a/prior_a));
	}
    }
    if(_isHAD){
        //HAD fit parameters...should we do this at all?
        //~ float p1 = 8.0583;
        //~ float p2 = -2.1100;
        //~ float p3 = 2.3683e-1;
        //~ float p4 = -1.2649e-2;
        //~ float p5 = 3.3106e-4;
        //~ float p6 = -3.4270e-6;
        //~ float e = log10(_E)+9.0; //log_10 of Energy in eV
        //~ float log10_shower_depth = p1+p2*pow(e,1)+p3*pow(e,2)+p4*pow(e,3)+p5*pow(e,4)+p6*pow(e,5);
        //~ float a = pow(10.0,log10_shower_depth);
        //~ this->setAskDepthA(a);
        //~ //Right here, record the reduction in n_max that I don't believe in.
        //~ if(_strictLowFreqLimit)
        //~ {
		//~ this->setNmax(_Nmax/(a/prior_a));
	//~ }
    }
}

void Askaryan::setFormScale(float d){
    _rho0 = d; //rho0 means rho0, not sqrt(2pi)rho0
}

void Askaryan::toggleFormFactor(){
    _useFormFactor = !_useFormFactor;
}

void Askaryan::toggleLowFreqLimit(){
	_strictLowFreqLimit = !_strictLowFreqLimit;
}

float Askaryan::getAskR(){
    return _askaryanR;
}

float Askaryan::getAskEta(float nu){
    return 2.0*3.14159*nu/0.3/_askaryanR*_askaryanDepthA*_askaryanDepthA*sin(_askaryanTheta)*sin(_askaryanTheta);
}

float Askaryan::getAskNmax(){
    return _Nmax;
}

void Askaryan::setIndex(float n){
	INDEX = n;
	COS_THETA_C = 1.0/n;
}