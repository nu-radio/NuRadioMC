#ifndef ASKARYAN_H_
#define ASKARYAN_H_
//Askaryan class
//Author: Jordan C. Hanson
//June 19th, 2018
//Adapted from Ralston and Buniy (2001)

//Variables defined for one interaction, (one angle and distance), 
//but continuous frequency.

#include <vector>
#include <cmath>
#include <complex>

typedef std::complex<float> cf;

class Askaryan {
	protected:
		int _isEM; //Electromagnetic parameterizations
		int _isHAD; //Hadronic parameterizations
		float _rho0; //Form factor parameter, with units 1/m
		//Use the _rho0 parameter above, in a single exponential model from the complex analysis paper (2017)
		bool _useFormFactor;
		float _askaryanDepthA; //meters
		float _askaryanR; //meters
		float _Nmax; //excess electrons over positrons, per 1000, at shower max
		float _askaryanTheta; //radians
		//Require that even under the LPM elongation, the low-frequency radiation is the same as without LPM
		//Similar to a strict total track length requirement
		bool _strictLowFreqLimit;
		std::string FFTW_CHOICE;
		float PI;
		float LIGHT_SPEED;
		float INDEX;
		float ICE_DENSITY;
		float ICE_RAD_LENGTH;
		float STANDARD_ASK_DEPTH;
		float STANDARD_ASK_R;
		float STANDARD_ASK_NMAX;
		float NORM;
		float RADDEG;
		float COS_THETA_C;
		float _E; //energy in GeV
		std::vector<float>* _askaryanFreq; //GHz
	public:
        Askaryan(): _isEM(0), //EM shower, use emShower()
					_isHAD(0), //HAD shower, use hadShower()
					_rho0(10.0),
					_useFormFactor(true),
					_askaryanDepthA(STANDARD_ASK_DEPTH),
					_askaryanR(STANDARD_ASK_R),
					_Nmax(STANDARD_ASK_NMAX),
					_askaryanTheta(55.82*PI/180.0),
					_strictLowFreqLimit(false),
					FFTW_CHOICE("FFTW_BACKWARD"),
					PI(3.14159),
					LIGHT_SPEED(0.29972),
					INDEX(1.78),
					ICE_DENSITY(0.9167),
					ICE_RAD_LENGTH(36.08),
					STANDARD_ASK_DEPTH(5.0),
					STANDARD_ASK_R(1000.0),
					STANDARD_ASK_NMAX(1000),
					NORM(1.0),
					RADDEG(0.01745),
					COS_THETA_C(0.561798),
					_E(0.0),
					_askaryanFreq(0){};
		void toggleFormFactor(); //What it sounds like: use or don't use form factor.
		void toggleLowFreqLimit(); //What it sounds like: turn on strictLowFreqLimit.
		void setAskTheta(float); //radians
		void setAskFreq(std::vector<float>*); //GHz
		void setAskR(float); //m
		void setAskDepthA(float); //m
		void setNmax(float); //per 1000
		void setAskE(float); //GeV
		float criticalF(); //GHz
		float getAskE(); //GeV
		float getAskR(); //meters
		float getAskDepthA(); //m
		float getAskNmax(); //pure number
		float getAskEta(float); //pure number
		void emShower(float); //Shower parameters from energy in GeV
		void hadShower(float); //Shower parameters from energy in GeV
		void setFormScale(float); //Set shape of shower (meters^{-1}).
		std::vector<float> k(); //1/meters
		std::vector<float> eta(); //unitless
		std::vector<cf> I_ff(); //m
		std::vector<std::vector<cf> > E_omega(); //V/m/MHz
		std::vector<std::vector<float> > E_t(); //V/m
		std::vector<float> time(); //ns
		void lpmEffect();
		void setIndex(float);
};
#endif
