#include <iostream>
#include <fstream>
#include <string>
#include "TGraph.h"
#include "TMultiGraph.h"
#include "TCanvas.h"
#include "TMath.h"
#include "TString.h"
#include "TAxis.h"
#include "TFile.h"
#include "TTree.h"

const Double_t pi=4.0*atan(1.0); /**< Gives back value of Pi */

const Double_t spedc=299792458.0; /**< Speed of Light in m/s */

Double_t h=0.001; /**< RK4 step size is set to 1mm */

const Double_t tmax=6000.0; /**< Total "time" Rk4 will run for. In this case it is total meters before the RK4 loop stops  */

const Double_t A=1.78;/**< Value of Parameter A for SP model of refractive index */
const Double_t B=-0.43; /**< Value of Parameter B for SP model of refractive index */
const Double_t C=0.0132;/**< Value of Parameter A for SP model of refractive index. There will be no negative sign for negative depth. */ 

/*! \brief Takes in values of A,B & C and the depth z and returns value for refrative index model n(z).
 *
 */
Double_t n(Double_t z);

/*! \brief Takes in values of A,B & C and the depth z and returns value for the first derivative of refrative index model.
 *
 */
Double_t dndz(Double_t z);

/*! \brief Takes in value of theta (angle w.r.t to the vertical) and calculates the derivative of x w.r.t to length of the step. 
 * 
 *Basically it helps calculate the value of x in each step in the RK4 calculations.
 */
Double_t fx(Double_t theta);

/*! \brief Takes in value of theta (angle w.r.t to the vertical) and calculates the derivative of depth z w.r.t to length of the step. 
 * 
 *Basically it helps calculate the value of depth z in each step in the RK4 calculations.
 */
Double_t fz(Double_t theta);

/*! \brief Takes in value of theta (angle w.r.t to the vertical), n(z) and dndz(z) and calculates the derivative of theta w.r.t to length of the step. 
 * 
 *Basically it helps calculate the value of theta in each step in the RK4 calculations.
 */
Double_t ftheta(Double_t theta, Double_t z);

/*! \brief Takes in value of depth z and returns the value of temperature. This is used in the RK4 calculations.
 *
 *The temperature model has been taken from AraSim which also took it from here http://icecube.wisc.edu/~mnewcomb/radio/atten/ . This is basically Matt Newcomb's icecube directory which has alot of information, plots and codes about South Pole Ice activities. Please read it if you find it interesting.
 * http://icecube.wisc.edu/~mnewcomb/radio/#iceabsorbtion
 */
Double_t temperature(Double_t z);

/*! \brief Takes in value of frequency in Ghz and depth z and returns you the value of attenuation length. This is used in the RK4 calculations.
 *
 */
Double_t Latten(Double_t z, Double_t frequency);

/*! \brief Takes in the value of depth z and calculates the derivative of amplitude Tau w.r.t to length of the step. 
 *
 *Basically it helps calculate that how much will the amplitude be attenuated in each step of the RK4 calculations
 */
Double_t ftau(Double_t tau,Double_t z);

/*! \brief Takes in the value of depth z and calculates the derivative of time w.r.t to length of the step. 
 *
 *Basically it helps calculate the value of time in each step in the RK4 calculations.
 */
Double_t ftime(Double_t z);

/*! \brief Takes in the value of Tx (x0,z0) and Rx (x1,z1) coordinates and calculates a guess for the launch angle of the Direct ray. 
 *
 *This function is just used to calculate launch angles for the Direct ray. There is a for loop which runs over all the possible values of the launch angle that can give a value closest to zero for the launch angle function. Please look at my raytracing_code.pdf presentation for more details about this function. 
 *If you want to understand how this function was derived look at Der_LauAng.pdf.
 */
Double_t *getDirAng(Double_t x0,Double_t z0, Double_t x1, Double_t z1);

/*! \brief Takes in the value of Tx (x0,z0) and Rx (x1,z1) coordinates and calculates a guess for the launch angle of the Reflected or Refracted ray. 
 *
 *This function is used to calculate launch angles for the Reflected & Refracted rays. For the refracted one I add 15 deg to the angle value it returns (implemented in getStartAngles() ). This is the primary method of calculating guess angles for the Reflected/Refracted rays if this fails I move on to the secondary method implemented in getStartAngles(). Please look at my raytracing_code.pdf presentation for more details about this function.
 */
Double_t getReflAng(Double_t x0, Double_t z0, Double_t x1, Double_t z1);

/*! \brief Takes in the value of Tx (x0,z0) and Rx (x1,z1) coordinates and calculates a guess for the start launch angle of whatever type of ray you are working with.
 *
 *Int_t solnum can have values 0 (Direct rays), 1 (Reflected rays) and 2 (Refracted rays).
 *For Direct rays it uses getDirAng() and for Reflected or Refracted rays it uses getReflAng. 
 *Then it corrects the launch angle a bit depending on what it gets from these functions. These corrections are just based on experience on how these functions work 
 *For Reflected rays if the primary method from getReflAng() fails to give any result then it uses the secondary method already present in it. Please refer to raytracing_code.pdf for more details.
 */
Double_t *getStartAngles(Double_t x0, Double_t z0, Double_t x1, Double_t z1, Int_t solnum);

/*! \brief Takes in the value of Tx (x0,z0) and Rx (x1,z1) coordinates and corrects/adjusts the launch angles based on the accuracy of the guess
 *
 *Int_t solnum can have values 0 (Direct rays), 1 (Reflected rays) and 2 (Refracted rays).
 *For Direct rays it corrects for preventing them from hitting the surface or from the bottom of the scan region if they miss the target.
 *For Reflected rays it corrects for if they do not hit the surface or miss the target and hit the bottom of the scan region.
 *For Refracted rays it corrects for preventing them from hitting the surface or from the bottom of the scan region if they miss the target.
 * Please refer to raytracing_code.pdf for more details.
 *These corrections are just based on experience on how these functions work 
 */
Double_t *getAngleCorrections(Int_t num, Int_t antco, Int_t D56co,Double_t test0, Double_t test1, Int_t psns,Int_t in,Int_t solnum);


Double_t* MRootScript_v2(Double_t c1,Double_t c2,Double_t c3,Double_t r1,Double_t r2,Double_t r3,Int_t wsol){

  Double_t *output=new Double_t[9];

  Double_t D56cor[3]{c1,c2,c3};//Tx positions
  Double_t antcor[3]={r1,r2,r3};//Rx Positions
  // string outFilename1="test.root";
  // Double_t D56cor[3]{200,0,-350};//Tx positions
  // Double_t antcor[3]={1000,0,-200};//Rx Positions

  Int_t isol_strt=0;
  Int_t isol_stop=1;

  //Find Direct Solutions
  if(wsol==0){
    isol_strt=0;
    isol_stop=1;
  }

  //Find Reflected Solutions
  if(wsol==1){
    isol_strt=1;
    isol_stop=2;
  }

  //Find Refracted Solutions
  if(wsol==2){
    isol_strt=2;
    isol_stop=3;
  }
  
  //Find Direct & Reflected Solutions
  if(wsol==3){
    isol_strt=0;
    isol_stop=2;
  }

   //Find Refracted & Reflected Solutions
  if(wsol==4){
    isol_strt=1;
    isol_stop=3;
  }

   //Find Direct,Refracted & Reflected Solutions
   if(wsol==5){
    isol_strt=0;
    isol_stop=3;
  }
  
  //const char *outFilename = outFilename1.c_str();
  //TFile *newfile = new TFile(outFilename,"recreate");

  const  Int_t nxyzout=3;
  const  Int_t nxzout=2;
  Int_t nxyz;
  Int_t nxz;
  Int_t ndhitbr;
  Int_t ovrallbr;
  Int_t isolbr;
  Double_t rTx[nxyzout];            
  Double_t rRx[nxyzout];
  Double_t xzTx[nxzout];               
  Double_t xzRx[nxzout];
  Double_t TransitTime;
  Double_t AttFac;
  Double_t L_ang;
  Double_t R_ang;
  Double_t I_ang;
  Double_t I_cor[nxzout];
  Bool_t WasSurfaceHit;
  
  TTree *chTreeR = new TTree("chTreeR","Channel info tree");
  chTreeR->Branch("nxyz",&nxyz,"nxyz/I");
  chTreeR->Branch("nxz",&nxz,"nxz/I");
  chTreeR->Branch("ndhitbr",&ndhitbr,"ndhitbr/I");
  chTreeR->Branch("ovrallbr",&ovrallbr,"ovrallbr/I");  
  chTreeR->Branch("isolbr",&isolbr,"isolbr/I");  
  chTreeR->Branch("rTx",rTx,"rTx[nxyz]/D");
  chTreeR->Branch("rRx",rRx,"rRx[nxyz]/D");
  chTreeR->Branch("xzTx",xzTx,"xzTx[nxz]/D");
  chTreeR->Branch("xzRx",xzRx,"xzRx[nxz]/D");
  chTreeR->Branch("TransitTime",&TransitTime,"TransitTime/D");
  chTreeR->Branch("AttFac",&AttFac,"AttFac/D");
  chTreeR->Branch("L_ang",&L_ang,"L_ang/D");
  chTreeR->Branch("R_ang",&R_ang,"R_ang/D");
  chTreeR->Branch("I_ang",&I_ang,"I_ang/D");
  chTreeR->Branch("I_cor",I_cor,"I_cor[nxz]/D");
  chTreeR->Branch("WasSurfaceHit",&WasSurfaceHit,"WasSurfaceHit/B");

  const Int_t N_tot=floor(tmax/h);
  
  Double_t K1_x;
  Double_t K2_x;
  Double_t K3_x;
  Double_t K4_x;
  
  Double_t K1_z;
  Double_t K2_z;
  Double_t K3_z;
  Double_t K4_z;

  Double_t K1_theta;
  Double_t K2_theta;
  Double_t K3_theta;
  Double_t K4_theta;

  Double_t K1_tau;
  Double_t K2_tau;
  Double_t K3_tau;
  Double_t K4_tau;

  Double_t K1_time;
  Double_t K2_time;
  Double_t K3_time;
  Double_t K4_time;

  Double_t Coef=1.0/6.0;
  Int_t i=0.0;
  Int_t brloop=0.0;
  Int_t wloop=0.0;
  Int_t chkn=0.0;
  Double_t s=0.0;
  Double_t x0,y0;
  Double_t x0d,y0d,z0d;
  Double_t z0=0.0;
  Double_t x1=0;
  Double_t z1=0;
  Double_t *ang=0;
  Double_t th0=0;
  Double_t dist=0;
  Double_t tau0=1.0;
  Double_t time0=0.0;
  Double_t th0i=0.0;
  
  Double_t minz=0.0;
  Double_t minx=0.0;
  Double_t minth=0.0;
  Double_t mintau=0.0;
  Double_t mintime=0.0;
  Double_t mindist=0.0;
  Double_t prdist=0.0;
  Bool_t dhit=true;
  Double_t testfir[2]={3*(pi/180),7*(pi/180)};
  Double_t dz[2]={0.0,0.0};
  Double_t midang=0;
  Int_t ndhit=0;
  Int_t ovrall=0;
  Double_t iniz=0;
  Int_t solexist=0;
  Bool_t surfhit=false;
  Bool_t depthpass[2]={false,false};

  for(Int_t isol=isol_strt;isol<isol_stop;isol++){
    if(isol==0){
      cout<<"~~~~~~~~~~~~~~~~~Finding DIRECT solutions~~~~~~~~~~~~~~~~~~"<<endl;
    }
    if(isol==1){
      cout<<"~~~~~~~~~~~~~~~~~Finding REFLECTED solutions~~~~~~~~~~~~~~~"<<endl;
    }
    if(isol==2){
      cout<<"~~~~~~~~~~~~~~~~~Finding REFRACTED solutions~~~~~~~~~~~~~~~"<<endl;
    }

    dhit=true;
    ndhit=0;
    ovrall=0;
    i=0;
    solexist=0;
    while(dhit==true){
      if(i>1){
      i=0.0;
      }
      wloop=0;
      brloop=0.0;
      s=0.0;
      x0=D56cor[0];
      y0=D56cor[1];
      z0=D56cor[2];
      x0d=antcor[0];
      y0d=antcor[1];
      z0d=antcor[2];
      iniz=z0;
      x1=sqrt(pow((x0d-x0),2)+pow((y0d-y0),2));
      dist=x1;
      z1=z0d;
      x0=0;
      if(ndhit==0 && i==0){
	Double_t *infir=getStartAngles(x0,z0,x1,z1,isol);
	testfir[0]=infir[0];
	testfir[1]=infir[1];
	cout<<"START angles "<<testfir[0]*(180/pi)<<" "<<testfir[1]*(180/pi)<<endl;
      }
      
      th0i=testfir[i];
      th0=th0i;
     
      cout<<"START: x0 "<<x0<<" ,z0 "<<z0<<" ,x1 "<<x1<<" ,z1 "<<z1<<" ,th0 "<<th0*(180/pi)<<" "<<i<<" "<<ndhit<<endl;

      tau0=1.0;
      time0=0.0;
      minz=1000.0;
      minx=1000.0;
      minth=0.0;
      mintau=0.0;
      mintime=0.0;
      mindist=0.0;
      prdist=1;
      chkn=0;
      surfhit=false;
      I_ang=0;
      I_cor[0]=0;
      I_cor[1]=0;
      depthpass[i]=false;
      while(wloop<N_tot){    
	
	K1_x=h*fx(th0);
	K2_x=h*fx(th0+0.5*K1_x);
	K3_x=h*fx(th0+0.5*K2_x);
	K4_x=h*fx(th0+K3_x);

	K1_z=h*fz(th0);
	K2_z=h*fz(th0+0.5*K1_z);
	K3_z=h*fz(th0+0.5*K2_z);
	K4_z=h*fz(th0+K3_z);

	K1_theta=h*ftheta(th0,              z0);
	K2_theta=h*ftheta(th0+0.5*K1_theta, z0+0.5*K1_theta);
	K3_theta=h*ftheta(th0+0.5*K2_theta, z0+0.5*K2_theta);
	K4_theta=h*ftheta(th0+K3_theta,     z0+K3_theta);
	
	K1_tau=h*ftau(tau0,                 z0);
	K2_tau=h*ftau(tau0+0.5*K1_tau,      z0+0.5*K1_tau);
	K3_tau=h*ftau(tau0+0.5*K2_tau,      z0+0.5*K2_tau);
	K4_tau=h*ftau(tau0+K3_tau,          z0+K3_tau);

	K1_time=h*ftime(z0);
	K2_time=h*ftime(z0+0.5*K1_time);
	K3_time=h*ftime(z0+0.5*K2_time);
	K4_time=h*ftime(z0+K3_time);

	x0=x0+Coef*(K1_x+2*(K2_x+K3_x)+K4_x);
	z0=z0+Coef*(K1_z+2*(K2_z+K3_z)+K4_z);
	th0=th0+Coef*(K1_theta+2*(K2_theta+K3_theta)+K4_theta);
	tau0=tau0-Coef*(K1_tau+2*(K2_tau+K3_tau)+K4_tau);
	time0=time0+Coef*(K1_time+2*(K2_time+K3_time)+K4_time);
	
	if(fabs(dist-x0)<0.1){
	  dz[i]=(z0-z1);
	  //cout<<"value of dz is "<<dz[i]<<" "<<i<<endl;
	}

	if(z0>0.0){
	  I_ang=th0*(180/pi);
	  I_cor[0]=x0;
	  I_cor[1]=z0;
	  th0=(pi/2-th0)+pi/2;
	  surfhit=true;
	  cout<<"Surface Hit"<<endl;
	}
	
	if(fabs(z0-z1)<0.5 && fabs(dist-x0)<0.5){	  
	  if(fabs(dist-x0)<0.1 && fabs(z0-z1)<0.1){
	    if(sqrt((z0-z1)*(z0-z1)+(dist-x0)*(dist-x0))<prdist){
	      prdist=sqrt((z0-z1)*(z0-z1)+(dist-x0)*(dist-x0));
	      minx=x0;
	      minz=z0;
	      minth=th0;
	      mintau=tau0;
	      mintime=time0;
	      mindist=dist;
	    }
	    chkn++;	  
	  }  
	}
		
	if(surfhit==true && (isol==0 || isol==2)){
	  wloop=N_tot+1;
	  brloop=1;
	  cout<<"Surface hit finishing loop"<<endl;
	}

	if(surfhit==false && dist-x0<0 && isol==1){
	  wloop=N_tot+1;
	  brloop=1;
	  cout<<"Surface is not being hit"<<endl;
	}
	
	if((dist-x0)<-2){
	  wloop=N_tot+1;
	  brloop=0;
	  cout<<"Target has been passed!"<<endl;
	}

	if((z0-(-1000))<-2){
	  wloop=N_tot+1;
	  brloop=1;
	  cout<<"Ray going straight down! Target has been missed!"<<endl;
	  depthpass[i]=true;
	  
	}

	if(chkn>0 && dist-x0<0){
	  wloop=N_tot+1;
	  brloop=1;
	  cout<<"Target hit multiple times"<<endl;
	}
	
	s=s+h;
	wloop++;
      }//while loop
      
      if(chkn==0 && i==1 && brloop==0){
	
	Double_t zdiff=0;
	Double_t newang=0;	
	Double_t tet0=testfir[0];
	Double_t tet1=testfir[1];

	if(dz[1]>dz[0]){
	  zdiff=dz[1]-dz[0];
	  cout<<"dz1>dz0"<<endl;
	  newang=testfir[0]+(fabs(dz[0])/zdiff)*(testfir[1]-testfir[0]);
	}
	if(dz[0]>dz[1]){
	  zdiff=dz[0]-dz[1];
	  cout<<"dz0>dz1"<<endl;
	  newang=testfir[1]+(fabs(dz[1])/zdiff)*(testfir[0]-testfir[1]);
	}
	
	//cout<<"newang is "<<newang*(180.0/pi)<<" "<<dz[1]-dz[0]<<" "<<testfir[1]-testfir[0]<<endl;

	testfir[1]=newang+fabs(tet1-tet0)*0.3;
	testfir[0]=newang-fabs(tet1-tet0)*0.3;

	if(dz[0]==0 && dz[1]==0){
	  newang=20*(pi/180);
	  cout<<"Both dz are zero!"<<endl;
	  testfir[1]=newang+2*(pi/180);
	  testfir[0]=newang-2*(pi/180);
	}

	if(testfir[0]<0){
	  newang=2.5*(pi/180);
	  if(isol==2){
	    newang=65.0*(pi/180);
	  }
	  testfir[1]=newang+fabs(tet1-tet0)*0.3;
	  testfir[0]=newang-fabs(tet1-tet0)*0.3;
	}

	if(newang>(65*(pi/180)) && (isol==1)){
	  newang=5.0*(pi/180);
	  testfir[1]=newang+fabs(tet1-tet0)*0.3;
	  testfir[0]=newang-fabs(tet1-tet0)*0.3;
	}
	
	if(newang>pi && (isol==1 || isol==2)){
	  newang=50*(pi/180);
	  testfir[1]=newang+fabs(tet1-tet0)*0.3;
	  testfir[0]=newang-fabs(tet1-tet0)*0.3;
	}

	if(newang>pi && isol==0){
	  newang=100*(pi/180);
	  testfir[1]=newang+fabs(tet1-tet0)*0.3;
	  testfir[0]=newang-fabs(tet1-tet0)*0.3;
	}

	cout<<"new angles "<<testfir[0]*(180/pi)<<" "<<testfir[1]*(180/pi)<<endl;

	if(testfir[1]>pi || testfir[0]>pi){
	  dhit=false;
          cout<<"NO solution. Angle got too large."<<chkn<<endl;
	}
	
	if(testfir[1]<0.1*(pi/180) || testfir[0]<0.1*(pi/180)){
	  dhit=false;
	  cout<<"NO solution. Angle got too small."<<chkn<<endl;
	}
	
	if(iniz==0){
	  cout<<"Source is right at the surfcace!!"<<endl;
	  if(testfir[0]<pi/2){
	    testfir[0]=(pi/2)+0.01*(pi/180);
	  }
	  if(testfir[1]<pi/2){
	    testfir[1]=(pi/2)+2*(pi/180);
	  }
	}
	
	dz[0]=0;
	dz[1]=0;
    
      }
   
      if(chkn>0 && surfhit==false && isol==0){
	dhit=false;
      }

      if(chkn>0 && surfhit==true && isol==1){
	dhit=false;
      }

      if(chkn>0 && surfhit==false && isol==2){
	dhit=false;
      }      
      
      if((ndhit>34 || ovrall>63) && chkn==0){
	dhit=false;
	cout<<"NO solution "<<chkn<<" "<<ndhit<<endl;
      }
      //cout<<i<<" ivalue "<<endl;
      i++;

      if((surfhit==true) && (isol==0 || isol==2)){
	Double_t *infir2=getAngleCorrections(ndhit,antcor[0],D56cor[0],testfir[0], testfir[1], +1,i,isol);
	if(surfhit==true){
	  cout<<"changing angles. surface was hit "<<endl;
	}
	if(i==1){
	  i=0;
	  testfir[0]=infir2[0];
	  testfir[1]=infir2[1];
	}
	if(i==2){
	  i=1;
	  testfir[0]=infir2[0];
	  testfir[1]=infir2[1];
	}
	if(ndhit>1){
	  ndhit--;
	}
      }
      
      if((depthpass[0]==true || depthpass[1]==true) && (isol==0 || isol==2)){
	Double_t *infir2=getAngleCorrections(ndhit,antcor[0],D56cor[0],testfir[0], testfir[1], -1,i,isol);
       	cout<<"changing angles. bottom was hit."<<endl;
	if(i==1){
	  i=0;
	  testfir[0]=infir2[0];
	  testfir[1]=infir2[1];
	}
	if(i==2){
	  i=1;
	  testfir[1]=infir2[1]; 
	}
	if(ndhit>1){
	  ndhit--;
	}
      }
      
      if((depthpass[0]==true || depthpass[1]==true) && (isol==1)){
	Double_t *infir2=getAngleCorrections(ndhit,antcor[0],D56cor[0],testfir[0], testfir[1], +1,i,isol);
       	cout<<"changing angles. bottom was hit "<<endl;
	if(i==1){
	 i=0;
	 testfir[0]=infir2[0];
	 testfir[1]=infir2[1];
	}
	if(i==2){
	 i=1;
	 testfir[0]=infir2[0];
	 testfir[1]=infir2[1];
	}
	if(ndhit>1){
	  ndhit--;
	}
      }

      if((surfhit==false) && (isol==1)){
	Double_t *infir2=getAngleCorrections(ndhit,antcor[0],D56cor[0],testfir[0], testfir[1], -1,i,isol);
       	cout<<"changing angles. surface is NOT being hit."<<endl;
	if(i==1){
	 i=0;
	 testfir[0]=infir2[0];
	 testfir[1]=infir2[1];
	}
	if(i==2){
	  i=1;
	  testfir[1]=infir2[1];
	}
	if(ndhit>1){
	  ndhit--;
	}
      }
      
      ndhit++;
      ovrall++;
    }///dhit
    
    if(chkn>0 && surfhit==false && (isol==0 || isol==2)){
      if(isol==0){
      cout<<"x0 "<<D56cor[0]<<" ,z0 "<<iniz<<" ,x1 "<<minx<<" ,z1 "<<minz<<" ,th0 "<<minth*(180/pi)<<" ,Lang "<<th0i*(180/pi)<<" ,time0 "<<mintime<<" ,tau0 "<<mintau<<" ,I_ang "<<I_ang<<" ,I_corx "<<I_cor[0]<<" ,Icorz "<<I_cor[1]<<" Direct HIT!!. Writing file. "<<ovrall<<" "<<ndhit<<endl;
      }

      if(isol==2){
      cout<<"x0 "<<D56cor[0]<<" ,z0 "<<iniz<<" ,x1 "<<minx<<" ,z1 "<<minz<<" ,th0 "<<minth*(180/pi)<<" ,Lang "<<th0i*(180/pi)<<" ,time0 "<<mintime<<" ,tau0 "<<mintau<<" ,I_ang "<<I_ang<<" ,I_corx "<<I_cor[0]<<" ,Icorz "<<I_cor[1]<<" Refracted HIT!!. Writing file. "<<ovrall<<" "<<ndhit<<endl;
      }
      
      cout<<((mindist-minx))<<" "<<minz-z1<<" "<<mindist<<" hit room"<<endl;
      
      nxyz=3;
      nxz=2;
      ndhitbr=ndhit;
      ovrallbr=ovrall;
      isolbr=isol;
      
      rTx[0]=D56cor[0];
      rTx[1]=D56cor[1];
      rRx[0]=antcor[0];
      rRx[1]=antcor[1];
      
      xzTx[0]=D56cor[0];
      xzTx[1]=iniz;
      xzRx[0]=minx;
      xzRx[1]=minz;
      TransitTime=mintime;
      AttFac=mintau;
      L_ang=th0i*(180/pi);
      R_ang=minth*(180/pi);
      WasSurfaceHit=surfhit;
      if(isol==0){
	output[0]=L_ang;
	output[1]=R_ang;
	output[2]=TransitTime;      
      }

      if(isol==2){
	output[3]=L_ang;
	output[4]=R_ang;
	output[5]=TransitTime;      
      }

      chTreeR->Fill();
      solexist=1;
    }
    
    if(chkn>0 && surfhit==true && isol==1){
      cout<<"x0 "<<D56cor[0]<<" ,z0 "<<iniz<<" ,x1 "<<minx<<" ,z1 "<<minz<<" ,th0 "<<minth*(180/pi)<<" ,Lang "<<th0i*(180/pi)<<" ,time0 "<<mintime<<" ,tau0 "<<mintau<<" ,I_ang "<<I_ang<<" ,I_corx "<<I_cor[0]<<" ,Icorz "<<I_cor[1]<<" Reflected HIT!!. Writing file. "<<ovrall<<" "<<ndhit<<endl;
      cout<<((mindist-minx))<<" "<<minz-z1<<" "<<mindist<<" hit room"<<endl;
      
      nxyz=3;
      nxz=2;
      ndhitbr=ndhit;
      ovrallbr=ovrall;
      isolbr=isol;
      
      rTx[0]=D56cor[0];
      rTx[1]=D56cor[1];
      rRx[0]=antcor[0];
      rRx[1]=antcor[1];
      
      xzTx[0]=D56cor[0];
      xzTx[1]=iniz;
      xzRx[0]=minx;
      xzRx[1]=minz;
      TransitTime=mintime;
      AttFac=mintau;
      L_ang=th0i*(180/pi);
      R_ang=minth*(180/pi);
      WasSurfaceHit=surfhit;

      output[6]=L_ang;
      output[7]=R_ang;
      output[8]=TransitTime;      
      
      chTreeR->Fill();
      solexist=1;
    }

    if((ndhit>34 || chkn==0 || ovrall>63) && solexist==0){
      cout<<"NO solution. Writing file. "<<chkn<<endl;
      cout<<"x0 "<<D56cor[0]<<" ,z0 "<<iniz<<" ,x1 "<<minx<<" ,z1 "<<minz<<" ,th0 "<<minth*(180/pi)<<" ,Lang "<<th0i*(180/pi)<<" ,time0 "<<mintime<<" ,tau0 "<<mintau<<" ,I_ang "<<I_ang<<" ,I_corx "<<I_cor[0]<<" ,Icorz "<<I_cor[1]<<" MISS!! "<<endl;
      cout<<((mindist-minx))<<" "<<minz-z1<<" "<<mindist<<" hit room"<<endl;
      
      nxyz=3;
      nxz=2;
      ndhitbr=ndhit;
      ovrallbr=ovrall;
      isolbr=isol;
      
      rTx[0]=D56cor[0];
      rTx[1]=D56cor[1];
      rRx[0]=antcor[0];
      rRx[1]=antcor[1];
      
      xzTx[0]=D56cor[0];
      xzTx[1]=iniz;
      xzRx[0]=minx;
      xzRx[1]=minz;
      TransitTime=mintime;
      AttFac=mintau;
      L_ang=th0i*(180/pi);
      R_ang=-1;
      WasSurfaceHit=surfhit;
      
      output[0]=L_ang;
      output[1]=R_ang;
      output[2]=TransitTime;      
      
      output[3]=L_ang;
      output[4]=R_ang;
      output[5]=TransitTime;      
      
      output[6]=L_ang;
      output[7]=R_ang;
      output[8]=TransitTime;      
      
      chTreeR->Fill();
    }
    
  }//isol loop

  // newfile->Write();
  // newfile->Close();    
  return output;
}

Double_t n(Double_t z){
  return A+B*exp(C*z);
}

Double_t dndz(Double_t z){
  return B*C*exp(C*z);
}
  
Double_t fx(Double_t theta){
  return sin(theta);
}

Double_t fz(Double_t theta){  
  return cos(theta);
}

Double_t ftheta(Double_t theta, Double_t z){  
  return -sin(theta)*(1.0/n(z))*dndz(z);
}

Double_t temperature(Double_t z){
	return(-51.5 + z*(-4.5319e-3 + 5.822e-6*z));
}

Double_t Latten(Double_t z, Double_t frequency){
  //	if(z>0.0)		return(std::numeric_limits<Double_t>::infinity());
  Double_t t = temperature(z);
  const Double_t f0=0.0001, f2=3.16;
  const Double_t w0=log(f0), w1=0.0, w2=log(f2), w=log(frequency);
  const Double_t b0=-6.74890+t*(0.026709-t*0.000884);
  const Double_t b1=-6.22121-t*(0.070927+t*0.001773);
  const Double_t b2=-4.09468-t*(0.002213+t*0.000332);
  Double_t a,bb;
  if(frequency<1.){
    a=(b1*w0-b0*w1)/(w0-w1);
    bb=(b1-b0)/(w1-w0);
  }
  else{
    a=(b2*w1-b1*w2)/(w1-w2);
    bb=(b2-b1)/(w2-w1);
  }
  return 1./exp(a+bb*w);
}

Double_t ftau(Double_t tau,Double_t z){  
  // Double_t Latten=1000;
  Double_t Lavg=0;
  Int_t frband=floor((0.8-0.2)/0.1);
  Double_t step=0.2;
  for(Int_t i=0;i<frband;i++){
    Lavg+=Latten(z,step);
    step=step+0.1;
  }
  Lavg=Lavg/(Double_t)frband;
  return tau/Lavg;
}

Double_t ftime(Double_t z){  
  return n(z)/spedc;
}


Double_t *getDirAng(Double_t x0,Double_t z0, Double_t x1, Double_t z1){

  // Double_t A=1.788;
  // Double_t B=-0.463;
  // Double_t C=0.014;//// no negative sign for negative depth
  // Double_t pi=4.0*atan(1.0);//// no negative sign for negative depth
  
  // Double_t z0=0;
  // Double_t z1=-180;
  Double_t n0=A+B*exp(C*z0);
  Double_t n=A+B*exp(C*z1);
  // Double_t x0=0;
  // Double_t x1=10;
  Bool_t swapbool=(n<n0);
  if(swapbool) {
    //swap(n,n0);
    n=A+B*exp(C*z0);
    n0=A+B*exp(C*z1);
    cout<<"SWAP"<<endl;
    //std::cout<<"swap!"<<std::endl; // changed (so that z0 is always near surface?)
  }
  
  Double_t sDiff=C*fabs(x0-x1);
 
  TF1 *fn=new TF1("fn","log((((sqrt([0]*[0]-sin(x)*sin(x)*[1]*[1])+sqrt([2]*[2]-sin(x)*sin(x)*[1]*[1]))/([0]-[2]))+([2]/sqrt([2]*[2]-sin(x)*sin(x)*[1]*[1])))/(((sqrt([1]*[1]-sin(x)*sin(x)*[1]*[1])+sqrt([2]*[2]-sin(x)*sin(x)*[1]*[1]))/([1]-[2]))+([2]/sqrt([2]*[2]-sin(x)*sin(x)*[1]*[1]))))-((sqrt([2]*[2]-sin(x)*sin(x)*[1]*[1])*[3])/(sin(x)*[1]))",0,pi);
  fn->FixParameter(0,n);
  fn->FixParameter(1,n0);
  fn->FixParameter(2,A);
  fn->FixParameter(3,sDiff);

  Double_t valarr1[20];
  Double_t valt1[20];
  Double_t valarr2[20];
  Double_t valt2[20];
  
  Double_t h=0.00001;//0.0000001;
  Int_t N=floor(pi/h)+1;
  Double_t value;
  Double_t t=0.0;
  Int_t n1=0;
  Int_t n2=0;
  Bool_t n1bool=true;
  Bool_t n2bool=true;  
  Double_t fnmax=-10000000;
  Double_t fnmaxloc=0;
  for(Int_t i=0;i<N;i++){
    value=fn->Eval(t);
    if(t>1 && t<2){
      if(value>fnmax){
	fnmax=value;
	fnmaxloc=t;
      }
    }
    t=t+h;   
  }
  // cout<<"fnmax "<<fnmax<<" fnmaxloc "<<fnmaxloc<<endl;
  t=0;
  for(Int_t i=0;i<N;i++){
    value=fn->Eval(t);
    if(value<0.01 && value>-0.01 && n1bool==true){
      valarr1[n1]=fabs(value);
      valt1[n1]=t;
      //  cout<<"t "<<t<<" , value "<<valarr1[n1]<<" "<<n1<<endl;
      n1++;
      if(n1==20)n1bool=false;
    }
    
    if(value<0.01 && value>-0.01 && t>fnmaxloc && n2bool==true ){
      valarr2[n2]=fabs(value);
      valt2[n2]=t;
      //       cout<<"t "<<t<<" , value "<<valarr2[n2]<<" "<<n2<<endl;
      n2++;
      if(n2==20)n2bool=false;
    }
    t=t+h;
  }
  
  Double_t minel1=TMath::MinElement(n1,valarr1);
  Int_t locmin1=TMath::LocMin(n1,valarr1);
  Double_t minel2=TMath::MinElement(n2,valarr2);
  Int_t locmin2=TMath::LocMin(n2,valarr2);

  Double_t *th0p=new Double_t[2];

  if(n1>0){
    th0p[0]=valt1[locmin1];
  } 
  if(n1==0){
    th0p[0]=0;
  }
  if(n2>0){
    th0p[1]=valt2[locmin2];
  }
  if(n2==0){
    th0p[1]=0;
  }

  // cout<<"z0 "<<z0<<" , x0 "<<x0<<", z "<<z1<<" , x1 "<<x1<<" "<<th0p[0]*(180/pi)<<" "<<th0p[1]*(180/pi)<<" launch output"<<endl;
  return th0p;
  
}


Double_t *getStartAngles(Double_t x0, Double_t z0, Double_t x1, Double_t z1, Int_t solnum){
  
  Double_t midang=0;
  Double_t *outfir=new Double_t[2];
  outfir[0]=0;
  outfir[1]=0;
  
  if(solnum==1 || solnum==2){

    midang=getReflAng(x0,z0,x1,z1);

    if(solnum==2 && midang!=0){
      midang=midang+15*(pi/180.0);
    }
    
    if(midang==0){
      midang=((pi/2)-atan(fabs((z0*2.0)/(x1-x0))));
      if(midang>49*(pi/180)){
	midang=midang-15*(pi/180);
      }
      if(midang<50*(pi/180)){
	midang=midang-0.5*(pi/180);
      } 
    }
    
    outfir[0]=midang-2*(pi/180);
    outfir[1]=midang+2*(pi/180);
    
    if(outfir[0]<0*(pi/180)){
      midang=3*(pi/180);
      outfir[0]=midang-2*(pi/180);
      outfir[1]=midang+2*(pi/180);
    }  
    
    if(outfir[1]>60*(pi/180) && solnum==1){
      //midang=3*(pi/180);
      outfir[0]=outfir[0]-2*(pi/180);
      outfir[1]=outfir[1]-2*(pi/180);
    }  
    
  }  
  
  if(solnum==0){
    
    Double_t *ang2=getDirAng(x0,z0,x1,z1);
    midang=ang2[1];
    
    if(z0<z1){
      midang=ang2[0];
    }
    
    if(z1>z0 && midang>90*(pi/180)){
      midang=pi-midang;
    }
    
    outfir[0]=midang-2*(pi/180);
    outfir[1]=midang+2*(pi/180);
    
    if(fabs(z0-z1)<0.15 || ang2[0]<0.01 || ang2[1]<0.01){
      midang=(pi/2)-atan(fabs((z1-z0)/x1));
      
      if(midang>pi/2){
	midang=(pi/2)-2*(pi/180);
      }
    
      outfir[0]=midang-2*(pi/180);
      outfir[1]=midang+2*(pi/180);
      
      if(outfir[0]<0*(pi/180)){
	midang=3*(pi/180);
	outfir[0]=midang-2*(pi/180);
	outfir[1]=midang+2*(pi/180);
      }
      
    }
    delete ang2;
  }
  
  return outfir;
}

Double_t *getAngleCorrections(Int_t num, Int_t antco, Int_t D56co,Double_t test0, Double_t test1, Int_t psns, Int_t in,Int_t solnum){
  
  if(psns<0 && in==1){
    test0=test0-fabs(test1-test0)*0.6;
    test1=test1-fabs(test1-test0)*0.6;
  }

  if(psns<0 && in==2){
    test1=test1-fabs(test1-test0)*0.6;
  }
  
  if(psns>0){
    if(num>8 && fabs(antco-D56co)<=200){
      test0=test0+0.1*(pi/180)*psns;
      test1=test1+0.1*(pi/180)*psns;
    }  
    if(num<9 && fabs(antco-D56co)<=200){
      test0=test0+0.5*(pi/180)*psns;
      test1=test1+0.5*(pi/180)*psns;
    }
    if(num>8 && fabs(antco-D56co)>200){
      test0=test0+2*(pi/180)*psns;
      test1=test1+2*(pi/180)*psns;
    }
    if(num<9 && fabs(antco-D56co)>200){
      test0=test0+4*(pi/180)*psns;
      test1=test1+4*(pi/180)*psns;
    }
  }

  if(psns>0 && solnum==2){
    if(num>8 && fabs(antco-D56co)<=200){
      test0=test0+1*(pi/180)*psns;
      test1=test1+1*(pi/180)*psns;
    }
    if(num<9 && fabs(antco-D56co)<=200){
      test0=test0+2*(pi/180)*psns;
      test1=test1+2*(pi/180)*psns;
    }
  }
 
  Double_t *outfir2=new Double_t[2];
  outfir2[0]=test0;
  outfir2[1]=test1;

  return outfir2;
}

Double_t getReflAng(Double_t x0, Double_t z0, Double_t x1, Double_t z1){
  // Double_t x0=0;
  // Double_t x1=150;

  // Double_t z0=-150;
  // Double_t z1=-200;

  Double_t fmin=fabs(z0);
  Double_t fmax=sqrt(z0*z0+(x0-x1)*(x0-x1));

  Double_t gmin=fabs(z1);
  Double_t gmax=sqrt(z1*z1+(x0-x1)*(x0-x1));

  Double_t thf=pi/2-acos(fabs(x1-x0)/fmax);
  Double_t thg=pi/2-acos(fabs(x1-x0)/gmax);

  Double_t thl=0;
  Double_t thh=0;
  //cout<<gmin<<" "<<gmax<<" "<<fmin<<" "<<fmax<<" "<<thl<<" "<<thh<<" "<<thg<<" "<<thf<<endl;
  Bool_t con1=false;
    
  Int_t num=500;
  Int_t iplt=0;    

  if(thf>thg){
    thl=thg;
    thh=thf;
    con1=true;
  }
  if(thg>thf && con1==false){
    thl=thf;
    thh=thg;
  }
  if(thg==thf){
    thl=thf;
    thh=thg;
  }

  Double_t rtrnang=0;
  
  Double_t dummyg=0;
  Double_t dummyf=0;
  Double_t dummyth=0;
  iplt=0;
  for(Int_t ign=0;ign<num;ign++){
    for(Int_t ifn=0;ifn<num;ifn++){
      for(Int_t itn=0;itn<num;itn++){
	dummyg=gmin+(fabs(gmin-gmax)/num)*ign;
	dummyf=fmin+(fabs(fmin-fmax)/num)*ifn;
	dummyth=thl+(fabs(thl-pi)/num)*itn;
	
	if(dummyg>gmin && dummyg<gmax && dummyf>fmin && dummyf<fmax && dummyth>thl && dummyth<pi){
	  if(fabs(fabs(z0/dummyf)-fabs(z1/dummyg))<0.01){
	    if(fabs(pow(dummyg,2)+pow(dummyf,2)-2*dummyg*dummyf*cos(dummyth)-pow(sqrt(pow(x0-x1,2)+pow(z0-z1,2)),2))<1 && iplt==0){
	      //cout<<ix<<" "<<iz<<" "<<ign<<" "<<dummyg<<" "<<ifn<<" "<<dummyf<<" "<<fabs(z0/dummyf)-fabs(z1/dummyg)<<" "<<thl*(180.0/pi)<<" "<<fabs(pow(dummyg,2)+pow(dummyf,2)-2*dummyg*dummyf*cos(dummyth)-pow(sqrt(pow(x0-x1,2)+pow(z0-z1,2)),2))<<endl;
	      rtrnang=(thl/2)-2.0*(pi/180.0);
	      iplt++;
	    }
	  }
	}	
      }
    }
  }

  return rtrnang;
}
  
