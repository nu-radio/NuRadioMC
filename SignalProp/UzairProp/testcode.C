#include <iostream>
#include <fstream>
#include <string>
#include "RayTraceRK4.C"
void testcode(){
  
  //Size of the frequency array
  Int_t size=10;
  
  //Frequency Array being defined and initialised in Ghz
  Double_t *freqarray=new Double_t[size];
  for(Int_t i=0;i<size;i++){
    freqarray[i]=0.1*i;
  }

  //Call the raytrace function at the specified Tx (800,0,-300)  and Rx (1000,0,-200) coordinates
  //For this example only look at Direct and Reflected ray (i.e. wsol==3)
  Double_t *getres=RayTraceRK4(800,0,-300,1000,0,-200,3,freqarray,size);

  cout<<"START THE OUTPUT"<<endl;
  //print the results of the pointer
  cout<<"Launch Angle (deg), Recieve Angle (deg) and Time of Direct Ray (s) in order"<<endl;
  for(Int_t i=0;i<3;i++){
    cout<<getres[i]<<endl;
  }
  cout<<"Launch Angle (deg), Recieve Angle (deg) and Time of Reflected Ray (s) in order"<<endl;
  for(Int_t i=3;i<6;i++){
    cout<<getres[i]<<endl;
  }
  // cout<<"Launch Angle (deg), Recieve Angle (deg) and Time of Refracted Ray (s) in order"<<endl;
  // for(Int_t i=6;i<9;i++){
  //   cout<<getres[i]<<endl;
  // }
  
  cout<<"Fraction of amplitudes left after attenuation at the frequency values specified in the array of tauszie length for Direct Rays"<<endl;
  for(Int_t i=0;i<size;i++){
    cout<<getres[i+9]<<" "<<i<<endl;
  }
    cout<<"Fraction of amplitudes left after attenuation at the frequency values specified in the array of tauszie length for Reflected Rays"<<endl;
  for(Int_t i=size;i<size*2;i++){
    cout<<getres[i+9]<<" "<<i<<endl;
  }
  // cout<<"Fraction of amplitudes left after attenuation at the frequency values specified in the array of tauszie length for Refracted Rays"<<endl;
  // for(Int_t i=size*2;i<size*3;i++){
  //   cout<<getres[i+9]<<" "<<i<<endl;
  // }
  
}
