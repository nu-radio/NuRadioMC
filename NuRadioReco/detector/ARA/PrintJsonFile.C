#include <iostream>
#include "AraGeomTool.h"
#include <fstream>

void PrintJsonFile(){

  ofstream aout("ara_detector_db.json");
  Double_t pi=TMath::Pi();
  Int_t nich=0;
  Double_t alpha= (90+36.773)*(pi/180.0);
  aout<<"{"<<endl;
  aout<<"{"<<endl;
  aout<<"\t\"_default\": {},"<<endl;
  aout<<"\t\"channels\": {"<<endl;

  for(int ist=1;ist<5;ist++) {
    Double_t stationId=ist;

    if(ist==1){
      stationId=100;
    }

    AraStationInfo *stationInfo=new AraStationInfo(stationId,2018);
    for(int ich=0;ich<16;ich++) {
      Double_t *antLocTx=stationInfo->getAntennaInfo(ich)->getLocationXYZ();
      TVector3 v1(antLocTx[0],antLocTx[1],0);
      v1.RotateZ(-alpha);
      Double_t dumx=v1.X();
      Double_t dumy=v1.Y();

      aout<<"\t\""<<nich<<"\": {"<<endl;
      aout<<"\t\t\"adc_id\": null,"<<endl;
      aout<<"\t\t\"adc_n_samples\": null,"<<endl;
      aout<<"\t\t\"adc_nbits\": null,"<<endl;
      aout<<"\t\t\"adc_sampling_frequency\": 1.6,"<<endl;
      aout<<"\t\t\"adc_time_delay\": null,"<<endl;
      aout<<"\t\t\"amp_reference_measurement\": null,"<<endl;
      aout<<"\t\t\"amp_type\": null,"<<endl;
      aout<<"\t\t\"ant_comment\": \"ARA"<<stationId<<" "<<stationInfo->getAntennaInfo(ich)->getRFChanName()<<" channel"<<ich<<"\","<<endl;
      if(ist==1){
        aout<<"\t\t\"ant_deployment_time\": \"{TinyDate}:2011-15-01T00:00:00\","<<endl;
      }
      if(ist==2 || ist==3){
        aout<<"\t\t\"ant_deployment_time\": \"{TinyDate}:2012-15-01T00:00:00\","<<endl;
      }
      if(ist==4 || ist==5){
        aout<<"\t\t\"ant_deployment_time\": \"{TinyDate}:2017-15-01T00:00:00\","<<endl;
      }
      aout<<"\t\t\"ant_orientation_phi\": 0.0,"<<endl;
      aout<<"\t\t\"ant_orientation_theta\": 0.0,"<<endl;
      aout<<"\t\t\"ant_position_x\": "<<dumx<<","<<endl;
      aout<<"\t\t\"ant_position_y\": "<<dumy<<","<<endl;
      aout<<"\t\t\"ant_position_z\": "<<antLocTx[2]<<","<<endl;
      aout<<"\t\t\"ant_rotation_phi\": 0.00,"<<endl;
      aout<<"\t\t\"ant_rotation_theta\": 0.0,"<<endl;
      if(ich<8){
        aout<<"\t\t\"ant_type\": \"XFDTD_Vpol_CrossFeed_150mmHole_n1.78\","<<endl;
      }
      if(ich>7){
        aout<<"\t\t\"ant_type\": \"XFDTD_Hpol_150mmHole_n1.78\","<<endl;
      }
      aout<<"\t\t\"cab_id\": null,"<<endl;
      aout<<"\t\t\"cab_length\": null,"<<endl;
      aout<<"\t\t\"cab_reference_measurement\": null,"<<endl;
      aout<<"\t\t\"cab_time_delay\": "<<stationInfo->getAntennaInfo(ich)->getCableDelay()<<","<<endl;
      aout<<"\t\t\"cab_type\": null,"<<endl;
      aout<<"\t\t\"channel_id\": "<<ich<<","<<endl;
      if(ist==1){
        aout<<"\t\t\"commission_time\": \"{TinyDate}:2011-15-01T00:00:00\","<<endl;
      }
      if(ist==2 || ist==3){
        aout<<"\t\t\"commission_time\": \"{TinyDate}:2012-15-01T00:00:00\","<<endl;
      }
      if(ist==4 || ist==5){
        aout<<"\t\t\"commission_time\": \"{TinyDate}:2017-15-01T00:00:00\","<<endl;
      }
      aout<<"\t\t\"decommission_time\": null,"<<endl;
      aout<<"\t\t\"station_id\": "<<stationId<<endl;
      if(nich<79){
        aout<<"\t\t},"<<endl;
      }
      nich++;
    }
    delete stationInfo;
  }
  aout<<"\t\t}"<<endl;
  aout<<"\t\t},"<<endl;
  aout<<"\t\t\"stations\": {"<<endl;

  Double_t StationCor[6][3];
  Int_t nist=0;

  for(int ist=1;ist<6;ist++) {
    Int_t stationId=ist;

    if(ist==1){
      stationId=100;
    }

    AraStationInfo *stationInfo=new AraStationInfo(stationId,2018);
    AraGeomTool *geom = AraGeomTool::Instance();
    StationCor[ist][0]=geom->getStationVector(stationId).x();
    StationCor[ist][1]=geom->getStationVector(stationId).y();
    StationCor[ist][2]=geom->getStationVector(stationId).z();
    Double_t eastings=(StationCor[ist][0]+22399.60*0.3048)*3.280284;
    Double_t northings=(StationCor[ist][1]+53907.39*0.3048)*3.280284;
    Double_t altitude=(StationCor[ist][2]+9312*0.3048)*3.280284;

    aout<<"\t\t\""<<ist<<"\": {"<<endl;
    aout<<"\t\t\"MAC_address\": null,"<<endl;
    aout<<"\t\t\"MBED_type\": null,"<<endl;
    aout<<"\t\t\"board_number\": null,"<<endl;
    if(ist==1){
      aout<<"\t\t\"commission_time\": \"{TinyDate}:2011-15-01T00:00:00\","<<endl;
    }
    if(ist==2 || ist==3){
      aout<<"\t\t\"commission_time\": \"{TinyDate}:2012-15-01T00:00:00\","<<endl;
    }
    if(ist==4 || ist==5){
      aout<<"\t\t\"commission_time\": \"{TinyDate}:2017-15-01T00:00:00\","<<endl;
    }
    aout<<"\t\t\"decommission_time\": \"{TinyDate}:2038-01-01T00:00:00\","<<endl;
    aout<<"\t\t\"pos_altitude\": "<<altitude<<","<<endl;
    aout<<"\t\t\"pos_easting\": "<<eastings<<","<<endl;
    aout<<"\t\t\"pos_measurement_time\": null,"<<endl;
    aout<<"\t\t\"pos_northing\": "<<northings<<","<<endl;
    aout<<"\t\t\"pos_position\": null,"<<endl;
    aout<<"\t\"pos_site\": \"southpole\","<<endl;
    aout<<"\t\t\"pos_zone\": \"SP-grid\","<<endl;
    aout<<"\t\t\"position\": null,"<<endl;
    aout<<"\t\t\"station_id\": "<<stationId<<","<<endl;
    aout<<"\t\t\"station_type\": null"<<endl;
    if(ist<5){
      aout<<"\t\t},"<<endl;
    }
    if(ist==5){
      aout<<"\t\t}"<<endl;
    }
    nist++;
    delete stationInfo;
  }

  aout<<"\t\t}"<<endl;
  aout<<"\t}"<<endl;
}
