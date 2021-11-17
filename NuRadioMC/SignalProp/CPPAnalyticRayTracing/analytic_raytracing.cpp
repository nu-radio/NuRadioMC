//basically a c++ recasting of the python code written by c glaser

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <math.h>
#include <vector>
#include <stdio.h>
#include <time.h>

#include <fstream>
#include <sstream>

//for gsl numerical integration
#include <gsl/gsl_integration.h>

//for gsl root finding
#include <gsl/gsl_errno.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_roots.h>
#include <units.h>
#include <attenuation.h>


using namespace std;

//some global constants
double speed_of_light = 299792458 * utl::m/utl::s; //meters/second
double pi = atan(1.)*4.; //compute and store pi
double inf = 1e130; //infinity for all practical purposes...

double index_vs_depth(double z, double n_ice, double delta_n, double z_0){
	//return the index of refraction at a given depth
	double index = n_ice - (delta_n*exp(z/z_0));
	return index;
}

double get_gamma(double z, double n_ice, double delta_n, double z_0){
	return delta_n * exp(z/z_0);
}

void get_turning_point(double c, double &gamma2, double &z2, double n_ice, double delta_n, double z_0){
	//calculate the turning point (the maximum of the ray tracing path)
	double b = 2. * n_ice;
	gamma2 = b*0.5 - sqrt(0.25 * pow(b,2.) - c);
	z2 = log(gamma2/delta_n) * z_0;
}

double get_y(double gamma, double C0, double C1, double n_ice, double delta_n, double z_0){
	//parameters
	//gamma: gamma is a function of the depth z
	//c0: first parameter
	//c1: second paramter
	double b = 2. * n_ice;
	double c = pow(n_ice,2.) - pow(C0,-2.);
	double root = abs( pow(gamma,2.) - gamma*b + c);
	double logargument = gamma / ( 2.*sqrt(c) * sqrt(root) - b*gamma + 2.*c);
	double result = z_0 * 1./sqrt((pow(n_ice,2.) * pow(C0,2.) - 1)) * log(logargument) + C1;
	return result;
}

double get_y_with_z_mirror(double n_ice, double delta_n, double z_0, double z, double C0, double C1=0.){
	//parameters
	//z: arrays of depths
	//c0: first parameter
	//c1: second parameter
	double c = pow(n_ice,2.) - pow(C0,-2.);
	double gamma_turn, z_turn;
	get_turning_point(c, gamma_turn, z_turn, n_ice, delta_n, z_0);
	if(z_turn >=0.){ //signal is reflected at surface
		z_turn=0.; //we've hit the surface
		gamma_turn=get_gamma(0.,n_ice, delta_n, z_0); //get gamma at the surface
	}
	double y_turn = get_y(gamma_turn,C0,C1,n_ice, delta_n, z_0);
	double result=0.;
	if(z < z_turn){
		double gamma = get_gamma(z,n_ice, delta_n, z_0);
		result=get_y(gamma,C0,C1,n_ice, delta_n, z_0);
	}
	else{
		double gamma = get_gamma(2*z_turn - z,n_ice, delta_n, z_0);
		result = 2*y_turn - get_y(gamma,C0,C1,n_ice, delta_n, z_0);
	}
	return result;
}

double get_y_turn(double C0, double x1[2], double n_ice, double delta_n, double z_0){
    // calculates the y-coordinate of the turning point. This is either the point of reflection off the ice surface
    // or the point where the saddle point of the ray (transition from upward to downward going)

    // Parameters
    // ----------
    // C_0: float
    //     C_0 parameter of function
    // x1: typle
    //   (y, z) start position of ray

	double c = n_ice * n_ice - 1./ (C0 * C0);
	double gamma_turn(0);
	double z_turn(0);
    get_turning_point(c, gamma_turn, z_turn, n_ice, delta_n, z_0);
    if(z_turn > 0){
        z_turn = 0;  // a reflection is just a turning point at z = 0, i.e. cases 2) and 3) are the same
        gamma_turn = get_gamma(z_turn, n_ice, delta_n, z_0);
    }
    double C1 = x1[0] - get_y_with_z_mirror(n_ice, delta_n, z_0, x1[1], C0);
    double y_turn = get_y(gamma_turn, C0, C1, n_ice, delta_n, z_0);
    return y_turn;
}

double get_C1(double pos[2], double C0, double n_ice, double delta_n, double z_0){
	//calculates C1 for a given C0 and starting point X1
	return pos[0] - get_y_with_z_mirror(n_ice, delta_n, z_0, pos[1],C0);
}

double get_c(double C0, double n_ice, double delta_n, double z_0){
	return pow(n_ice,2.)-pow(C0,-2.);
}

double get_C0_from_log(double logC0, double n_ice, double delta_n, double z_0){
	//transforms fit parameter C0 so that the likelihood looks better
	return exp(logC0) + 1./n_ice;
}

double get_z_unmirrored(double z, double C0, double n_ice, double delta_n, double z_0){
	//calculates the unmirrored z position
	double c = pow(n_ice,2.) - pow(C0,-2.);
	double gamma_turn, z_turn;
	get_turning_point(c, gamma_turn, z_turn,n_ice, delta_n, z_0);
	if(z_turn >=0.){ //signal is reflected at surface
		z_turn=0.; //we've hit the surface
		gamma_turn=get_gamma(0.,n_ice, delta_n, z_0); //get gamma at the surface
	}
	double z_unmirrored = z;
	if(z > z_turn) z_unmirrored = 2*z_turn - z;
	return z_unmirrored;
}

double get_y_diff(double z_raw, double C0, double n_ice, double delta_n, double z_0){
	//derivative dy(z)/dz
	double b = 2. * n_ice;
	double z = get_z_unmirrored(z_raw, C0,n_ice, delta_n, z_0);
	double c = pow(n_ice,2.) - pow(C0,-2.);
	double term1 =
			-sqrt(c) * exp(z/z_0) * b * delta_n
			+ 2.*sqrt(-b * delta_n * exp(z/z_0) + pow(delta_n,2.)*exp(2.*z/z_0) + c) * c
			+ 2.*pow(c,3./2.);
	double term2 =
			2. * sqrt(c) * sqrt(-b * delta_n * exp(z/z_0) + pow(delta_n,2.) * exp(2.*z/z_0) + c)
			-b*delta_n*exp(z/z_0)+
			+ 2.*c;
	double term3 =
			-b * delta_n * exp(z/z_0)
			+ pow(delta_n,2.) * exp(2.*z/z_0)
			+ c;
	double term4 = pow(n_ice,2.)*pow(C0,2.)-1;

	double res = term1 / term2 * 1./sqrt(term3) * 1./sqrt(term4);
	if(z != z_raw) res*=-1.;
	return res;
}

void get_z_mirrored(double pos[2], double pos2[2], double C0, double (&x2_mirrored)[2], double n_ice, double delta_n, double z_0){
	//calculates the mirrored x2 position so that y(z) can be used as a continuous function
	double c = pow(n_ice,2.) - pow(C0,-2.);
	double C1 = pos[0] - get_y_with_z_mirror(n_ice, delta_n, z_0, pos[1],C0);
	double gamma_turn, z_turn;
	get_turning_point(c, gamma_turn, z_turn,n_ice, delta_n, z_0);
	if(z_turn >=0.){ //signal is reflected at surface
		z_turn=0.; //we've hit the surface
		gamma_turn=get_gamma(0.,n_ice, delta_n, z_0); //get gamma at the surface
	}
	double y_turn = get_y(gamma_turn,C0,C1,n_ice, delta_n, z_0);
	double z_start = pos[1];
	double z_stop = pos2[1];
	if(y_turn <pos2[0]){
			z_stop = z_start + abs(z_turn - pos[1]) + abs(z_turn - pos2[1]);
	}
	x2_mirrored[0] = pos2[0];
	x2_mirrored[1] = z_stop;
}

//this function is explicity prepared for gsl integration in get_path_length
struct ds_params{ double a; double b; double c; double d;}; //a=C0, b=n_ice, c=delta_n, d = z_0
double ds (double t, void *p){
	//helper to calculate line integral
	struct ds_params *params = (struct ds_params *) p;
	double C0 = (params->a);
	double n_ice = (params->b);
	double delta_n = (params->a);
	double z_0 = (params->d);
	return sqrt((pow(get_y_diff(t,C0, n_ice, delta_n, z_0),2.)+1));
}

double get_path_length(double pos[2], double pos2[2], double C0, double n_ice, double delta_n, double z_0){
	double x2_mirrored[2]={0.};
	get_z_mirrored(pos,pos2,C0,x2_mirrored,n_ice, delta_n, z_0);

	gsl_integration_workspace *w = gsl_integration_workspace_alloc(1000);
	gsl_function F;
	F.function = &ds;
	struct ds_params params = {C0,n_ice, delta_n, z_0};
	F.params=&params;

	double result, error;
	double epsrel = 1.e-7; //small initial absolute error
	int max_badfunc_tries=7;
	int num_badfunc_tries=0;
	int status;

	/*
	This structuring allows for adaptive relative errors in the integral
	In many cases, the integral can be achieved with relative error between solutions of 1.e-7
	But there are several cases were the error bound needs to be as high as 30e-7
	So this raises the erorr bound by a factor of two, up to six times
	This means the largest relative error we're going to tolerate is 6.4e-6 (64e-7)
	If this relative error cannot be achieved, we will return an path length of zero
	*/
	gsl_error_handler_t *myhandler = gsl_set_error_handler_off(); //I want to handle my own errors (dangerous thing to do generally...)
	do{
		status = gsl_integration_qags(&F, pos[1], x2_mirrored[1],0,epsrel,1000,w,&result,&error);
		if(status!=GSL_SUCCESS){
			status=GSL_CONTINUE;
			num_badfunc_tries++;
			epsrel*=2.; //double the size of the relative error
		}
	}while(status == GSL_CONTINUE && num_badfunc_tries<max_badfunc_tries);
	gsl_set_error_handler (myhandler); //restore original error handler
	gsl_integration_workspace_free(w);
	double pathlength;
	if(status==GSL_SUCCESS){
		pathlength = result;
	}
	else{
		pathlength=NAN;
	}
	return pathlength;
}

//this function is explicitly prepared for gsl integration in get_travel_time
struct dt_params{ double a; double b; double c; double d;}; //a=C0, b=n_ice, c=delta_n, d = z_0
double dt (double t, void *p){
	struct dt_params *params = (struct dt_params *) p;
	double C0 = (params->a);
	double n_ice = (params->b);
	double delta_n = (params->c);
	double z_0 = (params->d);

	double z = get_z_unmirrored(t,C0,n_ice, delta_n, z_0);
	return sqrt((pow(get_y_diff(t,C0,n_ice, delta_n, z_0),2.)+1)) / speed_of_light * index_vs_depth(z, n_ice, delta_n, z_0);
}

double get_travel_time(double pos[2], double pos2[2], double C0, double n_ice, double delta_n, double z_0){
	double x2_mirrored[2]={0.};
	get_z_mirrored(pos,pos2,C0,x2_mirrored, n_ice, delta_n, z_0);

	gsl_integration_workspace *w = gsl_integration_workspace_alloc(1000);
	gsl_function F;
	F.function = &dt;
	struct dt_params params = {C0, n_ice, delta_n, z_0};
	F.params=&params;

	double result, error;

	gsl_integration_qags(&F, pos[1], x2_mirrored[1],0,10.e-7,1000,w,&result,&error);
	gsl_integration_workspace_free(w);
	return result;
}



//this function is explicitly prepared for gsl integration in get_attenuation_along_path
struct dt_freq_params{ double a; double c; double d; double e; double f; int model;}; //a=C0, c=freq, d=n_ice, e=delta_n, f=z_0
double dt_freq (double t, void *p){
	struct dt_freq_params *params = (struct dt_freq_params *)p;
	double C0 = (params->a);
	double freq = (params->c);
	double n_ice = (params->d);
	double delta_n = (params->e);
	double z_0 = (params->f);

	double z = get_z_unmirrored(t,C0,n_ice, delta_n, z_0);
	return sqrt((pow(get_y_diff(t,C0,n_ice, delta_n, z_0),2.)+1)) / get_attenuation_length(z,freq, params->model);
}

double get_attenuation_along_path(double pos[2], double pos2[2], double C0,
		double frequency, double n_ice, double delta_n, double z_0, int model){
	double x2_mirrored[2]={0.};
	get_z_mirrored(pos,pos2,C0,x2_mirrored, n_ice, delta_n, z_0);

	gsl_integration_workspace *w = gsl_integration_workspace_alloc(2000);
	gsl_function F;
	F.function = &dt_freq;
	struct dt_freq_params params = {C0,frequency, n_ice, delta_n, z_0, model};
	F.params=&params;

	double result, error;
	double epsrel = 1.e-7; //small initial absolute error
	int max_badfunc_tries=6;
	int num_badfunc_tries=0;
	int status;

	/*
	This structuring allows for adaptive relative errors in the integral
	In many cases, the integral can be achieved with relative error between solutions of 1.e-7
	But there are several cases were the error bound needs to be as high as 30e-7
	So this raises the erorr bound by a factor of two, up to six times
	This means the largest relative error we're going to tolerate is 6.4e-6 (64e-7)
	If this relative error cannot be achieved, we will return an attenuation of zero
	*/
	gsl_error_handler_t *myhandler = gsl_set_error_handler_off(); //I want to handle my own errors (dangerous thing to do generally...)
	do{
		status = gsl_integration_qags(&F, pos[1], x2_mirrored[1],0,epsrel,2000,w,&result,&error);
		if(status!=GSL_SUCCESS){
			status=GSL_CONTINUE;
			num_badfunc_tries++;
			epsrel*=2.; //double the size of the relative error
		}
	}while(status == GSL_CONTINUE && num_badfunc_tries<max_badfunc_tries);
	gsl_set_error_handler (myhandler); //restore original error handler
	gsl_integration_workspace_free(w);
	double attenuation;
	if(status==GSL_SUCCESS){
		attenuation = exp(-1 * result);
	}
	else{
		attenuation=NAN;
	}
	return attenuation;
}

double get_attenuation_along_path2(double pos_y, double pos_z, double pos2_y, double pos2_z,
		double C0, double frequency, double n_ice, double delta_n, double z_0, int model) {
	double pos[2] = {pos_y, pos_z};
	double pos2[2] = {pos2_y, pos2_z};
	return get_attenuation_along_path(pos, pos2, C0, frequency, n_ice, delta_n, z_0, model);
}

double get_angle(double x[2], double x_start[2], double C0, double n_ice, double delta_n, double z_0){
	double result[2]={0.};
	get_z_mirrored(x_start,x,C0,result, n_ice, delta_n, z_0);
	double z = result[1];
	double dy = get_y_diff(z,C0, n_ice, delta_n, z_0);
	double angle = atan(dy);
	if(angle < 0.) angle += pi;
	return angle;
}

double get_launch_angle(double x1[2], double C0, double n_ice, double delta_n, double z_0){
	return get_angle(x1,x1,C0, n_ice, delta_n, z_0);
}

double get_receive_angle(double x1[2], double x2[2], double C0, double n_ice, double delta_n, double z_0){
	return pi - get_angle(x2,x1,C0, n_ice, delta_n, z_0);
}

void get_reflection_point(double (&x2)[2], double C0, double C1, double n_ice, double delta_n, double z_0, double ice_reflection) {
    // calculates the point where the signal gets reflected off the bottom of the ice shelf
    //Returns tuple (y,z)
	double c = n_ice * n_ice - 1./ (C0 * C0);
	double gamma_turn(0);
	double z_turn(0);
	get_turning_point(c, gamma_turn, z_turn, n_ice, delta_n, z_0);
	if(z_turn > 0){
		z_turn = 0;  // a reflection is just a turning point at z = 0
	}
    x2[0] = 0.;
    x2[1] = ice_reflection;
    x2[0] = get_y_with_z_mirror(n_ice, delta_n, z_0, -x2[1] + 2 * z_turn, C0, C1);
}

double get_delta_y(double C0, double x1[2], double x2[2], double n_ice, double delta_n, double z_0,
					int reflection, int reflection_case, double ice_reflection){
	//calculates the difference in the y position between the analytic ray tracing path
	//specified by C0 at the position x2

	double lower_bound = 1./n_ice;
	double upper_bound = inf;
	if(C0<lower_bound || C0>upper_bound) {return inf;}
	double c = pow(n_ice,2.) - pow(C0,-2.);


	// we consider two cases here,
    // 1) the rays start rising -> the default case
    // 2) the rays start decreasing -> we need to find the position left of the start point that
    //    that has rising rays that go through the point x1
    if((reflection > 0) & (reflection_case == 2)) {
	   double y_turn = get_y_turn(C0, x1, n_ice, delta_n, z_0);
	   double dy = y_turn - x1[0];
	   x1[0] = x1[0] - 2 * dy;
    }

	for (int i = 0; i < reflection; ++i) {
		// we take account reflections at the bottom layer into account via
		// 1) calculating the point where the reflection happens
		// 2) starting a ray tracing from this new point

		// determine y translation first
		double C1 = x1[0] - get_y_with_z_mirror(n_ice, delta_n, z_0, x1[1], C0);
		double tmpx[2] = { 0. };
		get_reflection_point(tmpx, C0, C1, n_ice, delta_n, z_0, ice_reflection);
		x1[0] = tmpx[0];
		x1[1] = tmpx[1];
//		std::cout << "setting x1 to " << tmpx << ", " << x1[0] << ", " << x1[1]
//				<< std::endl;
	}

	//determine y translation
	double C1 = x1[0] - get_y_with_z_mirror( n_ice, delta_n, z_0, x1[1],C0);

	//for a given C0, 3 cases are possible to reach the position of x2
	//1: Direct ray--before the turning point
	//2: Refracted ray--after the turning point but not touching surface
	//3: Reflected ray--after the ray reaches the surface

	double gamma_turn, z_turn;
	get_turning_point(c, gamma_turn, z_turn, n_ice, delta_n, z_0);
	if(z_turn > 0.){
		z_turn = 0.; //a reflection is just a turning point at z=0, ie case 2 and 3 are the same
		gamma_turn = get_gamma(z_turn, n_ice, delta_n, z_0);
	}
	double y_turn = get_y(gamma_turn,C0,C1, n_ice, delta_n, z_0);
	if(z_turn < x2[1]){ //turning points is deeper than x2 positions, can't reach target
		// the minimizer has problems finding the minimum if inf is returned here. Therefore, we return the distance
		// between the turning point and the target point + 10 x the distance between the z position of the turning points
		// and the target position. This results in a objective function that has the solutions as the only minima and
		// is smooth in C_0
		return -(pow(pow(z_turn - x2[1], 2) + pow(y_turn - x2[0], 2), 0.5) + 10 * fabs(z_turn - x2[1]));
	}
	if(y_turn > x2[0]){//always propagate from left to right
		//direct ray
		double y2_fit = get_y(get_gamma(x2[1], n_ice, delta_n, z_0),C0,C1, n_ice, delta_n, z_0); //calculate the y position at get_path position
		double diff = (x2[0] - y2_fit);
		return diff;
	}
	else{
		//now it's a bit more complicated; we need to transform the coordinates to be on the mirrored part of the function
		double z_mirrored = x2[1];
		double gamma = get_gamma(z_mirrored, n_ice, delta_n, z_0);
		double y2_raw = get_y(gamma, C0, C1, n_ice, delta_n, z_0);
		double y2_fit = 2 * y_turn-y2_raw;
		double diff = x2[0] - y2_fit;
		return -1*diff;
	}
}

int determine_solution_type(double x1[2], double x2[2], double C0, double n_ice, double delta_n, double z_0){
	//return 1 for direct solution
	//return 2 for refracted
	//return 3 for reflected

	double c = pow(n_ice,2.) - pow(C0,-2.);
	double C1 = x1[0] - get_y_with_z_mirror(n_ice, delta_n, z_0, x1[1],C0);

	double gamma_turn, z_turn;
	get_turning_point(c, gamma_turn, z_turn, n_ice, delta_n, z_0);

	if(z_turn >= 0.){
		z_turn=0.;
		gamma_turn = get_gamma(0, n_ice, delta_n, z_0);
	}
	double y_turn = get_y(gamma_turn,C0,C1, n_ice, delta_n, z_0);
	if(x2[0] < y_turn) return 1; //direct
	else{
		if(abs(z_turn-0.0)<1e-6) return 3; // reflected, trying to do z_turn==0, but == is bad with doubles
		else return 2; //refracted
	}
}

//deprecated, but here for posterity
/*
double obj_delta_y_square(double logC0, double x1[2], double x2[2]){
	//objective function to find solution for C0
	double C0 = get_C0_from_log(logC0);
	return pow(get_delta_y(C0,x1,x2),2.);
}
*/

//this function is explicitly prepared for gsl root finding in find_solutions
struct obj_delta_y_square_params{double x1_x; double x1_z; double x2_x; double x2_z; double a; double b; double c; double d; int reflection; int reflection_case;}; //x1_x=x1[0] and so forth, a=n_ice, b=delta_n, c=z_0
double obj_delta_y_square(double logC0, void *p){
	struct obj_delta_y_square_params *params = (struct obj_delta_y_square_params *)p;
	double x1[2], x2[2];
	x1[0] = (params->x1_x);
	x1[1] = (params->x1_z);
	x2[0] = (params->x2_x);
	x2[1] = (params->x2_z);
	double n_ice = (params->a);
	double delta_n = (params->b);
	double z_0 = (params->c);
	double ice_reflection = (params->d);
	int reflection = (params->reflection);
	int reflection_case = (params->reflection_case);
	double C0 = get_C0_from_log(logC0, n_ice, delta_n, z_0);
	return pow(get_delta_y(C0,x1,x2, n_ice, delta_n, z_0, reflection, reflection_case, ice_reflection),2.);
}

//this function is explicity prepared for gsl root finding in find_solutions
//it is the "derivative"
//the derivative is f(x+h) - f(x) / h for small h
//so let's just hard code that
double obj_delta_y_square_df(double logC0, void *p){
	struct obj_delta_y_square_params *params = (struct obj_delta_y_square_params *)p;
	double x1[2], x2[2];
	x1[0] = (params->x1_x);
	x1[1] = (params->x1_z);
	x2[0] = (params->x2_x);
	x2[1] = (params->x2_z);
	double n_ice = (params->a);
	double delta_n = (params->b);
	double z_0 = (params->c);
	double ice_reflection = (params->d);
	int reflection = (params->reflection);
	int reflection_case = (params->reflection_case);
	double C0 = get_C0_from_log(logC0, n_ice, delta_n, z_0);
	double increment_size = C0/10000.;	//our small h
	return (pow(get_delta_y(C0+increment_size,x1,x2, n_ice, delta_n, z_0, reflection, reflection_case, ice_reflection),2.)-pow(get_delta_y(C0,x1,x2, n_ice, delta_n, z_0, reflection, reflection_case, ice_reflection),2.))/increment_size; //definition of derivative
}

//this function is explicity prepared for gsl root finding in find_solutions
//it is the "f * derivative"
void obj_delta_y_square_fdf(double logC0, void *p, double *y, double *dy){
	struct obj_delta_y_square_params *params = (struct obj_delta_y_square_params *)p;
	double x1[2], x2[2];
	x1[0] = (params->x1_x);
	x1[1] = (params->x1_z);
	x2[0] = (params->x2_x);
	x2[1] = (params->x2_z);
	double n_ice = (params->a);
	double delta_n = (params->b);
	double z_0 = (params->c);
	double ice_reflection = (params->d);
	int reflection = (params->reflection);
	int reflection_case = (params->reflection_case);
	double C0 = get_C0_from_log(logC0, n_ice, delta_n, z_0);
	double increment_size = C0/10000.;	//our small h
	*y = pow(get_delta_y(C0,x1,x2, n_ice, delta_n, z_0, reflection, reflection_case, ice_reflection),2.);
	*dy = (pow(get_delta_y(C0+increment_size,x1,x2, n_ice, delta_n, z_0, reflection, reflection_case, ice_reflection),2.)-pow(get_delta_y(C0,x1,x2, n_ice, delta_n, z_0, reflection, reflection_case, ice_reflection),2.))/increment_size; //definition of derivative
}
/*
double obj_delta_y(double logC0, double x1[2], double x2[2]){
	//function to find solution for C0, returns distance in y between function and x2 position
	//result is signed! (important to use root finder)
	double C0 = get_C0_from_log(logC0);
	return get_delta_y(C0,x1,x2);
}
*/


//this function is explicitly prepared for gsl root finding in find_solutions
double obj_delta_y(double logC0, void *p){
	struct obj_delta_y_square_params *params = (struct obj_delta_y_square_params *)p;
	double x1[2], x2[2];
	x1[0] = (params->x1_x);
	x1[1] = (params->x1_z);
	x2[0] = (params->x2_x);
	x2[1] = (params->x2_z);
	double n_ice = (params->a);
	double delta_n = (params->b);
	double z_0 = (params->c);
	double ice_reflection = (params->d);
	int reflection = (params->reflection);
	int reflection_case = (params->reflection_case);
	double C0 = get_C0_from_log(logC0, n_ice, delta_n, z_0);
	return get_delta_y(C0,x1,x2, n_ice, delta_n, z_0, reflection, reflection_case, ice_reflection);
}




vector <vector <double> > find_solutions(double x1[2], double x2[2], double n_ice, double delta_n,
										 double z_0, int reflection=0, int reflection_case=1, double ice_reflection=0.){
	//function finds all ray tracing solutions
	//we assume that x2 is above and to the right of x2_mirrored
	//this is perfectly general, as a coordinate transform can put any system in this configuration
	// printf("finding solution from %f %f to %f %f with %f %f %f", x1[0], x1[1], x2[0], x2[1], n_ice, delta_n, z_0);

	//returns a vector of vectors of the C0 solutions
	//entry 0 will be logC0
	//entry 1 will be CO
	//entry 2 will be C1
	//entry 3 will be type
	vector < vector <double> > results;

	struct obj_delta_y_square_params params = {x1[0],x1[1],x2[0],x2[1], n_ice, delta_n, z_0, ice_reflection,
											   reflection, reflection_case};


	/////////
	////Find root 1: a first solution; check around logC0=-1
	/////////

	int status;
	int iter=0, max_iter=200;
	double precision_fit = 1e-9;
	bool found_root_1=false;
	double root_1=-10000000; //some insane value we'd never believe
	int num_badfunc_tries = 0;
	int max_badfunc_tries = 100;

	const gsl_root_fdfsolver_type *Tfdf;
	gsl_root_fdfsolver *sfdf;
	gsl_function_fdf FDF;
	FDF.f = &obj_delta_y_square;
	FDF.df = &obj_delta_y_square_df;
	FDF.fdf = &obj_delta_y_square_fdf;
	FDF.params = &params;
	Tfdf = gsl_root_fdfsolver_secant;
	gsl_error_handler_t *myhandler = gsl_set_error_handler_off(); //I want to handle my own errors (dangerous thing to do generally...)
	
	// We have to guess at the location of the first root (if it it exists at all).
	// Because we might not guess correctly, or guess close enough,
	// it's in our favor (for numerical stability reasons) to try several times.
	// So, we start at 0, and walk back wards.
	// This issue on GitHub (https://github.com/nu-radio/NuRadioMC/issues/286)
	// revealed this case where only checking -1 didn't get us close enough
	// for the method (which is admittedly a *polishing* algorithm) to find the root.

	for (double x_guess_start = -1; x_guess_start>-3; x_guess_start-=1){
		if(found_root_1) break;
		double x_guess = x_guess_start;
		sfdf = gsl_root_fdfsolver_alloc(Tfdf);
		gsl_root_fdfsolver_set(sfdf,&FDF,x_guess);
		do{
			iter++;
			// cout<<"Got to iter "<<iter<<", guess is "<<x_guess<<" val is "<<GSL_FN_FDF_EVAL_F(&FDF,x_guess)<<endl;
			status = gsl_root_fdfsolver_iterate(sfdf);
			
			//we need to manually protect against the function blowing up, which *is an error*, but will casue GSL to fail
			//so, if we get a GSL_EBADFUNC, we want to manually say skip this, but re-enable the continue flag
			if(status==GSL_EBADFUNC) {status=GSL_CONTINUE; num_badfunc_tries++; continue;}
			
			root_1 = x_guess;
			x_guess = gsl_root_fdfsolver_root(sfdf);
			status = gsl_root_test_residual(GSL_FN_FDF_EVAL_F(&FDF,root_1),precision_fit);
			if(status == GSL_SUCCESS){
				// printf("Converged on root 1! Iteration %d\n",iter);
				// printf("minima =  %f\n",pow(get_delta_y(get_C0_from_log(root_1, n_ice, delta_n, z_0), x1, x2, n_ice, delta_n, z_0), 2));
				found_root_1=true;
			}
		} while (status == GSL_CONTINUE && iter < max_iter && num_badfunc_tries<max_badfunc_tries);
		gsl_root_fdfsolver_free (sfdf);		
		if(!found_root_1){ //reset
			num_badfunc_tries=0;
			iter=0;
			status = GSL_CONTINUE;
		}
	}
	gsl_set_error_handler (myhandler); //restore original error handler

	if(!found_root_1) {
		// printf("NOT converged on root 1! Iteration %d\n",iter);
	}

	if(found_root_1){
		vector <double> sol1;
		sol1.push_back(root_1);
		double C0 = get_C0_from_log(root_1, n_ice, delta_n, z_0);
		sol1.push_back(C0);
		sol1.push_back(get_C1(x1,C0, n_ice, delta_n, z_0));
		sol1.push_back(ceil(double(determine_solution_type(x1,x2,C0, n_ice, delta_n, z_0))));
		sol1.push_back(reflection);
		sol1.push_back(reflection_case);

		// printf("Solution 1 [logC0, C0, C1, type]: [%.4f, %.4f, %.4f, %f]]\n",sol1[0],sol1[1],sol1[2],sol1[3]);

		results.push_back(sol1);


		//reset this counter
		num_badfunc_tries = 0;

		/////////
		////Find root 2: a second solution
		/////////

		//now to check if another solution with higher logC0 exists
		//if the above algorithm failed, then we have to be more brute-forcy in the next go around
		double logC0_start;
		if(!found_root_1)  logC0_start=0.;
		else logC0_start = root_1+0.0001;

		gsl_function F;
		F.function = &obj_delta_y;
		F.params = &params;

		double logC0_stop = 100.;
		double delta_start = GSL_FN_EVAL(&F,logC0_start);
		double delta_stop = GSL_FN_EVAL(&F,logC0_stop);
		bool found_root_2 = false;
		double root_2 = -10000000;
		if(signbit(delta_start)!=signbit(delta_stop)){
			//printf("Solutions with logc0 > %.3f exist\n",root_1);

			//now we must solve again
			//let's use Brent's method, which should be faster
			//now that we know there is a y=0 (x-axis) crossing
			int status2;
			const gsl_root_fsolver_type *T;
			gsl_root_fsolver *s;

			T = gsl_root_fsolver_brent;
			s = gsl_root_fsolver_alloc(T);
			gsl_root_fsolver_set(s, &F, logC0_start, logC0_stop);
				gsl_error_handler_t *myhandler = gsl_set_error_handler_off(); //I want to handle my own errors (dangerous thing to do generally...)
				iter=0;
				do{
					iter++;
					status2 = gsl_root_fsolver_iterate(s);
					logC0_start = gsl_root_fsolver_x_lower(s);
					logC0_stop = gsl_root_fsolver_x_upper(s);
					//printf("[Iter, Xlo, Xhi]: [%d, %.8f, %.8f] \n",iter,logC0_start,logC0_stop);
					status2 = gsl_root_test_interval(logC0_start,logC0_stop, 0, precision_fit);
					if(status2==GSL_EBADFUNC) {status2=GSL_CONTINUE; num_badfunc_tries++; continue;}
					if(status2 == GSL_SUCCESS){
						// printf("Converged on root 2! Iteration %d\n",iter);
						found_root_2=true;
						root_2 = gsl_root_fsolver_root(s);
					}
				} while (status2 == GSL_CONTINUE && iter < max_iter && num_badfunc_tries<max_badfunc_tries);
				gsl_set_error_handler (myhandler); //restore original error handler
			gsl_root_fsolver_free (s);

			if(found_root_2){
				vector <double> sol2;
				sol2.push_back(root_2);
				double C0 = get_C0_from_log(root_2, n_ice, delta_n, z_0);
				sol2.push_back(C0);
				sol2.push_back(get_C1(x1,C0, n_ice, delta_n, z_0));
				sol2.push_back(ceil(double(determine_solution_type(x1,x2,C0, n_ice, delta_n, z_0))));
				sol2.push_back(reflection);
				sol2.push_back(reflection_case);

				// printf("Solution 2 [logC0, C0, C1, type]: [%.4f, %.4f, %.4f, %f]]\n",sol2[0],sol2[1],sol2[2],sol2[3]);

				results.push_back(sol2);
			}
		}
		// else printf("No solution with logc0 > %.3f exist\n",logC0_start);

		//reset this counter
		num_badfunc_tries = 0;

		/////////
		////Find root 3: a third solution
		/////////

		//now to check if another solution with lower logC0 exists
		//if the above algorithm failed, then we have to be more brute-forcy in the next go around

		if(!found_root_1)  logC0_stop=0.0001;
		else logC0_stop = root_1-0.0001;

		logC0_start = -100.;
		delta_start = GSL_FN_EVAL(&F,logC0_start);
		delta_stop = GSL_FN_EVAL(&F,logC0_stop);
		bool found_root_3 = false;
		double root_3 = -10000000;
		if(signbit(delta_start)!=signbit(delta_stop)){
			//printf("Solutions with logc0 > %.3f exist\n",root_1);

			//now we must solve again
			//let's use Brent's method, which should be faster
			//now that we know there is a y=0 (x-axis) crossing
			int status3;
			const gsl_root_fsolver_type *T;
			gsl_root_fsolver *s;
			gsl_function F;
			F.function = &obj_delta_y;
			F.params = &params;
			T = gsl_root_fsolver_brent;
			s = gsl_root_fsolver_alloc(T);
				gsl_root_fsolver_set(s, &F, logC0_start, logC0_stop);
				gsl_error_handler_t *myhandler = gsl_set_error_handler_off(); //I want to handle my own errors (dangerous thing to do generally...)
				iter=0;
				do{
					iter++;
					status3 = gsl_root_fsolver_iterate(s);
					logC0_start = gsl_root_fsolver_x_lower(s);
					logC0_stop = gsl_root_fsolver_x_upper(s);
					// printf("[Iter, Xlo, Xhi]: [%d, %.8f, %.8f] \n",iter,logC0_start,logC0_stop);
					status3 = gsl_root_test_interval(logC0_start,logC0_stop,0,precision_fit);
					if(status3==GSL_EBADFUNC) {status3=GSL_CONTINUE; num_badfunc_tries++; continue;}
					if(status3 == GSL_SUCCESS){
						// printf("Converged on root 3! Iteration %d\n",iter);
						found_root_3=true;
						root_3 = gsl_root_fsolver_root(s);
					}
				} while (status3 == GSL_CONTINUE && iter < max_iter && num_badfunc_tries<max_badfunc_tries);
				gsl_set_error_handler (myhandler); //restore original error handler
			gsl_root_fsolver_free (s);

			if(found_root_3){
				vector <double> sol3;
				sol3.push_back(root_3);
				double C0 = get_C0_from_log(root_3, n_ice, delta_n, z_0);
				sol3.push_back(C0);
				sol3.push_back(get_C1(x1,C0, n_ice, delta_n, z_0));
				sol3.push_back(ceil(double(determine_solution_type(x1,x2,C0, n_ice, delta_n, z_0))));
				sol3.push_back(reflection);
				sol3.push_back(reflection_case);

				// printf("Solution 3 [logC0, C0, C1, type]: [%.4f, %.4f, %.4f, %f]]\n",sol3[0],sol3[1],sol3[2],sol3[3]);

				results.push_back(sol3);
			}
		}
		// else printf("No solution with logc0 < %.3f exist\n",logC0_stop);
	}
	else{
		// printf("No solution exist anywhere!\n");
	}

	return results;
}

 void find_solutions2(double*& C0s, double*& C1s, int*& types, int& nSolutions, double y1, double z1, double y2,
		 double z2,  double n_ice, double delta_n, double z_0,
		 int reflection=0, int reflection_case=1, double ice_reflection=0.) {
	// clock_t begin = clock();
 	double x1[2] = {y1, z1};
 	double x2[2] = {y2, z2};
 	vector < vector<double> > solutions2 = find_solutions(x1, x2, n_ice, delta_n, z_0, reflection, reflection_case, ice_reflection);
 	nSolutions = solutions2.size();
 	C0s = new double[nSolutions];
 	C1s = new double[nSolutions];
 	types = new int[nSolutions];
 	for (int i = 0; i < nSolutions; ++i) {
 		C0s[i] = solutions2[i][1];
 		C1s[i] = solutions2[i][2];
 		types[i] = solutions2[i][3];
 	}
	// clock_t end = clock();
 	// double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
	// printf("%f (%d solutions)\n", 1000* elapsed_secs, nSolutions);
 }

void get_path(double n_ice, double delta_n, double z_0, double x1[2], double x2[2], double C0, vector<double> &res, vector<double> &zs, int n_points=100){

	//will return the ray tracing path between x1 and x2
	//this is only true if C0 is a solution to the ray tracing problem

	//parameters
	//x1: start position (y,z)
	//x2: stop position (y,z)
	//C0: first parameter
	//n_points: number of points to calcuate

	//returns two vectors by reference
	//res are y coordinates
	//zs are z coordinates


	double c = pow(n_ice,2.) - pow(C0,-.2);
	double C1 = x1[0] - get_y_with_z_mirror(n_ice, delta_n, z_0, x1[1],C0);
	double gamma_turn, z_turn;
	get_turning_point(c, gamma_turn, z_turn, n_ice, delta_n, z_0);
	if(z_turn >=0.){
		//signal reflects at surface
		z_turn=0.;
		gamma_turn = get_gamma(0, n_ice, delta_n, z_0);
	}
	double y_turn = get_y(gamma_turn, C0, C1, n_ice, delta_n, z_0);
	double zstart = x1[1];
	double result[2];
	get_z_mirrored(x1,x2,C0,result, n_ice, delta_n, z_0);
	double zstop = result[1];
	double step_size = (zstop-zstart)/double(n_points-1); //do n-1 so that the bounds are actually the bounds
	vector<double> z; //vector to hold z's
	for(int i=0; i<n_points; i++){
		z.push_back(zstart+i*step_size);
	}

	//c++ has no clever "masking" tools like python
	//so we have to do this the old fashioned way

	//some temporary stuff
	for(int i=0; i<n_points; i++){
		double gamma_temp;
		if(z[i]<z_turn){
			gamma_temp = get_gamma(z[i], n_ice, delta_n, z_0);
			res.push_back(get_y(gamma_temp,C0,C1, n_ice, delta_n, z_0));
			zs.push_back(z[i]);
		}
		else{
			gamma_temp = get_gamma(2 * z_turn - z[i], n_ice, delta_n, z_0);
			res.push_back(2*y_turn - get_y(gamma_temp,C0,C1, n_ice, delta_n, z_0));
			zs.push_back(2*z_turn - z[i]);
		}
	}
}

int main(int argc, char **argv){

	// first, set a source and transmitter location
	double x1[2] = {478., -149.}; // source in the x-z / y-z plane
	double x2[2] = {635., -5.}; //target inthe x-z/y-z plane; this particular target has both a direct and reflected ray solution

	// set some ice parameters
	double n_ice = 1.78;
	double delta_n = 0.427;
	double z_0 = 71. * utl::m; //meters

	// find solutions
	vector<vector<double> > solutions = find_solutions(x1,x2, n_ice, delta_n, z_0);

	/*
	  The thing stored in "solutions" are numerical factors that describe the ray tracing solution
	  in terms of some variables that are used to describe the ray tracing problem
	  they are not physically meaningful
	  to get the physically meaningful answers, we have to use the information in solutions and call functions
	  like "get_receive_angle", "get_travel_time" etc

	  The size of the first dimension of the vector tells you how many solutions there are
	  so, in a problem with two solutions, then the size will be two
	  solutions.size()=2

	  The first element is log(C0) parameter, second element is C0, third element is C1, last element is solution type
	  type 1 = "direct" solution
	  type 2 = "refracted" solution
	  type 3 = "reflected" solution

	  so solutions[0][3]=1 tells you you are looking at a "direct" ray solution
	 */

	if(solutions.size()>0){ // if there is a solution
		double C0 = solutions[0][1];
		int iceModel = 1;
		double att = get_attenuation_along_path(x1, x2,C0, 0.00390625*utl::GHz,n_ice, delta_n, z_0,iceModel);
		printf("Attenuation %.3f \n ", att);
		double receive_angle = get_receive_angle(x1, x2, C0, n_ice, delta_n, z_0);
		printf("Receive angle in radians %.3f \n", receive_angle);
		double launch_angle = get_launch_angle(x1,  C0, n_ice, delta_n, z_0);
		printf("Launch angle in radians %.3f \n", launch_angle);
	}

	return 0;

}
