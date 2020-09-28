from NuRadioMC.SignalProp.analyticraytracing import ray_tracing as rana
from NuRadioMC.SignalProp.Simple_radiopropa_tracer import ray_tracing as rnum
import numpy as np
from NuRadioMC.utilities import medium
from matplotlib import pyplot as plt
from copy import deepcopy
import h5py
import radiotools.helper as hp
import pickle

ice=medium.greenland_simple()

events = h5py.File('input1000.hdf5', 'r')

pos_ant = np.array([0,0,-140])
pos_showers = np.transpose(np.append(np.append([events['xx']],[events['yy']],axis=0),[events['zz']],axis=0))
dir_showers = np.transpose(np.append([events['zeniths']],[events['azimuths']],axis=0))

number_of_showers = len(events['shower_ids'])

Para = ['x','y','z','length','time','receive','launch','refl','stype']
res = {para: np.zeros((number_of_showers,2),dtype=object) for para in Para}
res['launch'] = np.zeros((number_of_showers,2,2))
res['receive'] = np.zeros((number_of_showers,2,2))
res['results'] = [None for i in range(number_of_showers)]
RES={'num':deepcopy(res),'ana':deepcopy(res)}


solution_types = {1: 'direct',
                  2: 'refracted',
                  3: 'reflected'}


for i in range(number_of_showers):
	print(10*'#'+' shower '+ str(i+1)+' of '+str(len(events['shower_ids']))+' '+10*'#')
	for rt in ('ana','num'):
		if rt == 'ana': 
			tracer = rana(ice,shower_dir=dir_showers[i])
			#print('running analytical raytracer')
			time_unit = 1
		else: 
			tracer = rnum(ice,shower_dir=dir_showers[i])
			#tracer.set_cut_viewing_angle(180)
			#print('running numerical raytracer')
			time_unit = 10**9

		tracer.set_start_and_end_point(pos_showers[i],pos_ant)
		tracer.find_solutions()

		results = tracer.get_results()
		print(rt+' '+str(len(results)))
		RES[rt]['results'][i]=results
		
		for j in range(len(results)):
			path = np.array(tracer.get_path(j))
			RES[rt]['x'][i][j] = path[:,0]
			RES[rt]['y'][i][j] = path[:,1]
			RES[rt]['z'][i][j] = path[:,2]
			RES[rt]['length'][i][j]=tracer.get_path_length(j)
			RES[rt]['time'][i][j]=tracer.get_travel_time(j)*time_unit
			launch_cart = tracer.get_launch_vector(j)
			RES[rt]['launch'][i][j]=np.array(hp.cartesian_to_spherical(launch_cart[0],launch_cart[1],launch_cart[2]))
			receive_cart = -tracer.get_receive_vector(j)
			if rt == 'ana': RES[rt]['receive'][i][j]=np.array(hp.cartesian_to_spherical(receive_cart[0],receive_cart[1],receive_cart[2]))
			if rt == 'num': RES[rt]['receive'][i][j]=np.array(hp.cartesian_to_spherical(-receive_cart[0],-receive_cart[1],-receive_cart[2]))
			RES[rt]['refl'][i][j]=tracer.get_reflection_angle(j)
			RES[rt]['stype'][i][j]=results[j]['type']


pickle.dump(RES,open('output1000.pickle','wb'))


"""print(RES['ana']['launch'])
print(20*'-')
print(RES['ana']['launch'][0])
print(20*'-')
print(RES['ana']['launch'][0,0])
print(20*'-')
print(RES['ana']['launch'][0,0,0])"""
"""
fig_length = plt.figure(1)
plt.hist2d(RES['ana']['length'][:,0],RES['num']['length'][:,0],bins=(50,50),cmap=plt.cm.BuPu)
plt.title('path length')
plt.xlabel('analytical (m)')
plt.ylabel('numerical (m)')
l_min = min(min(RES['ana']['length'][:,0]),min(RES['num']['length'][:,0]))
l_max = max(max(RES['ana']['length'][:,0]),max(RES['num']['length'][:,0]))
l = np.linspace(l_min,l_max,num=100)
plt.plot(l,l,'--r')
plt.savefig('compare_raytracers_multi_length.png')

fig_length = plt.figure(2)
plt.hist2d(RES['ana']['time'][:,0],RES['num']['time'][:,0],bins=(50,50),cmap=plt.cm.BuPu)
plt.title('propagation time')
plt.xlabel('analytical (ns)')
plt.ylabel('numerical (ns)')
t_min = min(min(RES['ana']['time'][:,0]),min(RES['num']['time'][:,0]))
t_max = max(max(RES['ana']['time'][:,0]),max(RES['num']['time'][:,0]))
t = np.linspace(t_min,t_max,num=100)
plt.plot(t,1.5*t,'--r')
plt.plot(t,2*t,'--r')
plt.savefig('compare_raytracers_multi_time.png')

fig_length = plt.figure(3)
dzen = RES['num']['launch'][:,0,0]-RES['ana']['launch'][:,0,0]
daz = RES['num']['launch'][:,0,1]-RES['ana']['launch'][:,0,1]
plt.hist2d(dzen,daz,bins=(50,50),cmap=plt.cm.BuPu)
plt.title('difference launch vector (num-ana)')
plt.xlabel('zenith (rad)')
plt.ylabel('azimuth (rad)')
plt.axvline(x=0, color='red',linestyle='--')
plt.axhline(y=0, color='red',linestyle='--')
plt.savefig('compare_raytracers_multi_launch.png')

fig_length = plt.figure(4)
dzen = RES['num']['receive'][:,0,0]-RES['ana']['receive'][:,0,0]
daz = RES['num']['receive'][:,0,1]-RES['ana']['receive'][:,0,1]
plt.hist2d(dzen,daz,bins=(50,50),cmap=plt.cm.BuPu)
plt.title('difference receive vector (num-ana)')
plt.xlabel('zenith (rad)')
plt.ylabel('azimuth (rad)')
plt.axvline(x=0, color='red',linestyle='--')
plt.axhline(y=0, color='red',linestyle='--')
plt.savefig('compare_raytracers_multi_receive.png')
"""