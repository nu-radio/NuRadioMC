from NuRadioMC.SignalProp.analyticraytracing import ray_tracing as rana
from NuRadioMC.SignalProp.Simple_radiopropa_tracer import ray_tracing as rnum
import numpy as np
from NuRadioMC.utilities import medium
from matplotlib import pyplot as plt
from copy import deepcopy
import h5py
import radiotools.helper as hp
import pickle

RES = pickle.load(open('output1000.pickle','rb'))

stype_ana = RES['ana']['stype']
stype_num = RES['num']['stype']
shape = stype_ana.shape

num_of_sol_ana = np.count_nonzero(stype_ana,axis = 1)
num_of_sol_num = np.count_nonzero(stype_num,axis = 1)

#### MASKS STYPE ####
mask_stype_ana = (stype_ana > 0)
mask_stype_num = (stype_num > 0)
mask_stype1_ana = (stype_ana == 1)
mask_stype1_num = (stype_num == 1)
mask_stype2_ana = (stype_ana == 2)
mask_stype2_num = (stype_num == 2)
mask_stype3_ana = (stype_ana == 3)
mask_stype3_num = (stype_num == 3)

mask_AandN_stype = mask_stype_ana & mask_stype_num
mask_AorN_stype = mask_stype_ana | mask_stype_num


#### MASKS NUMBER OF SOLUTIONS ####
mask_has2sol_ana = (stype_ana[:,1] > 0)
mask_has2sol_num = (stype_num[:,1] > 0)
mask_has2sol = mask_has2sol_num & mask_has2sol_ana

mask_has1sol_ana = ((stype_ana[:,1] == 0) & (stype_ana[:,0] > 0))
mask_has1sol_num = ((stype_num[:,1] == 0) & (stype_num[:,0] > 0))
mask_has1sol = mask_has1sol_num & mask_has1sol_ana

mask_has0sol_ana = (stype_ana[:,0] == 0)
mask_has0sol_num = (stype_num[:,0] == 0)
mask_has0sol = mask_has0sol_num & mask_has0sol_ana


def plot_length_comparison():
	plt.clf()
	plt.hist2d(RES['ana']['length'][mask_AandN_stype], RES['num']['length'][mask_AandN_stype], bins=(50,50), cmap=plt.cm.viridis, cmin=1)
	plt.colorbar()
	plt.title('path length')
	plt.xlabel('analytical (m)')
	plt.ylabel('numerical (m)')
	l_min = min(min(RES['ana']['length'][:,0]), min(RES['num']['length'][:,0]))
	l_max = max(max(RES['ana']['length'][:,0]), max(RES['num']['length'][:,0]))
	l = np.linspace(l_min, l_max, num=100)
	plt.plot(l, l, '--r')
	plt.savefig('compare_raytracers_multi_plot_length.png')
	plt.clf()
	plt.hist(RES['ana']['length'][mask_AandN_stype] -RES['num']['length'][mask_AandN_stype])
	plt.xlabel('analytical - numerical [m]')
	plt.ylabel('number of events')
	plt.savefig('compoare_raytracers_mulit_plot_length_hist.png')

def plot_time_comparison():
	plt.clf()
	plt.hist2d(RES['ana']['time'][mask_AandN_stype], RES['num']['time'][mask_AandN_stype], bins=(50,50), cmap=plt.cm.viridis, cmin=1)
	plt.colorbar()
	plt.title('propagation time')
	plt.xlabel('analytical (ns)')
	plt.ylabel('numerical (ns)')
	t_min = min(min(RES['ana']['time'][:,0]), min(RES['num']['time'][:,0]))
	t_max = max(max(RES['ana']['time'][:,0]), max(RES['num']['time'][:,0]))
	t = np.linspace(t_min, t_max, num=100)
	plt.plot(t, t, '--r')
	plt.plot(t, 1.5*t, '--r')
	plt.plot(t, 2*t, '--r')
	plt.savefig('compare_raytracers_multi_plot_time.png')
	plt.clf()
	plt.hist(RES['ana']['time'][mask_AandN_stype] -RES['num']['time'][mask_AandN_stype])
	plt.xlabel('analytical - numerical [ns]')
	plt.ylabel('number of events')
	plt.savefig('compoare_raytracers_mulit_plot_time_hist.png')

def plot_launch_comparison():
	plt.clf()
	d = RES['num']['launch'][mask_AandN_stype] - RES['ana']['launch'][mask_AandN_stype]
	plt.hist2d(d[:,0] , d[:,1] , bins=(21,21) , range=[[-0.01,0.01],[-0.01,0.01]] , cmap=plt.cm.viridis , cmin=1)
	plt.colorbar()
	plt.title('difference launch vector (num-ana)')
	plt.xlabel('zenith (rad)')
	plt.ylabel('azimuth (rad)')
	plt.axvline(x=0 , color='red' , linestyle='--')
	plt.axhline(y=0 , color='red' , linestyle='--')
	plt.savefig('compare_raytracers_multi_plot_launch.png')

def plot_receive_comparison():
	plt.clf()
	d = RES['num']['receive'][mask_AandN_stype] - RES['ana']['receive'][mask_AandN_stype]
	plt.hist2d(d[:,0] , d[:,1] , bins=(21,21) , range=[[-.01,.01],[-.01,0.01]] , cmap=plt.cm.viridis , cmin=1)
	plt.colorbar()
	plt.title('difference receive vector (num-ana)')
	plt.xlabel('zenith (rad)')
	plt.ylabel('azimuth (rad)')
	plt.axvline(x=0, color='red', linestyle='--')
	plt.axhline(y=0, color='red', linestyle='--')
	plt.savefig('compare_raytracers_multi_plot_receive.png')

def plot_stype_comparison(number_of_solutions="all"):
	plt.clf()
	ticks = np.arange(0, 4, 1)
	bins = (4,4)
	ax_range = [[-0.5,3.5],[-0.5,3.5]]
	colorscale = plt.cm.viridis
	vmin = 1
	name_fig = 'compare_raytracers_multi_plot_stype'
	suptitle = 'solutions types' 

	subplots_1 = ["all","1","2","01"]

	if number_of_solutions in subplots_1:
		fig,ax = plt.subplots(1)
		if number_of_solutions == "all":
			x = stype_ana[np.logical_not(mask_has0sol)]
			vmax = len(x)
			x = stype_ana[mask_AorN_stype]
			y = stype_num[mask_AorN_stype]
		elif number_of_solutions == "2":
			x = stype_ana[mask_has2sol].flatten()
			y = stype_num[mask_has2sol].flatten()
			vmax = len(x)/2
			name_fig += "_2sol"
			suptitle += " (ana & num have 2 solutions)"
		elif number_of_solutions == "1":
			x = stype_ana[mask_has1sol][:,0]
			y = stype_num[mask_has1sol][:,0]
			name_fig += "_1sol"
			suptitle += " (ana & num have 1 solution)"
			vmax = len(x)
		elif number_of_solutions == "01":
			mask = (mask_has0sol_ana & mask_has1sol_num)
			x = stype_ana[mask][:,0]
			y = stype_num[mask][:,0]
			name_fig += "_01sol"
			suptitle += " (ana has no solutions & num has 1 solution)"
			vmax = len(x)
		im = ax.hist2d(x,y,bins=bins,range=ax_range,cmap=colorscale,cmin=vmin,vmin=vmin,vmax=vmax)
		axs = np.array([ax])

	else:
		if number_of_solutions == "21":
			fig,axs = plt.subplots(1,2)
			mask = (mask_has2sol_ana & mask_has1sol_num)
			test = stype_ana[stype_ana[:,0]==stype_ana[:,1]]
			print(test[test>0])
			x = [stype_ana[mask][:,0],stype_ana[mask][:,1]]
			y = [stype_num[mask][:,0],stype_num[mask][:,0]]
			titles = ['num to first ana solution','num to second ana solution']
			name_fig += "_21sol"
			suptitle += " (ana has 2 solutions & num has 1 solution)"
			vmax = len(x[0])
		
		for i in range(len(axs)):
			im = axs[i].hist2d(x[i],y[i],bins=bins,range=ax_range,cmap=colorscale,cmin=vmin,vmin=vmin,vmax=vmax)
			axs[i].set_title(titles[i])	


	fig.colorbar(im[3])
	fig.suptitle(suptitle)
	for ax in axs.flat:	
		ax.set_xlabel('analytical')
		ax.set_ylabel('numerical')
		plt.sca(ax)
		plt.xticks(ticks)
		plt.yticks(ticks)


	plt.savefig(name_fig)
	print('saved '+name_fig)

def plot_numOfSol_comparison():
	plt.clf()
	plt.hist2d(num_of_sol_ana,num_of_sol_num,bins=(3,3),range=[[-0.5,2.5],[-0.5,2.5]],cmap=plt.cm.viridis,cmin=1)
	plt.colorbar()
	plt.title('number of solutions')
	plt.xlabel('analytical')
	plt.ylabel('numerical')
	plt.xticks(np.arange(0, 3, 1))
	plt.yticks(np.arange(0, 3, 1))
	plt.savefig('compare_raytracers_multi_plot_numOfSol.png')


plot_length_comparison()
plot_time_comparison()
plot_launch_comparison()
plot_receive_comparison()
number_of_solutions = input('number of solution you want to plot for stype comparison [all/1/2/21/01]:')
plot_stype_comparison(number_of_solutions)
plot_numOfSol_comparison()
