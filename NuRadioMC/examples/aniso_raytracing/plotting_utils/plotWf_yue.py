import numpy as np
import matplotlib.pyplot as plt

def plotWf(startBinDir, startBinRef, startBinDirTMP, startBinRefTMP, times, trace):
    dt = times[0][1] - times[0][0]
    fig, ax = plt.subplots(strNum, channelPerStr, figsize = (10, 5), sharex = True, sharey = True)
    fig.subplots_adjust(wspace = 0)
    #start = min(startBinDir)
    #end = max(startBinRef) + snipSize
    start = 0
    end = len(times[0])
    for i in range(strNum):
        ax[i][0].set_ylabel("amp(mV)")
    for j in range(strNum):
        ax[0][j].set_title("string {}".format(j))
        ax[channelPerStr - 1][j].set_xlabel("time(ns)")
    for i in range(channelPerStr):
        for j in range(strNum):
            chan = strNum * i + j
            ax[i][j].plot(times[chan][start:end] - channel.get_trace_start_time(), 1000. * trace[chan][start:end], linewidth = 0.5)
            ax[i][j].set_ylim([-0.2, 0.2])
            ax[i][j].plot(times[chan][start:end] - channel.get_trace_start_time(), 1000. * 2. * Vrms * np.ones(len(times[chan][start:end])), linewidth = 0.5)
            ax[i][j].plot(times[chan][start:end] - channel.get_trace_start_time(), 1000. * -2. * Vrms * np.ones(len(times[chan][start:end])), linewidth = 0.5)
            ax[i][j].add_patch(Rectangle((startBinDir[chan] * dt, -0.1), snipSize * dt, 0.2, edgecolor = "r", facecolor = "none", lw = 0.5, zorder = 10))
            ax[i][j].add_patch(Rectangle((startBinRef[chan] * dt, -0.1), snipSize * dt, 0.2, edgecolor = "r", facecolor = "none", lw = 0.5, zorder = 10))
            ax[i][j].add_patch(Rectangle((startBinDirTMP[chan] * dt, -0.1), snipSize * dt, 0.2, edgecolor = "b", facecolor = "none", lw = 0.5, zorder = 10))
            ax[i][j].add_patch(Rectangle((startBinRefTMP[chan] * dt, -0.1), snipSize * dt, 0.2, edgecolor = "b", facecolor = "none", lw = 0.5, zorder = 10))
    
    plt.tight_layout()

    '''
    if noiseMode == 1:
        if not os.path.exists("/data/user/ypan/bin/simulations/ARA02Recon/wf/{}/".format(hdf5)):
            os.makedirs("/data/user/ypan/bin/simulations/ARA02Recon/wf/{}/".format(hdf5))
        plt.savefig("/data/user/ypan/bin/simulations/ARA02Recon/wf/{}/ev{}.pdf".format(hdf5, evtId), bbox_inches = "tight")
    elif noiseMode in {2}:
        if not os.path.exists("/data/user/ypan/bin/simulations/ARA02Recon/wf/{}_nurNoise/".format(hdf5)):
            os.makedirs("/data/user/ypan/bin/simulations/ARA02Recon/wf/{}_nurNoise/".format(hdf5))
        plt.savefig("/data/user/ypan/bin/simulations/ARA02Recon/wf/{}_nurNoise/ev{}.pdf".format(hdf5, evtId), bbox_inches = "tight")
    '''
    
    plt.show()
    plt.clf()




