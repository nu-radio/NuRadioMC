import argparse, numpy as np
from NuRadioReco.modules.io import eventReader
from NuRadioReco.framework.parameters import channelParametersRNOG as chp

def plot(plot_data, outpath, fs = 13, zoomfact = 1.1):

    num_bins = 10
    
    channels = [key for key in plot_data.keys() if isinstance(key, int)]
    num_channels = len(channels)
    
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from matplotlib.ticker import AutoMinorLocator

    fig = plt.figure(figsize = (6, 5), layout = "constrained")
    gs = GridSpec(num_channels, 2, figure = fig, width_ratios = [0.8, 0.2])

    for ind, channel in enumerate(channels):
        channel_ts = plot_data[channel]
        ts_absmax = np.max(np.abs(channel_ts))
        
        ax = fig.add_subplot(gs[ind, 0])
        ax_hist = fig.add_subplot(gs[ind, 1], sharey = ax)
        ax.set_ylim(-zoomfact * ts_absmax, zoomfact * ts_absmax)
        ax.set_xlim(min(plot_data["evt_num"]), max(plot_data["evt_num"]))
        ax_hist.set_xlim(0.0, 5 * len(channel_ts) / num_bins)

        ax.text(0.1, 0.9, f"CH {channel}", fontsize = fs, transform = ax.transAxes)
        ax.set_ylabel("Test stat.", fontsize = fs)
        
        ax.scatter(plot_data["evt_num"], channel_ts, s = 2)
        ax_hist.hist(channel_ts, histtype = "step", bins = num_bins, orientation = "horizontal")

        ax.tick_params(axis = "x", direction = "in", which = "both", bottom = True, top = True, labelbottom = ind == len(channels) - 1,
                       labelsize = fs)
        ax.tick_params(axis = "y", direction = "in", which = "both", left = True, right = True, labelsize = fs)

        ax.axhline(0.0, color = "gray", ls = "dashed", lw = 1.0)
        ax_hist.axhline(0.0, color = "gray", ls = "dashed", lw = 1.0)
        
        ax_hist.tick_params(axis = "x", direction = "in", which = "both", bottom = True, top = True, labelbottom = ind == len(channels) - 1,
                            labelsize = fs)
        ax_hist.tick_params(axis = "y", direction = "in", which = "both", left = True, right = True, labelsize = fs,
                            labelleft = False)         

    ax.set_xlabel("Event Number", fontsize = fs)
    ax_hist.set_xlabel("Evts. / bin", fontsize = fs)
        
    fig.savefig(outpath)
    plt.close()

def make_glitch_summary_plot(nur_path, plot_path, channels = [0, 1, 2, 3]):

    reader = eventReader.eventReader()
    reader.begin(nur_path)

    plot_data = {"evt_num": []}
    
    for evt in reader.run():
        station = evt.get_station()
        plot_data["evt_num"].append(evt.get_id())

        for channel in channels:
            ch_data = station.get_channel(channel)
            ch_number = ch_data.get_id()
            
            ts = ch_data.get_parameter(chp.glitch_test_statistic)

            if ch_number not in plot_data:
                plot_data[ch_number] = []
            plot_data[ch_number].append(ts)

    plot(plot_data, plot_path)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--nur", type = str, action = "store", dest = "nur_path")
    parser.add_argument("--plot", type = str, action = "store", dest = "plot_path")
    args = vars(parser.parse_args())

    make_glitch_summary_plot(**args)
