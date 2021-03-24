import NuRadioReco.modules.io.araroot.readARAData
from NuRadioReco.modules.base.module import setup_logger
import matplotlib.pyplot as plt
from NuRadioReco.utilities import units

logger = setup_logger()

readARAData = NuRadioReco.modules.io.araroot.readARAData.readARAData()

n = readARAData.begin("/home/uzair/Documents/AraRootStuff/event9129.root")
print(n)

for iE, evt in enumerate(readARAData.run()):
    print(evt.get_id())
    channel = evt.get_station(2).get_channel(0)
    plt.figure()
    plt.plot(channel.get_times() / units.ns, channel.get_trace() / units.mV)
    plt.show()
    break
