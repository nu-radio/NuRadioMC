# Imports
import mailbox
import numpy as np
import matplotlib.pyplot as plt
import uproot
import glob, os

path_to_data = "../../../../pnfs/ifh.de/acs/radio/diskonly/data/inbox/"

def GetWindyEvents(path_to_data, station, channel):

    # Save original directory
    owd = os.getcwd()

    # Change to the data directory
    os.chdir(path_to_data)

    # Create a empty np.array
    data = np.empty((0, 2048))

    # For each root-file in the directory
    for file in glob.glob("*.root"):

        # If it belongs to the station
        if file.startswith(f"run329"):

            # Open file with uprott
            f = uproot.open(file)

            # Check if there is actually any data in the file
            if not "combined;1" in f.keys():
                print("I am here")
                continue

            # Fetch the force_triggered events
            trigger_flag = np.array(f["combined"]['header/trigger_info/trigger_info.force_trigger'], dtype=bool)
            forced_data = np.array(f["combined"]['waveforms/radiant_data[24][2048]'])[trigger_flag, :, :]

            # Fetch the data for the specific channel
            channel_data = forced_data[:,channel-1,:]

            # Add to data array
            data = np.append(data, channel_data, axis=0)

            if np.shape(data)[0] > 5000:
                break
            print(np.shape(data))
    
    # Go back to original directory
    os.chdir(owd)
    print(f"{np.shape(data)[0]} events fetched and saved to 'data.npy'")

    # Save data to file
    np.save('data',data)

    return data

if __name__ == "__main__":
    path_to_data = "../shallman/data/rno_g/forced_triggers/inbox/"
    data = GetData(path_to_data,24,13)
    plt.plot(data[23])
    plt.show()


