# Imports
import numpy as np
import matplotlib.pyplot as plt
import uproot
import glob, os

def GetData(path_to_data, station, channels):

    '''
        Function creating a dataset with traces
        
        Args: 
            path_to_data: Path to folder with .root files
            station: Which station to collect data from
            channels: Which channels to collect data from

        Outputs:
            Returns and saves .npy file with data
        '''

    # Save original directory
    owd = os.getcwd()

    # Change to the data directory
    os.chdir(path_to_data)

    # Create a empty np.array
    data = np.empty((0, 2048))

    # For each root-file in the directory
    for file in glob.glob("*.root"):

        # If it belongs to the station
        if file.startswith(f"forced_triggers_station{station}"):

            # Open file with uprott
            f = uproot.open(file)

            # Check if there is actually any data in the file
            if not "combined;1" in f.keys():
                print("I am here")
                continue

            # Fetch the force_triggered events
            trigger_flag = np.array(f["combined"]['header/trigger_info/trigger_info.force_trigger'], dtype=bool)
            forced_data = np.array(f["combined"]['waveforms/radiant_data[24][2048]'])[trigger_flag, :, :]
            # print(np.shape(forced_data))

            for channel in channels:
                # Fetch the data for the specific channel
                channel_data = forced_data[:,channel-1,:]
                # print(f"{channel-1}, data : {np.shape(channel_data)}")

                # Add to data array
                data = np.append(data, channel_data, axis=0)

            if np.shape(data)[0] > 15000:
                break
            print(np.shape(data)[0])
            
    
    # Go back to original directory
    os.chdir(owd)
    print(f"{np.shape(data)[0]} events fetched and saved to 'data.npy'")

    # Save data to file
    np.save('data',data)

    return data

if __name__ == "__main__":
    path_to_data = ""
    data = GetData(path_to_data,24,[13,16,19])
    plt.plot(data[23])
    plt.show()


