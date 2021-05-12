from aenum import Enum, auto
import uproot
import numpy as np

# layout to read the ARIANNA data for unknown formatting uproot does not guess right
ARIANNA_uproot_interpretation = {
        # TTimeStamp objects in root are 2 ints, first is time in sec since 01/01/1970, second one is nanoseconds
        # will return a jagged array of shape n_events x [t_s, t_ns]
        "time": uproot.interpretation.jagged.AsJagged(uproot.interpretation.numerical.AsDtype(
            [("fSec", ">i4"), ("fNanoSec", ">i4")]), header_bytes=6),

        # Interpretation of the trigger mask. See ARIANNA_TRIGGER for the definition of a mask to compare against.
        # No idea why the number of header_bytes in the root files is so odd.
        "trigger": uproot.interpretation.jagged.AsJagged(uproot.interpretation.numerical.AsDtype("uint8"), header_bytes=7),
        # the DAQConfig.fLPComWin / fComWin are complex objects consisting of ints/shors and a map, which is not readily read in by uproot.
        # An extremely inconvenient but working solution was to find and extract uint32 information
        # for fPer, fDur (period and duration) by hand from this. Before this information, there are 16 additional bytes, skipped here
        # as header_bytes, i.e. we ignore everything before that. All the info after the 2x4 bytes should be ignored later.
        "com_win": uproot.interpretation.jagged.AsJagged(uproot.interpretation.numerical.AsDtype(
            [("fPer", ">i4"), ("fDur", ">i4"), ("trash_0", 'uint16'), ("trash_1", 'uint32')]+
            [("trash_"+str(k), ">i8") for k in range(2,9+2)]), header_bytes=16)
        }


class EStdConfig(Enum):
      # order/values should not change to ensure proper reading of
      # old ReadoutConfig's written to files
      ATWD4ch = 0
      SST4ch = auto()
      SST8ch = auto()
      SST4ch1GHz = auto()
      SST4ch512 = auto()
      SST4ch512_1GHz = auto()
      Custom = auto()
      SST8ch1GHz = auto()

def getReadoutConfigTypeFromNum(num):
    dictionary = {k.value: k.name for k in EStdConfig}
    return dictionary[num]

def getNumFromReadoutConfigType(name):
    dictionary = {k.name: k.value for k in EStdConfig}
    return dictionary[name]


# arianna tigger map
ARIANNA_TRIGGER = {
        # Interpretation of the trigger mask.
        # - Bit 0: Thermal trigger
        # - Bit 1: Forced trigger
        # - Bit 2: External trigger (not used with current DAQ)
        # - Bit 3: L1 Trigger satisfied: the L1 trigger cuts away events with a large fraction of power in a single frequency.
        #          true = event PASSES, false = event would be cut by this L1 (may be in data via scaledown)
        # - Bit 4: 0 <not used>
        # - Bit 5: whether event is written thanks to L1 scaledown
        # - Bit 6: whether throwing away events based on L1 triggers
        # - Bit 7: flag events that took too long to get data from dCards to MB

        "thermal" : 2**0,
        "forced"  : 2**1,
        "external": 2**2,
        "l1":       2**3,
        # not_used: 2**4,
        "l1_scaledown":     2**5,
        "l1_enabled" :      2**6,
        "exceeding_buffer": 2**7
        }

class ariannaVoltageTreeParameters(Enum):
    time = 1
    V1_average = auto()
    V1_rms = auto()
    V2_average = auto()
    V2_rms = auto()

class ariannaTemperatureTreeParameters(Enum):
    time = 1
    temperature = auto()

class ariannaCalibTreeParameters(Enum):
    run_number = 1
    station_mac = auto()
    sequence_number = auto()
    time = auto()
    event_id = auto()
    trigger_mask = auto()
    DTms = auto()

    RawData_data = 101
    RawData_stop_bits = auto()
    RawData_CRC = auto()
    RawData_StationCRC = auto()

    PFNSubData_data = 201
    PFNSubData_error = auto()

    AmpOutData_data = 301
    AmpOutData_error = auto()

class ariannaConfigTreeParameters(Enum):
    run_number = 1
    station_mac = auto()
    sequence_number = auto()

    ReadoutConfig_Type = 101
    ReadoutConfig_Nchans = auto()
    ReadoutConfig_Nsamps = auto()
    ReadoutConfig_MaxPlas = auto()
    ReadoutConfig_SampDT = auto()

    DAQConfig_Label = 201
    DAQConfig_Usage = auto()
    DAQConfig_User = auto()
    DAQConfig_Desc = auto()
    DAQConfig_Built = auto()
    DAQConfig_Dacs = auto()
    DAQConfig_Plas = auto()
    DAQConfig_TrigSet = auto()
    DAQConfig_ComWin = auto()
    DAQConfig_LPComWin = auto()
    DAQConfig_RunMode = auto()
    DAQConfig_HrtBt = auto()
    DAQConfig_StreamHiLoPlas = auto()
    DAQConfig_WvLoseLSB = auto()
    DAQConfig_WvLoseMSB = auto()
    DAQConfig_WvBaseline = auto()
    DAQConfig_DatPackType = auto()
    DAQConfig_PowMode = auto()
    DAQConfig_BatVoltLowPwr = auto()
    DAQConfig_BatVoltFromLowPwr = auto()
    DAQConfig_VoltCheckPer = auto()
    DAQConfig_WchDogPer = auto()
    DAQConfig_TempCheckPer = auto()

    RunInfo_ConfLabel = 301
    RunInfo_StationLabel = auto()
    RunInfo_Run = auto()
    RunInfo_FirstSeq = auto()
    RunInfo_EvtsPerSeq = auto()

    TrigStartClock_PrevTime = 401
    TrigStartClock_SetTime = auto()
    TrigStartClock_CurrTime = auto()
    TrigStartClock_USsinceSet = auto()
    TrigStopClock_PrevTime = auto()
    TrigStopClock_SetTime = auto()
    TrigStopClock_CurrTime = auto()
    TrigStopClock_USsinceSet = auto() 

par_data = ariannaCalibTreeParameters
par_voltage = ariannaVoltageTreeParameters
par_temperature = ariannaTemperatureTreeParameters
par_config = ariannaConfigTreeParameters

arianna_voltage_dict = {
    # VoltageTree -> Contains power information
    par_voltage.V1_average : ['VoltageTree', 'PowerReading./PowerReading.faveV1', None],
    par_voltage.V2_average : ['VoltageTree', 'PowerReading./PowerReading.faveV2', None],
    par_voltage.V1_rms     : ['VoltageTree', 'PowerReading./PowerReading.frmsV1', None],
    par_voltage.V2_rms     : ['VoltageTree', 'PowerReading./PowerReading.frmsV2', None],
    par_voltage.time       : ['VoltageTree',  'PowerReading./PowerReading.fTime', ARIANNA_uproot_interpretation['time']],
}

arianna_temperature_dict = {
    #""" Dictionary for reading in ARIANNA root files with uproot
    #
    #Keys are keys of the arianna<treename>Parameters enum.
    #Values are a list of
    #    * top level tree name,
    #    * branch names, and 
    #    * None or custom interpretation of the branch if not interpreted  correctly by uproot
    #"""
    # TemperatureTree -> Contains temperature readings
    par_temperature.temperature : ['TemperatureTree', 'Temperature./Temperature.fTemp', None],
    par_temperature.time        : ['TemperatureTree', 'Temperature./Temperature.fTime', ARIANNA_uproot_interpretation['time']],
}

# HeartbeatTree -> not needed

arianna_config_dict = {
    #""" Dictionary for reading in ARIANNA root files with uproot
    #
    #Keys are keys of the arianna<treename>Parameters enum.
    #Values are a list of
    #    * top level tree name,
    #    * branch names, and 
    #    * None or custom interpretation of the branch if not interpreted  correctly by uproot
    #"""
    # ConfigTree
    par_config.run_number      : ['ConfigTree', 'ConfigMetadata./ConfigMetadata.fRun', None],
    par_config.station_mac     : ['ConfigTree', 'ConfigMetadata./ConfigMetadata.fStnId', None],
    par_config.sequence_number : ['ConfigTree', 'ConfigMetadata./ConfigMetadata.fSeq', None],

    par_config.ReadoutConfig_Type    : ['ConfigTree', 'ReadoutConfig./ReadoutConfig.fType', None],
    par_config.ReadoutConfig_Nchans  : ['ConfigTree', 'ReadoutConfig./ReadoutConfig.fNchans', None],
    par_config.ReadoutConfig_Nsamps  : ['ConfigTree', 'ReadoutConfig./ReadoutConfig.fNsamps', None],
    par_config.ReadoutConfig_MaxPlas : ['ConfigTree', 'ReadoutConfig./ReadoutConfig.fMaxPlas', None],
    par_config.ReadoutConfig_SampDT  : ['ConfigTree', 'ReadoutConfig./ReadoutConfig.fSampDT', None],

    par_config.DAQConfig_Label             : ['ConfigTree', 'DAQConfig./DAQConfig.fLabel', None],
    #par_config.DAQConfig_Usage            : ['ConfigTree', 'DAQConfig./DAQConfig.fUsage', None],
    #par_config.DAQConfig_User             : ['ConfigTree', 'DAQConfig./DAQConfig.fUser', None],
    #par_config.DAQConfig_Desc             : ['ConfigTree', 'DAQConfig./DAQConfig.fDesc', None],
    #par_config.DAQConfig_Built            : ['ConfigTree', 'DAQConfig./DAQConfig.fBuilt', None],
    par_config.DAQConfig_Dacs              : ['ConfigTree', 'DAQConfig./DAQConfig.fDacs', None],
    par_config.DAQConfig_Plas              : ['ConfigTree', 'DAQConfig./DAQConfig.fPlas', None],
    #par_config.DAQConfig_TrigSet           : ['ConfigTree', 'DAQConfig./DAQConfig.fTrigSet', !cannot be none],
    par_config.DAQConfig_ComWin            : ['ConfigTree', 'DAQConfig./DAQConfig.fComWin', ARIANNA_uproot_interpretation['com_win']],
    par_config.DAQConfig_LPComWin          : ['ConfigTree', 'DAQConfig./DAQConfig.fLPComWin', ARIANNA_uproot_interpretation['com_win']],
    par_config.DAQConfig_RunMode           : ['ConfigTree', 'DAQConfig./DAQConfig.fRunMode', None],
    #par_config.DAQConfig_HrtBt            : ['ConfigTree', 'DAQConfig./DAQConfig.fHrtBt', None],
    par_config.DAQConfig_StreamHiLoPlas    : ['ConfigTree', 'DAQConfig./DAQConfig.fStreamHiLoPlas', None],
    par_config.DAQConfig_WvLoseLSB         : ['ConfigTree', 'DAQConfig./DAQConfig.fWvLoseLSB', None],
    par_config.DAQConfig_WvLoseMSB         : ['ConfigTree', 'DAQConfig./DAQConfig.fWvLoseMSB', None],
    par_config.DAQConfig_WvBaseline        : ['ConfigTree', 'DAQConfig./DAQConfig.fWvBaseline', None],
    #par_config.DAQConfig_DatPackType      : ['ConfigTree', 'DAQConfig./DAQConfig.fDatPackType', None],
    par_config.DAQConfig_PowMode           : ['ConfigTree', 'DAQConfig./DAQConfig.fPowMode', None],
    par_config.DAQConfig_BatVoltLowPwr     : ['ConfigTree', 'DAQConfig./DAQConfig.fBatVoltLowPwr', None],
    par_config.DAQConfig_BatVoltFromLowPwr : ['ConfigTree', 'DAQConfig./DAQConfig.fBatVoltFromLowPwr', None],
    #par_config.DAQConfig_VoltCheckPer     : ['ConfigTree', 'DAQConfig./DAQConfig.fVoltCheckPer', None],
    #par_config.DAQConfig_WchDogPer        : ['ConfigTree', 'DAQConfig./DAQConfig.fWchDogPer', None],
    #par_config.DAQConfig_TempCheckPer     : ['ConfigTree', 'DAQConfig./DAQConfig.fTempCheckPer', None],

    # do not need run info for now. 
    #par_config.RunInfo_ConfLabel    : ['ConfigTree', 'RunInfo./RunInfo.fConfLabel', None],
    #par_config.RunInfo_StationLabel : ['ConfigTree', 'RunInfo./RunInfo.fStationLabel', None],
    #par_config.RunInfo_Run          : ['ConfigTree', 'RunInfo./RunInfo.fRun', None],
    #par_config.RunInfo_FirstSeq     : ['ConfigTree', 'RunInfo./RunInfo.fFirstSeq', None],
    #par_config.RunInfo_EvtsPerSeq   : ['ConfigTree', 'RunInfo./RunInfo.fEvtsPerSeq', None],

    par_config.TrigStartClock_CurrTime : ['ConfigTree', 'TrigStartClock./TrigStartClock.fCurrTime', ARIANNA_uproot_interpretation['time']],
    par_config.TrigStopClock_CurrTime  : ['ConfigTree', 'TrigStopClock./TrigStopClock.fCurrTime', ARIANNA_uproot_interpretation['time']]
}


arianna_calib_dict = {
    #""" Dictionary for reading in ARIANNA root files with uproot
    # 
    #Keys are keys of the arianna<treename>Parameters enum.
    #Values are a list of
    #    * top level tree name,
    #    * branch names, and 
    #    * None or custom interpretation of the branch if not interpreted  correctly by uproot
    #"""
    # CalibTree -> Contains event and waveform data
    par_data.run_number        : ['CalibTree', 'EventMetadata./EventMetadata.fRun', None],
    par_data.station_mac       : ['CalibTree', 'EventMetadata./EventMetadata.fStnId', None],
    par_data.sequence_number   : ['CalibTree', 'EventMetadata./EventMetadata.fSeq', None],
    par_data.time              : ['CalibTree', 'EventHeader./EventHeader.fTime', ARIANNA_uproot_interpretation['time']],
    par_data.event_id          : ['CalibTree', 'EventHeader./EventHeader.fNum', None],
    par_data.trigger_mask      : ['CalibTree', 'EventHeader./EventHeader.fTrgInfo', ARIANNA_uproot_interpretation['trigger']],
    par_data.DTms              : ['CalibTree', 'EventHeader./EventHeader.fDTms', None],
                   
    par_data.RawData_data      : ['CalibTree', 'RawData./RawData.fData', None],
    par_data.RawData_stop_bits : ['CalibTree', 'RawData./RawData.fStop', None],
    par_data.RawData_CRC       : ['CalibTree', 'RawData./RawData.fCRC', None],
    par_data.RawData_StationCRC: ['CalibTree', 'RawData./RawData.fStationCRC', None],   
    par_data.PFNSubData_data   : ['CalibTree', 'FPNSubData./FPNSubData.fData', None],
    par_data.PFNSubData_error  : ['CalibTree', 'FPNSubData./FPNSubData.fError', None],
    par_data.AmpOutData_data   : ['CalibTree', 'AmpOutData./AmpOutData.fData', None],
    par_data.AmpOutData_error  : ['CalibTree', 'AmpOutData./AmpOutData.fError', None]
}
