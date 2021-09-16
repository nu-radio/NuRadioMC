from NuRadioReco.detector import detector_mongo as det


def get_table(name):
    if(name == "surface_boards"):
        return det.db.surface_boards
    elif(name == "DRAB"):
        return det.db.DRAB
    elif(name == "IGLU"):
        return det.db.IGLU
    elif(name == "CABLE"):
        return det.db.CABLE
    elif(name == "VPol"):
        return det.db.VPol
    elif(name == "PULSER"):
        return det.db.PULSER
