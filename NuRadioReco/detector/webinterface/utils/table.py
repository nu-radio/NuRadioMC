from NuRadioReco.detector import detector_mongo as det


def get_table(name):
    if(name == "surface_boards"):
        return det.db.surface_boards
    if(name == "DRAB"):
        return det.db.DRAB
