from NuRadioReco.modules.base.module import setup_logger
logger = setup_logger("AriUtils_dicts")

# script containing dictionaries defined in snowshovel software
# copied from snowshovel/scripts/online/AriUtils.py

class macStnMap:
    unknownStn = 0
    unknownMac = "000000000000"
    stnToMac = {  
                  #formerly deployed
                  3:"0002F7F0C3B6",  4:"0002F7F0C41C",
                  6:"0002F7F0C445",  8:"0002F7F0C0F8",
                  10:"0002F7F0C61A", 11:"0002F7F175B7",
                  12:"0002F7F0C561", 
                  20:"0002F7F0AEE0", 31:"0002F7F1F634",
                  
                  # removed in season 2017/18
                  32:"0002F7F1F21A",
                  40:"0002F7F1E9ED",
                  
                  13:"0002F7F2244B",
                  14:"0002F7F20A9C", 15:"0002F7F1F7A8",
                  16:"0002F7F1E9ED", 17:"0002F7F202C1",
                  18:"0002F7F21A8A", 19:"0002F7F22444",
                  30:"0002F7F1F212", 
                  41:"0002F7F1F7C6",
                  
                  50:"0002F7F2E24B", 51:"0002F7F2E7B9",
                  52:"0002F7F2E1CE",
                  61:"0002F7F2EC55",
                  
                  1000:"0002AAAAAAAA",  # for testing only
                  1010:"0002BBBBBBBB"  # for testing only
    }
    macToStn = dict(zip(stnToMac.values(), stnToMac.keys()))

def getMacAdrFromStn(stnnum):
    try:
        return macStnMap.stnToMac[stnnum]
    except KeyError as e:
        logger.error("Unknown station number [{0}]".format(stnnum))
        return macStnMap.unknownMac

def getStnFromMacAdr(mac):
    logger.debug("getStnFromMacAdr({0})".format(mac))
    try:
        return macStnMap.macToStn[str(mac)]
    except KeyError as e:
        logger.error("Unknown mac address {0}".format(mac))
        return macStnMap.unknownStn



class macBoardMap:
    unknownStn = 0
    unknownMac = "000000000000"
    boardToMac = {101: "0002F7F1F7C6",
                  102: "0002F7F2244B",
                  104: "0002F7F20A9C",
                  105: "0002F7F22444",
                  108: "0002F7F1E9ED",
                  109: "0002F7F202C1",
                  110: "0002F7F1F7A8",
                  111: "0002F7F21A8A",
                  112: "0002F7F1F21A",
                  113: "0002F7F1F212",
                  201: "0002F7F1F634", 202: "0002F7F2E24B",
                  203: "0002F7F2E7B9", 204: "0002F7F2E1CE",
                  205: "0002F7F2DA83", 206: "0002F7F2F28A",
                  207: "0002F7F2EC55", 208: "0002F7F2EDFF"
                  }
    macToStn = dict(zip(boardToMac.values(), boardToMac.keys()))


def getMacAdrFromBoard(boardnum):
    try:
        return macBoardMap.boardToMac[boardnum]
    except KeyError as e:
        logger.error("Unknown board number [{0}]".format(boardnum))
        return macBoardMap.unknownMac
    
def getBoardFromMacAdr(mac):
    try:
        return macBoardMap.macToStn[mac]
    except KeyError as e:
        logger.error("Unknown mac address {0}".format(mac))
        return macBoardMap.unknownStn


