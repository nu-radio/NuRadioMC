from pymongo import MongoClient
import pymongo
import datetime
# pprint library is used to make the output look more pretty
from pprint import pprint
import logging

logging.basicConfig()
logger = logging.getLogger("database")
logger.setLevel(logging.DEBUG)

# connect to MongoDB, change the << MONGODB URL >> to reflect your own connection string
client = MongoClient("mongodb+srv://detector_write:detector_write@cluster0-fc0my.mongodb.net/test?retryWrites=true&w=majority")
# client = MongoClient("mongodb+srv://detector_write:detector_write@localhost/test?retryWrites=true&w=majority")
db = client.detector


def get_surface_board_names():
    """
    returns the unique names of all amplifier boards
    
    Returns list of strings
    """
    return db.surface_boards.distinct("name")

# def insert_amp_board_channel_S12(board_name, Sparameter, channel_id, ff, mag, phase):
#     """
#     inserts a new S12 measurement of one channel of an amp board
#     If the board dosn't exist yet, it will be created.
#     """
#     db.amp_boards.update_one({'name': board_name},
#                               {"$push" :{'channels': {
#                                   'id': channel_id,
#                                   'last_updated': datetime.datetime.utcnow(),
#                                   'S_parameter': Sparameter,
#                                   'frequencies': list(ff),
#                                   'mag': list(mag),
#                                   'phase': list(phase)
#                                   }}},
#                              upsert=True)


def surface_board_channel_set_not_working(board_name, channel_id):
    """
    inserts a new S parameter measurement of one channel of an amp board
    If the board dosn't exist yet, it will be created. 
    
    Parameters
    ---------
    board_name: string
        the unique identifier of the board
    channel_id: int
        the channel id
    S_data: array of floats
        1st collumn: frequencies
        2nd/3rd collumn: S11 mag/phase
        4th/5th collumn: S12 mag/phase
        6th/7th collumn: S21 mag/phase
        8th/9th collumn: S22 mag/phase
        
    """
    db.surface_boards.update_one({'name': board_name},
                                  {"$push" :{'channels': {
                                      'id': channel_id,
                                      'last_updated': datetime.datetime.utcnow(),
                                      'function_test': False,
                                      }}},
                                 upsert=True)


def surface_board_channel_add_Sparameters(board_name, channel_id, S_data):
    """
    inserts a new S parameter measurement of one channel of an amp board
    If the board dosn't exist yet, it will be created. 
    
    Parameters
    ---------
    board_name: string
        the unique identifier of the board
    channel_id: int
        the channel id
    S_data: array of floats
        1st collumn: frequencies
        2nd/3rd collumn: S11 mag/phase
        4th/5th collumn: S12 mag/phase
        6th/7th collumn: S21 mag/phase
        8th/9th collumn: S22 mag/phase
        
    """
    S_names = ["S11", "S12", "S21", "S22"]
    for i in range(4):
        db.surface_boards.update_one({'name': board_name},
                                  {"$push" :{'channels': {
                                      'id': channel_id,
                                      'last_updated': datetime.datetime.utcnow(),
                                      'function_test': True,
                                      'S_parameter': S_names[i],
                                      'frequencies': list(S_data[0]),
                                      'mag': list(S_data[2 * i + 1]),
                                      'phase': list(S_data[2 * i + 2])
                                      }}},
                                 upsert=True)

