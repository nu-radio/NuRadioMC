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
db = client.detector


def get_amp_board_names():
    """
    returns the unique names of all amplifier boards
    
    Returns list of strings
    """
    return db.amp_boards.distinct("name")


def insert_amp_board_channel_S12(board_name, channel_id, ff, mag, phase):
    """
    inserts a new S12 measurement of one channel of an amp board
    If the board dosn't exist yet, it will be created. 
    """
    db.amp_boards.update_one({'name': board_name},
                              {"$push" :{'channels': {
                                  'id': channel_id,
                                  'S12': {
                                      'frequencies': list(ff),
                                      'mag': list(mag),
                                      'phase': list(phase),
                                      }
                                  }}},
                             upsert=True)

