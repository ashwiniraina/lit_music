from DatasetReader import DatasetReader
from Constants import Constants

constants = Constants()

# read the lastfm-1K dataset
dataset_reader = DatasetReader()
user_db, song_db = dataset_reader.read(constants.DATASET_LASTFM_1K);
print ("User db len=",len(user_db), " Song db len=", len(song_db))

# load the pre-processed map files
# user_db = dataset_reader.read(constants.MAPS_LASTFM_1K);
# print ("User db len=",len(user_db))