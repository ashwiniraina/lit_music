from DatasetReader import DatasetReader
from Constants import Constants
import numpy as np
from scipy.sparse import csc_matrix


constants = Constants()

# read the lastfm-1K dataset
dataset_reader = DatasetReader()
(user_db, song_db) = dataset_reader.read(constants.DATASET_LASTFM_1K);
print ("User db len=",len(user_db), " Song db len=", len(song_db))

# load the pre-processed map files
#(user_db, song_db) = dataset_reader.read(constants.MAPS_LASTFM_1K);
#print ("User db len=",len(user_db), " Song db len=",len(song_db))

# dataset_reader.get_ratings_matrix(user_db, song_db)
