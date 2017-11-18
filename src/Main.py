from DatasetReader import DatasetReader
from Constants import Constants
import numpy as np
#from scipy.sparse import csc_matrix
from SongToVec import SongToVec
from TrainTestSetGen import TrainTestSetGen
from UserNN import UserNN

constants = Constants()

# read the lastfm-1K dataset
dataset_reader = DatasetReader()
(user_db, song_db) = dataset_reader.read(constants.DATASET_LASTFM_1K);
print ("User db len=",len(user_db), " Song db len=", len(song_db))

# load the pre-processed map files
#(user_db, song_db) = dataset_reader.read(constants.MAPS_LASTFM_1K);
#print ("User db len=",len(user_db), " Song db len=",len(song_db))

dataset_reader.get_ratings_matrix(user_db, song_db)

# train_test_set_gen = TrainTestSetGen()
# train_test_set_gen.split_data_into_train_test_sets(user_db, song_db)

# run the SongToVec model on combined song sequences for all users
# song_to_vec_comb = SongToVec()
# song_to_vec_comb.run(user_db, song_db, constants.RUN_SONG2VEC_COMBINED)

# run the SongToVec model on individual user song sequences
#song_to_vec_ind = SongToVec()
#song_to_vec_ind.run(user_db, song_db, constants.RUN_SONG2VEC_INDIVIDUAL)

# train the user DNN model
# userNN = UserNN()
# userNN.train_dnn('user_000002')
