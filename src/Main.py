from DatasetReader import DatasetReader
from collections import defaultdict
from Constants import Constants
import numpy as np
#from scipy.sparse import csc_matrix
from SongToVec import SongToVec
from TrainTestSetGen import TrainTestSetGen
from scipy.sparse import save_npz
from Evaluator import Evaluator

# from UserNN import UserNN
import matplotlib.pyplot as plt

constants = Constants()

# read the lastfm-1K dataset
dataset_reader = DatasetReader()
# (user_db, song_db) = dataset_reader.read(constants.DATASET_LASTFM_1K);
# print ("User db len=",len(user_db), " Song db len=", len(song_db))

# load the pre-processed map files
# (user_db, song_db) = dataset_reader.read(constants.MAPS_LASTFM_1K);
# print ("User db len=",len(user_db), " Song db len=",len(song_db))

# dataset_reader.get_ratings_matrix(user_db, song_db)
# m = dataset_reader.get_transition_probabilities(user_db, song_db, "user_000002")
# save_npz('../datasets/lastfm-dataset-1K/extracts/transition_probs_user_000002', m)

# train_sessions = dataset_reader.get_user_train_sessions('user_000002')
# dataset_reader.save_hop_distances(user_db, ['user_000002'])

# get training sessions for users

# dataset_reader.save_hop_distances() -- uncomment

# for each user
  # generate combined song vectors -- combined_song_vectors_userid

# for each user
#   generate_bmf_files -- qdata_userid, bmf_item_mapping_userid

# for each user
#  transform_song_vectors(user_id)

# for each user
#  get_actual_predicted_songs(user_id):


# user_id = 'user_000002'
# user = user_db[user_id]
# sessions = user.play_sessions.sessions
# p = defaultdict(int)
# for session in sessions:
#     for e1,e2 in zip(session[:-1], session[1:]):
#         s1,s2 = e1[1], e2[1]
#         s1, s2 = s1.song_id_int, s2.song_id_int
#         p[s1, s2] += 1/user.songs[s1]
#         data,rows,cols = [],[],[]

#         for ((s1,s2),prob) in p.items():
#             data.append(prob)
#             rows.append(s1)
#             cols.append(s2)

# mat = coo_matrix((data, (rows,cols)))


# print (user_db['user_000002'].get_num_unique_songs())


# train_test_set_gen = TrainTestSetGen()
# train_test_set_gen.split_data_into_train_test_sets(user_db, song_db)

# run the SongToVec model on combined song sequences for all users
# song_to_vec_comb = SongToVec()
# song_to_vec_comb.run(user_db, song_db, constants.RUN_SONG2VEC_ON_ALL_SONGS)

# run the SongToVec model on individual user training song sequences
# song_to_vec_ind_training = SongToVec()
# song_to_vec_ind_training.run(user_db, song_db, constants.RUN_SONG2VEC_ON_USER_TRAINING_SONGS)

# run the SongToVec model on individual user training as well as test song sequences
# song_to_vec_ind_train_test = SongToVec()
# song_to_vec_ind_train_test.run(user_db, song_db, constants.RUN_SONG2VEC_ON_ALL_USER_SONGS)

# train the user DNN model
# userNN = UserNN()
# userNN.train_dnn('user_000002')

# EVALUATION

evaluator = Evaluator()
evaluator.run()
