from DatasetReader import DatasetReader
from collections import defaultdict
from Constants import Constants
import numpy as np
#from scipy.sparse import csc_matrix
from SongToVec import SongToVec
from TrainTestSetGen import TrainTestSetGen
from scipy.sparse import save_npz
from Evaluator import Evaluator
from transform_song_vectors import transform_song_vectors
from get_user_item_rating import generate_train_test_set_for_librec

# from UserNN import UserNN
import matplotlib.pyplot as plt

user_id_list = ['user_000002', 'user_000691', 'user_000345', 'user_000861', 'user_000774', 'user_000882', 'user_000577', 'user_000910', 'user_000031', 'user_000249', 'user_000149']

constants = Constants()

# read the lastfm-1K dataset
dataset_reader = DatasetReader()
# (user_db, song_db) = dataset_reader.read(constants.DATASET_LASTFM_1K);
# print ("User db len=",len(user_db), " Song db len=", len(song_db))

# load the pre-processed map files
(user_db, song_db) = dataset_reader.read(constants.MAPS_LASTFM_1K);
print ("User db len=",len(user_db), " Song db len=",len(song_db))

# for user_id, user_obj in user_db.items():
# 	print ("User id : ",user_id, " num songs : ", user_obj.get_num_unique_songs())



dataset_reader.get_ratings_matrix(user_db, song_db)

for user_id in user_id_list[1:2]:
	train_test_set_gen = TrainTestSetGen()
	train_test_set_gen.split_data_into_train_test_sets(user_db, song_db, user_id)

	dataset_reader.save_hop_distances(user_db, [user_id])


	# m = dataset_reader.get_transition_probabilities(user_db, song_db, user_id)
	# save_npz('../datasets/lastfm-dataset-1K/extracts/transition_probs_'+user_id, m)

	# # run the SongToVec model on combined song sequences for all users
	song_to_vec_comb = SongToVec()
	song_to_vec_comb.run(user_db, song_db, user_id, constants.RUN_SONG2VEC_ON_ALL_SONGS)

	transform_song_vectors(user_id, 'MMC')

	generate_train_test_set_for_librec(user_id)


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

# evaluator = Evaluator()
# evaluator.run()
