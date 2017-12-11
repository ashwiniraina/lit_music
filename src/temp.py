import numpy as np
import numpy.random
from song2vecMF import Song2vecMF
from scipy.sparse import load_npz, csr_matrix, coo_matrix, find

initialQ = np.load('../datasets/lastfm-dataset-1K/song_features.npy')
# change Q to just contain songs of user 2
ratings_mat = load_npz('../datasets/lastfm-dataset-1K/extracts/rating_mat.npz')
test_songs_for_user2 = np.loadtxt('../datasets/lastfm-dataset-1K/extracts/test_songs_user_000002',
         dtype=np.int)
user2_ratings = ratings_mat.getrow(1)
# rows,cols,ratings = find(user2_ratings) # pos, value of non-zero elements
# initialQ = initialQ[cols,:]
user2_sim = np.loadtxt('../datasets/lastfm-dataset-1K/extracts/ind_full_song_sim_matrix_user_000002', delimiter=',')
simMatrix = coo_matrix((user2_sim[:,2], (user2_sim[:,0], user2_sim[:,1])))
abc
# print(simMatrix.toarray())
# print(user2_ratings.nnz)
# print(user2_ratings.shape)
user2_ratings = user2_ratings.toarray() # convert to a normal array
# print(user2_ratings.shape)
user2_ratings[:,test_songs_for_user2] = 0
user2_ratings = csr_matrix(user2_ratings)
# print(user2_ratings.nnz)
num_items = initialQ.shape[0]
model = Song2vecMF(1, num_items, initialQ, user2_ratings, simMatrix, 10)
model.buildModel()
predicted_ratings = model.P.dot(model.Q.T)
np.save('../datasets/lastfm-dataset-1K/user2_ratings', predicted_ratings)
np.save('../datasets/lastfm-dataset-1K/user2_P', model.P)
np.save('../datasets/lastfm-dataset-1K/user2_Q', model.Q)
