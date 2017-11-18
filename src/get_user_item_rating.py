import numpy as np
import numpy.random
from song2vecMF import Song2vecMF
from scipy.sparse import load_npz, csr_matrix, coo_matrix

ratings_mat = load_npz('../datasets/lastfm-dataset-1K/extracts/rating_mat.npz')
test_songs_for_user2 = np.loadtxt('../datasets/lastfm-dataset-1K/extracts/test_songs_user_000002',
         dtype=np.int)
user2_ratings = ratings_mat.getrow(1)

user2_ratings = user2_ratings.toarray()[0] # convert to a normal array
# print(user2_ratings.shape)
train_ratings = np.copy(user2_ratings)
train_ratings[test_songs_for_user2] = 0
test_ratings = user2_ratings[test_songs_for_user2]

# output train data
data,row,col= coo_matrix(train_ratings).tocoo()
output  = '\n'.join([' '.join(entry) for entry in zip(data,row,col)])
f = open('../datasets/lastfm-dataset-1K/extracts/trainset_user_000002', 'w')
f.write(output)
f.close()

# output train data
data,row,col= coo_matrix(test_ratings).tocoo()
output  = '\n'.join([' '.join(entry) for entry in zip(data,row,col)])
f = open('../datasets/lastfm-dataset-1K/extracts/testset_user_000002', 'w')
f.write(output)
f.close()
