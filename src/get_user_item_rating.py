import numpy as np
import numpy.random
from song2vecMF import Song2vecMF
from scipy.sparse import load_npz, csr_matrix, coo_matrix

# output train data
def get_output_string(ratings, user, skip_user, cols_to_include=None):
    train_ratings = ratings.tocoo()
    data, row, col = train_ratings.data, train_ratings.row, train_ratings.col
    if skip_user:
        output  = '\n'.join([' '.join([str(x) for x in entry])
                             for entry in zip(row,col,data)
                             if entry[0] != user and entry[1] in cols_to_include])
    else:
        row += user
        output  = '\n'.join([' '.join([str(x) for x in entry]) for entry in zip(row,col,data)])
    return output

def generate_train_test_set_for_librec(user_id):

	ratings_mat = load_npz('../datasets/lastfm-dataset-1K/extracts/rating_mat.npz')
	test_songs_for_user2 = np.loadtxt('../datasets/lastfm-dataset-1K/extracts/test_songs_'+user_id,
	         dtype=np.int)

	train_songs_for_user2 = np.loadtxt('../datasets/lastfm-dataset-1K/extracts/train_songs_'+user_id, dtype=np.int, delimiter=',')
	user2_ratings = ratings_mat.getrow(1)

	user2_ratings = user2_ratings.toarray()[0] # convert to a normal array
	# print(user2_ratings.shape)
	train_ratings = np.copy(user2_ratings)
	# train_ratings = np.copy(ratings_mat.toa)
	non_train_idxs = np.array(list(set(range(user2_ratings.shape[0])) - set(train_songs_for_user2)))
	train_ratings[non_train_idxs] = 0
	test_ratings = np.copy(user2_ratings)
	non_test_idxs = np.array(list(set(range(user2_ratings.shape[0])) - set(test_songs_for_user2)))
	test_ratings[non_test_idxs] = 0

	output = get_output_string(csr_matrix(train_ratings), 2, False)
	output += '\n'
	user2_songs = set(train_songs_for_user2) | set(test_songs_for_user2)
	output += get_output_string(ratings_mat, 2, True, user2_songs)
	f = open('../datasets/lastfm-dataset-1K/extracts/trainset_'+user_id, 'w')
	f.write(output)
	f.close()

	# output test data
	test_ratings = csr_matrix(test_ratings).tocoo()
	data, row, col = test_ratings.data, test_ratings.row, test_ratings.col
	row +=2
	output  = '\n'.join([' '.join([str(x) for x in entry]) for entry in zip(row,col,data)])
	f = open('../datasets/lastfm-dataset-1K/extracts/testset_'+user_id, 'w')
	f.write(output)
	f.close()
