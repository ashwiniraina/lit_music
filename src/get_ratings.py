import numpy as np
from scipy.sparse import load_npz
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import euclidean_distances

def apply_mapping(mapping, song_ids):
    return [mapping[song_id] for song_id in song_ids]


def get_wrmf_predictions(user_id):
    n = 10
    q = np.loadtxt('../datasets/lastfm-dataset-1K/extracts/qdata_' + user_id + '.txt',
                   delimiter=',')
    p = np.loadtxt('../datasets/lastfm-dataset-1K/extracts/pdata_' + user_id + '.txt',
                   delimiter=',')
    train_songs = np.loadtxt('../datasets/lastfm-dataset-1K/extracts/train_songs_' + user_id,
                             dtype=np.int, delimiter=',')

    mapping = np.loadtxt('../datasets/lastfm-dataset-1K/extracts/wrmf_item_mapping_' +
                         user_id + '.txt', dtype=np.int)
    mapping = {orig_id:wrmf_id for orig_id,wrmf_id in mapping}
    inv_mapping = {wrmf_id:orig_id for orig_id,wrmf_id in mapping.items()}
    user_idx = int(user_id.split('_')[1]) - 1 # because ids start with 1
    p = p[user_idx,:]
    # print(p.shape)
    predicted_ratings = p.dot(q.T)
    # print(predicted_ratings.shape)
    train_wrmf_idxs = apply_mapping(mapping, train_songs)
    predicted_ratings[train_wrmf_idxs] = 0 # because indices start with 0

    topn_wrmf_idxs = (np.argsort(predicted_ratings)[::-1][:n])
    topn_predicted_idxs = apply_mapping(inv_mapping, topn_wrmf_idxs)
    topn_predicted = np.vstack([topn_predicted_idxs, predicted_ratings[topn_wrmf_idxs]]).T
    # print(topn_predicted)
    return (topn_predicted[:,0])
    if save_lists:
            np.save('../datasets/lastfm-dataset-1K/extracts/top10_wrmf_predictions_' + user_id,
                    wrmf_predicted)


def get_actual_predicted_songs(user_id, use_transformed_songs=True, save_lists=False):
    test_songs = set(np.loadtxt('../datasets/lastfm-dataset-1K/extracts/test_songs_' +
                                str(user_id),dtype=np.int))
    train_songs = np.loadtxt('../datasets/lastfm-dataset-1K/extracts/train_songs_' + user_id,
                             dtype=np.int, delimiter=',')

    if use_transformed_songs:
        x = np.load('../datasets/lastfm-dataset-1K/extracts/transformed_songs_vectors_' +
                    user_id + '.npy')
    else:
        x = np.loadtxt('../datasets/lastfm-dataset-1K/extracts/combined_song_vectors_' +
                       user_id)
        song_ids = x[:,0].astype(np.int)
        arg_sorted_ids = np.argsort(song_ids)
        x = x[arg_sorted_ids,1:] # sort songs by song id, and remove id column
        user_song_ids = list(sorted(set(train_songs) | test_songs))
        x = x[user_song_ids,:]


    # ml_sim_matrix = euclidean_distances(x)
    x_normalized = normalize(x)
    ml_sim_matrix = x_normalized.dot(x_normalized.T)
    ratings_mat = load_npz('../datasets/lastfm-dataset-1K/extracts/rating_mat.npz')

    n = 10
    mapping = np.load('../datasets/lastfm-dataset-1K/extracts/song_mapping_' + user_id + '.npy')
    mapping = {orig_id: user_id for orig_id,user_id in mapping}
    inv_mapping = {user_id: orig_id for orig_id,user_id in mapping.items()}
    user_idx = int(user_id.split('_')[1]) - 1 # because ids start with 1
    user_ratings = ratings_mat.getrow(user_idx).toarray()[0]

    test_ratings = []
    test_songs_np = np.array(list(test_songs))
    test_songs_idx = apply_mapping(mapping, test_songs_np)

    for row in range(ml_sim_matrix.shape[0]):
        if inv_mapping[row] in test_songs:
            ml_sim_matrix[row, test_songs_idx] = -float('inf')
            ml_sim_matrix[row, row] = -float('inf')
            # get topn most similar train songs to this one
            topn = np.argsort(ml_sim_matrix[row])[::-1][:n]
            simz = ml_sim_matrix[row, topn]
            # for ratings have to apply inv_mapping
            topn_ids = apply_mapping(inv_mapping, topn)
            # get ratings
            topn_ratings = user_ratings[topn_ids]
            # weighted sum
            rating = topn_ratings.dot(simz)
            # divide by sum of similarities
            scaled_rating = rating/(simz.sum())
            test_ratings.append((scaled_rating, inv_mapping[row]))

    topn_predicted = np.array([[song, rating]
                                for rating, song
                                in sorted(test_ratings, reverse=True)])[:n]
    if save_lists:
        if use_transformed_songs:
            np.save('../datasets/lastfm-dataset-1K/extracts/top10_predicted_ratings_' + user_id,
                    topn_predicted)
        else:
            np.save('../datasets/lastfm-dataset-1K/extracts/top10_s2v_ratings_' + user_id,
                    topn_predicted)
    # user_ratings = user_ratings
    user_ratings[train_songs] = 0 # because indices start with 0

    topn_actual_idxs = (np.argsort(user_ratings)[::-1][:n])
    # print(topn_actual_idxs)
    # print(user_ratings[topn_actual_idxs])
    topn_actual = np.vstack([topn_actual_idxs, user_ratings[topn_actual_idxs]]).T
    if save_lists:
        np.save('../datasets/lastfm-dataset-1K/extracts/top10_actual_ratings_' + user_id,
                topn_actual)
    return (topn_actual[:,0], topn_predicted[:,0])

def get_accuracy(user_ids, use_transformed_songs=False, use_wrmf=False):
    if not user_ids:
        user_ids = [str(i) for i in range(1,1001)]
        user_ids = ['user_' + '0' * (6 - len(user_id)) + user_id
                    for user_id in user_ids]
    precision = 0
    precision_wrmf = 0
    precisions, precisions_wrmf = [],[]
    for user_id in user_ids:
        actual, predicted = get_actual_predicted_songs(user_id, use_transformed_songs,
                                                       save_lists=False)
        if use_wrmf:
            wrmf_predicted = get_wrmf_predictions(user_id)
            tmp = len(set(actual) & set(wrmf_predicted))/(actual.shape[0])
            precision_wrmf += tmp
            precisions_wrmf.append(tmp)
        tmp = len(set(actual) & set(predicted))/(actual.shape[0])
        precision += tmp
        precisions.append(tmp)
        # print('----actual----')
        # print(actual)
        # print('----predicted----')
        # print(predicted)
        # print('----wrmf------')
        # print(wrmf_predicted)
    precision /= len(user_ids)
    print(precision)
    if use_wrmf:
        precision_wrmf /= len(user_ids)
        print(precision_wrmf)
    return (precision, precision_wrmf, precisions, precisions_wrmf)


if __name__ == '__main__':
    get_accuracy(['user_000002'])
    get_accuracy(['user_000002'], use_transformed_songs=True)
