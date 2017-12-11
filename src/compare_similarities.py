# CHANGE OTHERS TO USE EUCLIDEAN_DISTANCES. CHANGE ORDER
from scipy.sparse import coo_matrix
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
from scipy.sparse import load_npz
from collections import defaultdict
from heapq import heappush, heappop

def apply_mapping(mapping, song_ids):
    return [mapping[song_id] for song_id in song_ids]

def get_actual_predicted_songs(user_id):
    x = np.load('../datasets/lastfm-dataset-1K/extracts/transformed_songs_vectors_' +
                user_id + '.npy')
    # x_normalized = normalize(x)
    # ml_sim_matrix = x_normalized.dot(x_normalized.T)
    ml_sim_matrix = euclidean_distances(x)
    # np.save('../datasets/lastfm-dataset-1K/extracts/sim_matrix_metric_learning_' +
            # str(user_id), sim_matrix)


    # s2v_song_sims = np.loadtxt('../datasets/lastfm-dataset-1K/extracts/ind_full_song_sim_matrix_train_and_test_' + user_id, delimiter=',')
    # transition_probs = load_npz('../datasets/lastfm-dataset-1K/extracts/transition_probs_user_000002.npz').toarray()
    # s2v_song_sims[:,0] = apply_mapping(mapping, s2v_song_sims[:,0])
    # s2v_song_sims[:,1] = apply_mapping(mapping, s2v_song_sims[:,1])
    # s2v_sim_mat = coo_matrix((s2v_song_sims[:,2], (s2v_song_sims[:,0], s2v_song_sims[:,1])))
    n = 10
    mapping = np.load('../datasets/lastfm-dataset-1K/extracts/song_mapping' + user_id + '.npy')
    mapping = {orig_id: user_id for orig_id,user_id in mapping}
    inv_mapping = {user_id: orig_id for orig_id,user_id in mapping.items()}
    test_songs = set(np.loadtxt('../datasets/lastfm-dataset-1K/extracts/test_songs_' +
                                str(user_id),dtype=np.int))

    # save predicted topn songs for user
    predicted = defaultdict(list)
    for row in range(ml_sim_matrix.shape[0]):
        if inv_mapping[row] in test_songs:
            ml_sim_matrix[row][row] = float('inf')
            topn = np.argsort(ml_sim_matrix[row])[:n]
            predicted[user_id].append([inv_mapping[row]] + apply_mapping(inv_mapping, topn))

    np.save('../datasets/lastfm-dataset-1K/extracts/top10predictions_' + user_id, predicted)

    # save actual topn songs for user
    hop_dist = np.load('../datasets/lastfm-dataset-1K/extracts/avg_hop_dist_train_test_' +
                       user_id + '.npy').item()
    actual = defaultdict(list)
    topn = defaultdict(list)
    for ((s1,s2),dist) in hop_dist.items():
        topnfors1 = topn[s1]
        if s1 == s2:
            continue
        heappush(topnfors1, (-dist,s2)) # when we remove we want to remove the largest
        if len(topnfors1) > n:
            heappop(topnfors1)
        topn[s1] = topnfors1

    for row in range(ml_sim_matrix.shape[0]):
        if inv_mapping[row] in test_songs:
            s1 = inv_mapping[row]
            # because distances were multiplied by -1
            topnfors1 = [song for dist, song in sorted(topn[s1], reverse=True)]
            actual[user_id].append([inv_mapping[row]] + topnfors1)

    np.save('../datasets/lastfm-dataset-1K/extracts/top10actual_' + user_id, actual)


    # compare with bmf
    bmf_predicted = defaultdict(list)
    q = np.loadtxt('../datasets/lastfm-dataset-1K/extracts/qdata_' + user_id + '.txt',
                   delimiter=',')
    q_normalized = normalize(q)
    bmf_sim_mat = q_normalized.dot(q_normalized.T)
    np.save('../datasets/lastfm-dataset-1K/extracts/sim_matrix_bmf_' + str(user_id), bmf_sim_mat)
    bmf_mapping = np.loadtxt('../datasets/lastfm-dataset-1K/extracts/bmf_item_mapping_' +
                             user_id + '.txt', dtype=np.int)
    bmf_mapping = {orig_id:bmf_id for orig_id,bmf_id in bmf_mapping}
    bmf_inv_mapping = {bmf_id:orig_id for orig_id,bmf_id in bmf_mapping.items()}
    for row in range(ml_sim_matrix.shape[0]):
        orig_id = inv_mapping[row]
        if orig_id in test_songs:
            bmf_id = bmf_mapping[orig_id]
            bmf_sim_mat[bmf_id][bmf_id] = -float('inf')
            topn = np.argsort(bmf_sim_mat[bmf_id])[::-1][:n]
            bmf_predicted[user_id].append([orig_id] + apply_mapping(bmf_inv_mapping, topn))

    np.save('../datasets/lastfm-dataset-1K/extracts/top10_bmf_predictions_' + user_id,
            bmf_predicted)

    # compare with song2vec
    song_features = np.loadtxt('../datasets/lastfm-dataset-1K/extracts/combined_song_vectors_' +
                               user_id)
    train_songs = set(np.loadtxt('../datasets/lastfm-dataset-1K/extracts/train_songs_' +
                                 str(user_id), delimiter=',', dtype=np.int))
    test_songs = set(np.loadtxt('../datasets/lastfm-dataset-1K/extracts/test_songs_' +
                                str(user_id),dtype=np.int))
    user_song_ids = list(sorted(train_songs | test_songs))
    song_ids = song_features[:,0].astype(np.int)
    arg_sorted_ids = np.argsort(song_ids)
    song_features = song_features[arg_sorted_ids,1:] # sort songs by song id, and remove id column
    song_features = song_features[user_song_ids,:] # each row is a song
    s2v_normalized = normalize(song_features)
    s2v_sim_matrix = s2v_normalized.dot(s2v_normalized.T)
    s2v_predicted = defaultdict(list)
    for row in range(ml_sim_matrix.shape[0]):
        if inv_mapping[row] in test_songs:
            s2v_sim_matrix[row][row] = -float('inf')
            topn = np.argsort(s2v_sim_matrix[row])[::-1][:n]
            s2v_predicted[user_id].append([inv_mapping[row]] + apply_mapping(inv_mapping, topn))

    np.save('../datasets/lastfm-dataset-1K/extracts/top10_s2v_predictions_' + user_id, s2v_predicted)

if __name__ == '__main__':
    get_actual_predicted_songs('user_000002')
