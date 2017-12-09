from scipy.sparse import coo_matrix
from sklearn.preprocessing import normalize
import numpy as np
from scipy.sparse import load_npz
from collections import defaultdict
from heapq import heappush, heappop

def apply_mapping(mapping, song_ids):
    return [mapping[song_id] for song_id in song_ids]

def get_actual_predicted_songs(user_id):
    x = np.load('../datasets/lastfm-dataset-1K/extracts/transformed_songs_vectors_' + user_id + '.npy')
    x_normalized = normalize(x)
    sim_matrix = x_normalized.dot(x_normalized.T)
    np.save('../datasets/lastfm-dataset-1K/extracts/sim_matrix_metric_learning_' + str(user_id), sim_matrix)


    # s2v_song_sims = np.loadtxt('../datasets/lastfm-dataset-1K/extracts/ind_full_song_sim_matrix_train_and_test_' + user_id, delimiter=',')
    # transition_probs = load_npz('../datasets/lastfm-dataset-1K/extracts/transition_probs_user_000002.npz').toarray()
    # s2v_song_sims[:,0] = apply_mapping(mapping, s2v_song_sims[:,0])
    # s2v_song_sims[:,1] = apply_mapping(mapping, s2v_song_sims[:,1])
    # s2v_sim_mat = coo_matrix((s2v_song_sims[:,2], (s2v_song_sims[:,0], s2v_song_sims[:,1])))
    ml_sim_matrix = np.load('../datasets/lastfm-dataset-1K/extracts/sim_matrix_metric_learning_' + user_id + '.npy')
    mapping = np.load('../datasets/lastfm-dataset-1K/extracts/song_mapping' + user_id + '.npy')
    mapping = {orig_id: user_id for orig_id,user_id in mapping}
    inv_mapping = {user_id: orig_id for orig_id,user_id in mapping.items()}
    test_songs = set(np.loadtxt('../datasets/lastfm-dataset-1K/extracts/test_songs_' +
                                str(user_id),dtype=np.int))

    # save predicted top10 songs for user
    predicted = defaultdict(list)
    for row in range(ml_sim_matrix.shape[0]):
        if inv_mapping[row] in test_songs:
            # ml_sim_matrix[row][row] = -float('inf')
            top10 = np.argsort(ml_sim_matrix[row])[::-1][:10]
            predicted[user_id].append([inv_mapping[row]] + apply_mapping(inv_mapping, top10))

    np.save('../datasets/lastfm-dataset-1K/extracts/top10predictions_' + user_id, predicted)

    # save actual top10 songs for user
    hop_dist = np.load('../datasets/lastfm-dataset-1K/extracts/avg_hop_dist_train_test_' + user_id + '.npy').item()
    actual = defaultdict(list)
    top10 = defaultdict(list)
    for ((s1,s2),dist) in hop_dist.items():
        top10fors1 = top10[s1]
        heappush(top10fors1, (-dist,s2)) # when we remove we want to remove the largest
        if len(top10fors1) > 10:
            heappop(top10fors1)

    for row in range(ml_sim_matrix.shape[0]):
        if inv_mapping[row] in test_songs:
            s1 = inv_mapping[row]
            # because distances were multiplied by -1
            top10fors1 = [song for dist, song in sorted(top10[s1], reverse=True)]
            actual[user_id].append([inv_mapping[row]] + top10fors1)

    np.save('../datasets/lastfm-dataset-1K/extracts/top10actual_' + user_id, actual)


    # compare with baseline
    bmf_predicted = defaultdict(list)
    q = np.loadtxt('../datasets/lastfm-dataset-1K/extracts/qdata_' + user_id + '.txt', delimiter=',')
    q_normalized = normalize(q)
    bmf_sim_mat = q_normalized.dot(q_normalized.T)
    np.save('../datasets/lastfm-dataset-1K/extracts/sim_matrix_bmf_'+str(user_id), bmf_sim_mat)
    bmf_mapping = np.loadtxt('../datasets/lastfm-dataset-1K/extracts/bmf_item_mapping_' + user_id + '.txt', dtype=np.int)
    bmf_mapping = {orig_id:bmf_id for orig_id,bmf_id in bmf_mapping}
    bmf_inv_mapping = {bmf_id:orig_id for orig_id,bmf_id in bmf_mapping.items()}
    for row in range(ml_sim_matrix.shape[0]):
        orig_id = inv_mapping[row]
        if orig_id in test_songs:
            bmf_id = bmf_mapping[orig_id]
            bmf_sim_mat[bmf_id][bmf_id] = -float('inf')
            top10 = np.argsort(bmf_sim_mat[bmf_id])[::-1][:10]
            bmf_predicted[user_id].append([orig_id] + apply_mapping(bmf_inv_mapping, top10))

    np.save('../datasets/lastfm-dataset-1K/extracts/top10bmfpredictions_' + user_id, bmf_predicted)

if __name__ == '__main__':
    get_actual_predicted_songs('user_000002')
