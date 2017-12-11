# compare with the un-transformed vectors as well.
from metric_learn import MMC, LSML, ITML, SDML
from scipy.sparse import load_npz
import numpy as np
from Constants import Constants
from collections import defaultdict
from heapq import heappush, heappop
import random

def get_song_pairs(idxs, song_pairs):
    mat = song_pairs[idxs]
    return (mat[:,0], mat[:,1])

def transform_song_vectors(user_id, method):
    # read user similarity matrix
    user_sim = np.load('../datasets/lastfm-dataset-1K/extracts/avg_hop_dist_' +
                       str(user_id) + ".npy").item()

    # file created using tr
    train_songs = set(np.loadtxt('../datasets/lastfm-dataset-1K/extracts/train_songs_' +
                            str(user_id), delimiter=',', dtype=np.int))

    test_songs = set(np.loadtxt('../datasets/lastfm-dataset-1K/extracts/test_songs_' +
                                str(user_id),dtype=np.int))
    user_song_ids = list(sorted(train_songs | test_songs))
    print(len(user_song_ids))
    mapping = {}
    inv_mapping = {}
    curr_id = 0
    for song_id in user_song_ids:
        mapping[song_id] = curr_id
        inv_mapping[curr_id] = song_id
        curr_id += 1

    def apply_mapping(song_ids):
        return [mapping[song_id] for song_id in song_ids]


    n = 5
    topn_heap = defaultdict(list)
    for ((s1,s2),dist) in user_sim.items():
        topnfors1 = topn_heap[s1]
        if s1 == s2:
            continue
        heappush(topnfors1, (-dist,s2)) # when we remove we want to remove the largest
        if len(topnfors1) > n:
            heappop(topnfors1)
        topn_heap[s1] = topnfors1

    topn = defaultdict(list)
    for s1 in topn_heap.keys():
        topn[s1] = [song for dist, song in sorted(topn_heap[s1], reverse=True)]

    a,b,c,d = [],[],[],[]
    for ((s1,s2), dist) in user_sim.items():
        if s2 in topn[s1]:
            a.append(mapping[s1])
            b.append(mapping[s2])
        else:
            c.append(mapping[s1])
            d.append(mapping[s2])

    # read song features
    song_features = np.loadtxt('../datasets/lastfm-dataset-1K/extracts/combined_song_vectors_' +
                               user_id)
    song_ids = song_features[:,0].astype(np.int)
    arg_sorted_ids = np.argsort(song_ids)
    song_features = song_features[arg_sorted_ids,1:] # sort songs by song id, and remove id column
    song_features = song_features[user_song_ids,:]
    W = np.ones((len(user_song_ids), len(user_song_ids))) * -1
    W[a,b] = 1
    # W[c,d] = 0
    c,d = np.where(W == -1)
    random.seed(42)
    rand_idx = random.sample(range(c.shape[0]), 500000)
    c,d = c[rand_idx], d[rand_idx]
    c,d = list(c), list(d)
    print(len(c),len(d),len(a),len(b))

    if 'MMC' in method:
        model = MMC(max_iter=60, verbose=True,
                    convergence_threshold=1e-5, diagonal=True, diagonal_c=1)
        # constraints = (np.array(a),np.array(b),np.array(c),np.array(d))
        # song_features = model.fit_transform(song_features, constraints)
    if 'SDML' in method:
        W = np.ones((len(user_song_ids), len(user_song_ids))) * -1
        W[a,b] = 1
        # W[c,d] = -1
        model = SDML(verbose=True, use_cov=False)

    if 'LSML' in method or 'ITML' in method:
        if 'LSML' in method:
            model = LSML(verbose=True)
        elif 'ITML' in method:
            model = ITML(verbose=True)
        min_len = min(len(a), len(c))
        print(min_len)
        a,b,c,d = a[:min_len], b[:min_len], c[:min_len], d[:min_len]
        print(np.array(a[:10]))

    if 'SDML' in method:
        transformed_songs = model.fit_transform(song_features, W)
    else:
        constraints = (np.array(a),np.array(b),np.array(c),np.array(d))
        transformed_songs = model.fit_transform(song_features, constraints)
    # np.save('song_features', song_features)
    np.save('../datasets/lastfm-dataset-1K/extracts/transformed_songs_vectors_' +
            str(user_id), transformed_songs)
    mapping_list = np.array([[orig_id,user_id] for orig_id, user_id in mapping.items()])
    np.save('../datasets/lastfm-dataset-1K/extracts/song_mapping_'+str(user_id), mapping_list)

if __name__ == "__main__":
    transform_song_vectors('user_000002', 'MMC')
