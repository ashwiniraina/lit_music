# compare with the un-transformed vectors as well.
from metric_learn import MMC, MMC_Supervised
from scipy.sparse import load_npz
import numpy as np
from Constants import Constants

def get_song_pairs(idxs, song_pairs):
    mat = song_pairs[idxs]
    return (mat[:,0], mat[:,1])

def transform_song_vectors(user_id):
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

    a,b,c,d = [],[],[],[]
    for ((s1,s2), dist) in user_sim.items():
        if dist <= Constants.SIM_THRESHOLD_FOR_HOP_DIST:
            a.append(mapping[s1])
            b.append(mapping[s2])
        elif dist >= Constants.DISSIM_THRESHOLD_FOR_HOP_DIST:
            c.append(mapping[s1])
            d.append(mapping[s2])

    print(len(c),len(d),len(a),len(b))
    # read song features
    song_features = np.loadtxt('../datasets/lastfm-dataset-1K/extracts/combined_song_vectors_' +
                               user_id)
    song_ids = song_features[:,0].astype(np.int)
    arg_sorted_ids = np.argsort(song_ids)
    song_features = song_features[arg_sorted_ids,1:] # sort songs by song id, and remove id column
    song_features = song_features[user_song_ids,:]
    mmc = MMC(max_iter=10000)
    constraints = (np.array(a),np.array(b),np.array(c),np.array(d))
    transformed_songs = mmc.fit_transform(song_features, constraints)
    # np.save('song_features', song_features)
    np.save('../datasets/lastfm-dataset-1K/extracts/transformed_songs_vectors_' +
            str(user_id), transformed_songs)
    mapping_list = np.array([[orig_id,user_id] for orig_id, user_id in mapping.items()])
    np.save('../datasets/lastfm-dataset-1K/extracts/song_mapping_'+str(user_id), mapping_list)

if __name__ == "__main__":
    transform_song_vectors('user_000002')
