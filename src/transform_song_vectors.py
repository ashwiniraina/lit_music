import numpy as np
from metric_learn import MMC, MMC_Supervised

def get_song_pairs(idxs):
    mat = song_pairs[idxs]
    return (mat[:,0], mat[:,1])

# read user similarity matrix
user_sim = np.loadtxt('ind_knn_song_sim_matrix_user_000002', delimiter=',', usecols=2)
# user_sim[i] is the similarity of the songs in song_pairs[i]
song_pairs = np.loadtxt('ind_knn_song_sim_matrix_user_000002',
                        delimiter=',', usecols=[0,1], dtype=np.int)


# songs that have a similarity value within the 10th percentile of
# similarity values are considered similar.
sim_cut_off = int(len(user_sim)/10)
sim_songs_idxs = np.argpartition(user_sim, sim_cut_off)
# a[i] and b[i] are similar songs
a,b = get_song_pairs(sim_songs_idxs[:sim_cut_off])

# songs with similarity values greater than the 90th percentile are
# considered dissimilar.
dissim_cut_off = len(user_sim) - int(len(user_sim)/10)
dissim_songs_idxs = np.argpartition(user_sim, dissim_cut_off)
# c[i] and d[i] are dissimilar songs.
c,d = get_song_pairs(dissim_songs_idxs[dissim_cut_off:])


# read song features
song_features = np.loadtxt('song_vectors')
song_ids = song_features[:,0].astype(np.int)
arg_sorted_ids = np.argsort(song_ids)
song_features = song_features[arg_sorted_ids,1:]
mmc = MMC(max_iter=1000)
constraints = (np.array(a),np.array(b),np.array(c),np.array(d))
transformed_songs = mmc.fit_transform(song_features, constraints)
# np.save('song_features', song_features)
np.save('transformed_songs_user_002', transformed_songs)
