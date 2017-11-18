
from gensim.models import word2vec
import logging
import glob
import numpy as np
from scipy import spatial
from sklearn.neighbors import NearestNeighbors
from Constants import Constants
from metric_learn import MMC, MMC_Supervised
import matplotlib.pyplot as plt

class SongToVec:

	constants = Constants()
	song_vectors = {} # map of song_id_int to song vector
	song_vectors_array = []
	song_vectors_int_ids = []
	#song_similarity_matrix = np.empty((0,0))

	def __init__(self):
		logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

	def read_combined_sessions(self):
		sessions = []
		# read the user play session files
		play_session_files = glob.glob("../datasets/lastfm-dataset-1K/extracts/play_session_*")
		for play_session_file in play_session_files:
			with open(play_session_file, 'r') as ps_file:
				for session in ps_file:
					session_song_id_ints = session.strip("\n").split(",")[1::2] # select every odd element from the list
					sessions.append(session_song_id_ints)
					for song_id_int in session_song_id_ints:
						if song_id_int not in self.song_vectors:
							self.song_vectors[song_id_int] = [] # empty list, will be filled with song vector in generate_song_vectors

		return sessions

	def read_individual_sessions(self, user_id):
		sessions = []
		# read the user play session files
		with open("../datasets/lastfm-dataset-1K/extracts/training_sessions_"+str(user_id), 'r') as ps_file:
			for session in ps_file:
				session_song_id_ints = session.strip("\n").split(",")[:-1] # last element is empty
				sessions.append(session_song_id_ints)
				for song_id_int in session_song_id_ints:
					if song_id_int not in self.song_vectors:
						self.song_vectors[song_id_int] = [] # empty list, will be filled with song vector in generate_song_vectors

		return sessions		

	def run(self, user_db, song_db, mode):

		if mode == self.constants.RUN_SONG2VEC_COMBINED:
			sessions = self.read_combined_sessions()
			model = self.run_word2vec_model(sessions)
			self.generate_song_vectors(model, 'combined_song_vectors')
			self.find_knn_for_song_vectors(self.constants.NUM_NEAREST_NEIGHBORS, 'combined_knn_song_sim_matrix')
		else:
			for user_id in user_db:
				if user_id == 'user_000002':
					#sessions = self.read_individual_sessions(user_id)
					#model = self.run_word2vec_model(sessions)
					#self.generate_song_vectors(model, 'ind_song_vectors_'+str(user_id))
					#self.generate_full_similarity_matrix('ind_full_song_sim_matrix_'+str(user_id))
					#self.find_knn_for_song_vectors(self.constants.NUM_NEAREST_NEIGHBORS, 'ind_knn_song_sim_matrix_'+str(user_id))
					self.transform_song_vectors(user_id)

	def run_word2vec_model(self, sessions):

		# train the skip-gram model; default window=5
		model = word2vec.Word2Vec(sessions, size=10, window=5, min_count=1, workers=4)
		 
		# # pickle the entire model to disk, so we can load&resume training later
		# model.save('../datasets/lastfm-dataset-1K/extracts/song2vec.model')
		# # store the learned weights, in a format the original C tool understands
		# model.save_word2vec_format('../datasets/lastfm-dataset-1K/extracts/song2vec.model.bin', binary=True)
		# # or, import word weights created by the (faster) C word2vec
		# # this way, you can switch between the C/Python toolkits easily
		# model = word2vec.Word2Vec.load_word2vec_format('../datasets/lastfm-dataset-1K/extracts/song2vec.model.bin', binary=True)

		return model
		
	def generate_song_vectors(self, model, filename):
		with open('../datasets/lastfm-dataset-1K/extracts/'+filename, 'w') as song_vectors_file:
			for song_id_int in self.song_vectors:
				self.song_vectors[song_id_int] = model.wv[song_id_int]
				self.song_vectors_array.append(self.song_vectors[song_id_int])
				self.song_vectors_int_ids.append(song_id_int)
				#print ("song :",song_id_int," vector: ",self.song_vectors[song_id_int])
				song_vectors_file.write(str(song_id_int)+","+str(self.song_vectors[song_id_int])+"\n")

	def find_knn_for_song_vectors(self, k, filename):
		nbrs = NearestNeighbors(k, algorithm='auto').fit(self.song_vectors_array)
		distance, indices = nbrs.kneighbors(self.song_vectors_array)

		with open('../datasets/lastfm-dataset-1K/extracts/'+filename, 'w') as sim_matrix_file:
			for i in range(0,len(self.song_vectors_int_ids)):
				song_i = self.song_vectors_int_ids[i]
				#print (song_i)
				nbrs_i = indices[i,:]
				#print (nbrs_i)
				dist_i = distance[i,:]
				#print (dist_i)
				for j in range(1,len(nbrs_i)): # 0 is the song_i itself
					song_j = self.song_vectors_int_ids[nbrs_i[j]]
					dist_i_j = dist_i[j] 
					sim_matrix_file.write(str(song_i)+":"+str(song_j)+","+str(dist_i_j)+"\n")

	def generate_full_similarity_matrix(self, filename):
		#self.song_similarity_matrix = np.zeros((len(self.song_vectors), len(self.song_vectors)))

		with open('../datasets/lastfm-dataset-1K/extracts/'+filename, 'w') as sim_matrix_file:
			i=0
			print ("Length of song vectors =",len(self.song_vectors))
			song_id_ints = list(self.song_vectors.keys())
			for song_id_int_row in song_id_ints:
				print ("Writing row ",i," of song similarity matrix to file ")
				write_str = ""
				for j in range(i,len(song_id_ints)):
					song_id_int_col = song_id_ints[j]
					write_str += str(song_id_int_row)+","+str(song_id_int_col)+","+str(1-spatial.distance.cosine(self.song_vectors[song_id_int_row], self.song_vectors[song_id_int_col]))+"\n"
				sim_matrix_file.write(write_str)
				i += 1
		print ("full song similiarity matrix written to file.")

	def get_song_pairs(self, idxs, song_pairs):
		mat = song_pairs[idxs]
		return (mat[:,0], mat[:,1])

	def transform_song_vectors(self, user_id):

		# read user similarity matrix
		user_sim = np.loadtxt('../datasets/lastfm-dataset-1K/extracts/ind_full_song_sim_matrix_'+str(user_id), delimiter=',', usecols=2)
		# plt.hist(user_sim)
		# plt.show()
		# user_sim[i] is the similarity of the songs in song_pairs[i]
		song_pairs = np.loadtxt('../datasets/lastfm-dataset-1K/extracts/ind_full_song_sim_matrix_'+str(user_id), delimiter=',', usecols=[0,1], dtype=np.int)

		# songs that have a similarity value within the 10th percentile of
		# similarity values are considered similar.
		sim_cut_off = len(user_sim) - int(len(user_sim)/10)
		sim_songs_idxs = np.argpartition(user_sim, sim_cut_off)
		# a[i] and b[i] are similar songs
		a,b = self.get_song_pairs(sim_songs_idxs[sim_cut_off:], song_pairs)

		# songs with similarity values greater than the 90th percentile are
		# considered dissimilar.
		dissim_cut_off = int(len(user_sim)/10)
		dissim_songs_idxs = np.argpartition(user_sim, dissim_cut_off)
		# c[i] and d[i] are dissimilar songs.
		c,d = self.get_song_pairs(dissim_songs_idxs[:dissim_cut_off], song_pairs)

		# read song features
		song_features = np.loadtxt('../datasets/lastfm-dataset-1K/extracts/combined_song_vectors')
		song_ids = song_features[:,0].astype(np.int)
		arg_sorted_ids = np.argsort(song_ids)
		song_features = song_features[arg_sorted_ids,1:]
		mmc = MMC(max_iter=1000)
		constraints = (np.array(a),np.array(b),np.array(c),np.array(d))
		transformed_songs = mmc.fit_transform(song_features, constraints)
		# np.save('song_features', song_features)
		np.save('../datasets/lastfm-dataset-1K/extracts/transformed_songs_vectors_'+str(user_id), transformed_songs)



