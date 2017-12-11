from gensim.models import word2vec
from glove import Corpus, Glove
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

	def read_combined_sessions(self, user_id="*"):
		sessions = []
		# read the user play session files
		play_session_files = glob.glob("../datasets/lastfm-dataset-1K/extracts/play_session_"+user_id)
		for play_session_file in play_session_files:
			with open(play_session_file, 'r') as ps_file:
				for session in ps_file:
					session_song_id_ints = session.strip("\n").split(",")[1::2] # select every odd element from the list
					sessions.append(session_song_id_ints)
					for song_id_int in session_song_id_ints:
						if song_id_int not in self.song_vectors:
							self.song_vectors[song_id_int] = [] # empty list, will be filled with song vector in generate_song_vectors
		return sessions

	def read_combined_sessions_minus_test_songs(self, user_id):
		sessions = []
		# read the user play session files
		play_session_files = glob.glob("../datasets/lastfm-dataset-1K/extracts/play_session_*")
		for play_session_file in play_session_files:
			if user_id not in play_session_file:
				with open(play_session_file, 'r') as ps_file:
					for session in ps_file:
						session_song_id_ints = session.strip("\n").split(",")[1::2] # select every odd element from the list
						sessions.append(session_song_id_ints)
						for song_id_int in session_song_id_ints:
							if song_id_int not in self.song_vectors:
								self.song_vectors[song_id_int] = [] # empty list, will be filled with song vector in generate_song_vectors

		with open("../datasets/lastfm-dataset-1K/extracts/training_sessions_"+str(user_id), 'r') as training_file:
			for session in training_file:
				session_song_id_ints = session.strip("\n").split(",")[:-1] # last element is empty
				sessions.append(session_song_id_ints)
				for song_id_int in session_song_id_ints:
					if song_id_int not in self.song_vectors:
						self.song_vectors[song_id_int] = [] # empty list, will be filled with song vector in generate_song_vectors
		return sessions

	def read_individual_sessions(self, user_id, mode):
		sessions = []
		# read the user play session files
		with open("../datasets/lastfm-dataset-1K/extracts/training_sessions_"+str(user_id), 'r') as training_file:
			for session in training_file:
				session_song_id_ints = session.strip("\n").split(",")[:-1] # last element is empty
				sessions.append(session_song_id_ints)
				for song_id_int in session_song_id_ints:
					if song_id_int not in self.song_vectors:
						self.song_vectors[song_id_int] = [] # empty list, will be filled with song vector in generate_song_vectors


		# if mode == self.constants.RUN_SONG2VEC_ON_ALL_USER_SONGS:
		# 	# read the user play test session files
		# 	with open("../datasets/lastfm-dataset-1K/extracts/test_songs_"+str(user_id), 'r') as test_file:
		# 		for song in test_file:
		# 			song_id_int = song.strip("\n")
		# 			sessions.append(song_id_int)
		# 			self.song_vectors[song_id_int] = [] # empty list, will be filled with song vector in generate_song_vectors
		return sessions

	def run(self, user_db, song_db, mode):

		if mode == self.constants.RUN_SONG2VEC_ON_ALL_SONGS:
			for user_id in user_db:
				if user_id == 'user_000002':
					sessions = self.read_combined_sessions_minus_test_songs(user_id)
					model = self.run_word2vec_model(sessions)
					self.generate_song_vectors(model, 'combined_song_vectors_'+str(user_id))
					self.find_knn_for_song_vectors(self.constants.NUM_NEAREST_NEIGHBORS, 'combined_knn_song_sim_matrix_'+str(user_id))
		elif mode == self.constants.RUN_SONG2VEC_ON_USER_TRAINING_SONGS:
			for user_id in user_db:
				if user_id == 'user_000002':
					sessions = self.read_individual_sessions(user_id, mode)
					model = self.run_word2vec_model(sessions)
					self.generate_song_vectors(model, 'ind_song_vectors_'+str(user_id))
					self.generate_full_similarity_matrix('ind_full_song_sim_matrix_'+str(user_id))
					#self.find_knn_for_song_vectors(self.constants.NUM_NEAREST_NEIGHBORS, 'ind_knn_song_sim_matrix_'+str(user_id))
					self.transform_song_vectors(user_id)
		elif mode == self.constants.RUN_SONG2VEC_ON_ALL_USER_SONGS:
			for user_id in user_db:
				if user_id == 'user_000002':
					sessions = self.read_combined_sessions(user_id)
					model = self.run_word2vec_model(sessions)
					self.generate_song_vectors(model, 'ind_song_vectors_train_and_test_'+str(user_id))
					self.generate_full_similarity_matrix('ind_full_song_sim_matrix_train_and_test_'+str(user_id))
		else:
			print ("error : incorrect mode")
			return

	def run_word2vec_model(self, sessions):

		# train the skip-gram model; default window=5

		model = word2vec.Word2Vec(sessions, size=30, window=5, min_count=1, workers=4, sg=1)

		# # pickle the entire model to disk, so we can load&resume training later
		# model.save('../datasets/lastfm-dataset-1K/extracts/song2vec.model')
		# # store the learned weights, in a format the original C tool understands
		# model.save_word2vec_format('../datasets/lastfm-dataset-1K/extracts/song2vec.model.bin', binary=True)
		# # or, import word weights created by the (faster) C word2vec
		# # this way, you can switch between the C/Python toolkits easily
		# model = word2vec.Word2Vec.load_word2vec_format('../datasets/lastfm-dataset-1K/extracts/song2vec.model.bin', binary=True)

		return model
        # def run_glove_model(self, sessions):
        #         corpus = Corpus()
        #         corpus.fit(sessions, window=5)
        #         model = Glove(no_components=30, learning_rate=0.05)
        #         model.fit(corpus.matrix, epochs=30, no_threads=4, verbose=True)
        #         # model.add_dictionary(corpus.dictionary)
        #         return model


	def generate_song_vectors(self, model, filename):
		with open('../datasets/lastfm-dataset-1K/extracts/'+filename, 'w') as song_vectors_file:
			for song_id_int in self.song_vectors:
				self.song_vectors[song_id_int] = model.wv[song_id_int]
				self.song_vectors_array.append(self.song_vectors[song_id_int])
				self.song_vectors_int_ids.append(song_id_int)
				#print ("song :",song_id_int," vector: ",self.song_vectors[song_id_int])
				# song_vectors_file.write(str(song_id_int)+","+str(self.song_vectors[song_id_int])+"\n")
				song_vectors_file.write(str(song_id_int)+" "+' '.join([str(x) for x in self.song_vectors[song_id_int])+"\n")

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
				if i%100 == 0:
					print ("Writing row ",i," of song similarity matrix to file ")
				write_str = ""
				for j in range(i,len(song_id_ints)):
					song_id_int_col = song_id_ints[j]
					write_str += str(song_id_int_row)+","+str(song_id_int_col)+","+str(1-spatial.distance.cosine(self.song_vectors[song_id_int_row], self.song_vectors[song_id_int_col]))+"\n"
				sim_matrix_file.write(write_str)
				i += 1
		print ("full song similiarity matrix written to file.")
