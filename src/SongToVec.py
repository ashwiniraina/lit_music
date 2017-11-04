
from gensim.models import word2vec
import logging
import glob
import numpy as np
from scipy import spatial

class SongToVec:

	individual_song_vectors = {}
	individual_song_similarity_matrix = np.empty((0,0))

	combined_song_vectors = {}
	combined_song_similarity_matrix = np.empty((0,0))

	def __init__(self):
		logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

	def run(self):

		sessions = []
		# read the user play session files
		play_session_files = glob.glob("../datasets/lastfm-dataset-1K/extracts/play_session_*")
		for play_session_file in play_session_files:
			with open(play_session_file, 'r') as ps_file:
				for session in ps_file:
					session_song_id_ints = session.strip("\n").split(",")[1::2] # select every odd element from the list
					sessions.append(session_song_id_ints)
					for song_id_int in session_song_id_ints:
						if song_id_int not in self.combined_song_vectors:
							self.combined_song_vectors[song_id_int] = [] # empty list, will be filled with song vector in generate_song_vectors
		
		model = self.run_word2vec_model(sessions)

		self.generate_song_vectors(model)

		self.generate_similarity_matrix()


	def run_word2vec_model(self, sessions):

		# train the skip-gram model; default window=5
		model = word2vec.Word2Vec(sessions, size=100, window=5, min_count=1, workers=4)
		 
		# # pickle the entire model to disk, so we can load&resume training later
		# model.save('../datasets/lastfm-dataset-1K/extracts/song2vec.model')
		# # store the learned weights, in a format the original C tool understands
		# model.save_word2vec_format('../datasets/lastfm-dataset-1K/extracts/song2vec.model.bin', binary=True)
		# # or, import word weights created by the (faster) C word2vec
		# # this way, you can switch between the C/Python toolkits easily
		# model = word2vec.Word2Vec.load_word2vec_format('../datasets/lastfm-dataset-1K/extracts/song2vec.model.bin', binary=True)

		return model
		
	def generate_song_vectors(self, model):
		for song_id_int in self.combined_song_vectors:
			self.combined_song_vectors[song_id_int] = model.wv[song_id_int]
			#print ("song :",song_id_int," vector: ",self.combined_song_vectors[song_id_int])

	def generate_similarity_matrix(self):
		self.combined_song_similarity_matrix = np.zeros((len(self.combined_song_vectors), len(self.combined_song_vectors)))
		with open('../datasets/lastfm-dataset-1K/extracts/combined_song_similarity_matrix', 'w') as sim_matrix_file:
			i=j=0
			print ("Length of combined song vectors =",len(self.combined_song_vectors))
			for song_vec_i in self.combined_song_vectors:
				print ("Writing row ",i," of combined song similarity matrix to file ")
				for song_vec_j in self.combined_song_vectors:
					#print ("computing similarity for song vec i=",self.combined_song_vectors[song_vec_i]," and song vec j=",self.combined_song_vectors[song_vec_j])
					#self.combined_song_similarity_matrix[i][j] = 1-spatial.distance.cosine(self.combined_song_vectors[song_vec_i], self.combined_song_vectors[song_vec_j])
					#print ("similarity matrix [",i,"],[",j,"] = ", self.combined_song_similarity_matrix[i][j])
					sim_matrix_file.write(str(i)+","+str(j)+","+str(1-spatial.distance.cosine(self.combined_song_vectors[song_vec_i], self.combined_song_vectors[song_vec_j]))+"\n")
					j += 1
				i += 1
				j = 0
		print ("combined song similiarity matrix written to file.")



