
from Constants import Constants
import matplotlib.pyplot as plt
import numpy as np

class Evaluator:
	
	# input arguments are dictionaries which maps user_id to list of lists. 
	# list of lists is the top-n recommendation for each test song
	def precision(self, actual_rec_dict, predicted_rec_dict):
		total_precision_sum = 0.0
		num_users = 0

		for user_id in actual_rec_dict:
			user_actual_rec_list = actual_rec_dict[user_id]
			user_predicted_rec_list = predicted_rec_dict[user_id]
			num_users += 1.0
			user_precision_sum = 0.0
			num_test_songs = 0.0

			for i in range(0, len(user_actual_rec_list)):
				test_song_actual_rec_list = user_actual_rec_list[i][1:]
				test_song_predicted_rec_list = user_predicted_rec_list[i][1:]
				print ("test song id actual=",user_actual_rec_list[i][0]," test song id predicted=",user_predicted_rec_list[i][0])
				if user_actual_rec_list[i][0] != user_predicted_rec_list[i][0]:
					print ("actual and predicted test songs not same")
					abd
				if len(test_song_actual_rec_list) != len(test_song_predicted_rec_list):
					print ("skipping")
					continue
				num_matches = 0.0
				num_test_songs += 1.0
				effective_len = 0.0

				#tempList = [5120, 4477, 3234, 1833, 4544, 1773, 3981, 1326, 1788, 2552]

				for song_id_int in test_song_predicted_rec_list:
				#for song_id_int in tempList:	
					if song_id_int != user_actual_rec_list[i][0]: 
						if song_id_int in test_song_actual_rec_list:
							num_matches += 1.0
						effective_len += 1.0

				print (test_song_actual_rec_list, test_song_predicted_rec_list)
				#print (test_song_actual_rec_list, tempList)
				#print ("song precision@10 = ", num_matches/len(test_song_actual_rec_list))
				#user_precision_sum += num_matches/len(test_song_actual_rec_list)
				print ("song precision@10 = ", num_matches/effective_len)
				user_precision_sum += num_matches/effective_len

			print ("user "+user_id+" precision@10 = ", user_precision_sum/num_test_songs)
			total_precision_sum += user_precision_sum/num_test_songs
		print ("total precision@10 = ", total_precision_sum/num_users)

	# input arguments are dictionaries which maps user_id to list of lists. 
	# list of lists is the top-n recommendation for each test song
	def map(self, actual_rec_dict, predicted_rec_dict):
		total_map_sum = 0.0
		num_users = 0

		for user_id in actual_rec_dict:
			user_actual_rec_list = actual_rec_dict[user_id]
			user_predicted_rec_list = predicted_rec_dict[user_id]
			num_users += 1.0
			user_map_sum += 0.0
			num_test_songs = 0.0

			for i in range(0, len(user_actual_rec_list)):
				test_song_actual_rec_list = user_actual_rec_list[i]
				test_song_predicted_rec_list = user_predicted_rec_list[i]
				if len(test_song_actual_rec_list) != len(test_song_predicted_rec_list):
					continue
				num_test_songs += 1.0
				precision = []
				num_matches = 0.0
				for i in range(0, len(test_song_actual_rec_list)):
					if test_song_actual_rec_list[i] == test_song_predicted_rec_list[i]:
						num_matches += 1.0
						precision.append(num_matches/(i+1))

				print (test_song_actual_rec_list, test_song_predicted_rec_list, precision)
				print ("song map = ", sum(precision)/num_matches)
				
				user_map_sum += sum(precision)/num_matches

			print ("user "+user_id+" map = ", user_map_sum/num_test_songs)
			total_map_sum += user_map_sum/num_test_songs
		print ("total map = ", total_map_sum/num_users)

	def run(self):
		actual_rec_dict = np.load('../datasets/lastfm-dataset-1K/extracts/top10actual.npy').item()
		#predicted_rec_dict = np.load('../datasets/lastfm-dataset-1K/extracts/top10predictions.npy').item()
		predicted_rec_dict = np.load('../datasets/lastfm-dataset-1K/extracts/top10bmfpredictions.npy').item()

		self.precision(actual_rec_dict, predicted_rec_dict)
		#self.map(actual_rec_dict, predicted_rec_dict)

