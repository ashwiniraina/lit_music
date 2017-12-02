
import glob
import numpy as np
from scipy import spatial
from Constants import Constants
import matplotlib.pyplot as plt

class Evaluator:
    
    # input arguments are dictionaries which maps user_id to list of lists. 
    # list of lists is the top-n recommendation for each test song
    def precision(actual_rec_list, predicted_rec_list):
    	total_precision_sum = 0.0
    	num_users = 0

    	for user_id in actual_rec_list:
    		user_actual_rec_list = actual_rec_list[user_id]
    		user_predicted_rec_list = predicted_rec_list[user_id]
    		num_users += 1.0
    		user_precision_sum += 0.0
    		num_test_songs = 0.0

    		for i in range(0, len(user_actual_rec_list)):
    			test_song_actual_rec_list = user_actual_rec_list[i]
    			test_song_predicted_rec_list = user_predicted_rec_list[i]
    			num_matches = 0.0

    			for song_id_int in test_song_predicted_rec_list:
    				num_test_songs += 1.0
    				if song_id_int in test_song_actual_rec_list:
    					num_matches += 1.0
				
				print (test_song_actual_rec_list, test_song_predicted_rec_list)
				print ("song precision@10 = ", num_matches/len(test_song_actual_rec_list))
				user_precision_sum += num_matches/len(test_song_actual_rec_list)

			print ("user "+user_id+" precision@10 = ", user_precision_sum/num_test_songs)
			total_precision_sum += user_precision_sum/num_test_songs
		print ("total precision@10 = ", total_precision_sum/num_users)

    # input arguments are dictionaries which maps user_id to list of lists. 
    # list of lists is the top-n recommendation for each test song
    def map(actual_rec_list, predicted_rec_list):
    	total_map_sum = 0.0
    	num_users = 0

    	for user_id in actual_rec_list:
    		user_actual_rec_list = actual_rec_list[user_id]
    		user_predicted_rec_list = predicted_rec_list[user_id]
    		num_users += 1.0
    		user_map_sum += 0.0
    		num_test_songs = 0.0

    		for i in range(0, len(user_actual_rec_list)):
    			test_song_actual_rec_list = user_actual_rec_list[i]
    			test_song_predicted_rec_list = user_predicted_rec_list[i]
		    	precision = []
		    	num_matches = 0.0
		    	for i in range(0, len(test_song_actual_rec_list)):
		    		if test_song_actual_rec_list[i] == test_song_predicted_rec_list[i]:
		    			num_matches += 1.0
		    			precision.append(num_matches/(i+1))

		    	print (test_song_actual_rec_list, test_song_predicted_rec_list)
		    	print ("song map = ", sum(precision)/num_matches)
				
				user_map_sum += sum(precision)/num_matches

			print ("user "+user_id+" map = ", user_map_sum/num_test_songs)
			total_map_sum += user_map_sum/num_test_songs
		print ("total map = ", total_map_sum/num_users)
