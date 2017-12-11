from Constants import Constants
from Song import Song
from User import User
import json
import datetime as dt
import pickle
import sys
import math
import numpy as np
from scipy.sparse import csc_matrix
import random
import collections
import concurrent.futures

class TrainTestSetGen:

	def __init__(self):
		self.training_sessions = {} # maps user_id to training song sessions (list of lists)
		self.test_songs = {} # maps user_id to test song map
		self.all_session_stats = {} # maps user_id to another dict. Second dict maps session_len to frequency of occurence
		self.training_session_stats = {} # maps user_id to another dict. Second dict maps session_len to frequency of occurence

	def add_session_stats(self, stats_map, user_id, session):
		session_len = len(session)
		if user_id in stats_map:
			if session_len in stats_map[user_id]:
				stats_map[user_id][session_len] = stats_map[user_id][session_len] + 1
			else:
				stats_map[user_id][session_len] = 1
		else:
			stats_map[user_id] = {}
			stats_map[user_id][session_len] = 1

	def del_session_stats(self, stats_map, user_id, session):
		session_len = len(session)
		if user_id in stats_map:
			if session_len in stats_map[user_id]:
				stats_map[user_id][session_len] = stats_map[user_id][session_len] - 1
				if stats_map[user_id][session_len] == 0:
					del stats_map[user_id][session_len]
			else:
				print ("error: session len not in session_stats")
		else:
			print ("error: user_id not in session_stats")

	def print_session_stats(self, stats_map):
		for user_id in stats_map:
			print ("Session stats for user: ",user_id)
			#for k, v in stats_map[user_id].iteritems()
			od = collections.OrderedDict(sorted(stats_map[user_id].items()))
			for session_len, session_freq in od.items():
				#print ("session len:",session_len," count:",stats_map[user_id][session_len])
				print ("session len:",session_len," count:",session_freq)

	def print_user_session_stats(self, user_id, stats_map):
		print ("Session stats for user: ",user_id)
		#for k, v in stats_map[user_id].iteritems()
		od = collections.OrderedDict(sorted(stats_map[user_id].items()))
		for session_len, session_freq in od.items():
			#print ("session len:",session_len," count:",stats_map[user_id][session_len])
			print ("session len:",session_len," count:",session_freq)

	def add_session(self, user_id, session):
		if user_id in self.training_sessions:
			sessions = self.training_sessions[user_id]
		else:
			sessions = []
		sessions.append(session)
		self.training_sessions[user_id] = sessions
		self.add_session_stats(self.training_session_stats, user_id, session)
		return len(self.training_sessions[user_id])-1 # returns the index at which session was added

	def del_session(self, user_id, add_index):
		session = self.training_sessions[user_id].pop(add_index)
		self.del_session_stats(self.training_session_stats, user_id, session)

	def create_training_sessions(self, user_id, song_id_int, session_song_id_ints):
		start_index = 0
		curr_index = 0
		s1 = []
		s2 = []
		num_split_sessions = 0
		song_present = False
		while curr_index < len(session_song_id_ints):
			if song_id_int == session_song_id_ints[curr_index]:
				song_present = True
				s1 = session_song_id_ints[start_index:curr_index]
				s2 = session_song_id_ints[curr_index+1:]
				if len(s1) > 0:
					self.add_session(user_id, s1)
					num_split_sessions += 1
				start_index = curr_index + 1
				#print ("Splitting into two lists s1 ",s1, " s2 ", s2," num split sessions=",num_split_sessions)
			curr_index += 1
		if len(s2) > 0:
			self.add_session(user_id, s2)
			num_split_sessions += 1
			#print ("Adding s2 to split session list num split sessions=",num_split_sessions)

		if num_split_sessions > 0:
			return True
		elif (song_present==True):
			return True
		else:
			return False

	def create_training_sessions_without_split_on_test_song(self, user_id, song_id_int, session_song_id_ints):

		curr_index = 0
		song_present = False
		while curr_index < len(session_song_id_ints):
			if song_id_int == session_song_id_ints[curr_index]:
				song_present = True
				session_song_id_ints.remove(song_id_int)
			else:
				curr_index += 1

		if song_present == True:
			self.add_session(user_id, session_song_id_ints)
			return True
		else:
			return False

	def validate_and_write_training_sessions(self):
		for user_id in self.test_songs:
			#test_set_songs = self.test_songs[user_id]
			#test_set_songs_map = {k: 0 for k in test_set_songs} # map for faster lookups
			training_set_sessions = self.training_sessions[user_id]
			with open('../datasets/lastfm-dataset-1K/extracts/training_sessions_'+user_id, 'w') as training_sessions_file:
				# make sure none of the test set songs are in training sessions
				session_idx = 0
				for session in training_set_sessions:
					if len(session) == 0:
						continue
					song_count = 0
					for song_id_int in session:
						if song_id_int in self.test_songs[user_id]:
							print ("error: found ",song_id_int," session idx ",session_idx," session ",session," in test_set_songs")
							return False
						song_count += 1
						if song_count == len(session):
							training_sessions_file.write(str(song_id_int))
						else:
							training_sessions_file.write(str(song_id_int)+",")

					training_sessions_file.write("\n")
					session_idx += 1

			with open('../datasets/lastfm-dataset-1K/extracts/test_songs_'+user_id, 'w') as test_songs_file:
				for song_id_int in self.test_songs[user_id].keys():
					test_songs_file.write(str(song_id_int)+"\n")
		return True

	def pick_test_songs_equidistantly(self, user_id, session_idx, session, num_songs_to_pick):

		step_count = math.floor(len(session)/(num_songs_to_pick+1))
		#print ("Session #", session_idx," num songs in session ",len(session)," step count is ",step_count," num songs to pick = ",num_songs_to_pick)
		#print ("Session songs ", session)
		for i in range(1,num_songs_to_pick+1):
			#print ("Iteration is ",i)
			if session[i*step_count] not in self.test_songs[user_id]:
				self.test_songs[user_id][session[i*step_count]] = 1
				#print ("Picked song ",session[i*step_count]," size of test set = ",len(self.test_songs[user_id]))
			#else:
				#print ("Skipping song ",session[i*step_count],", already in test list")

	def pick_test_songs_randomly(self, user_id, session_idx, session, num_songs_to_pick):

		random_indices = random.sample(range(0, len(session)), len(session))
		idx = 0
		num_songs_picked = 0
		#print ("Session #", session_idx," num songs in session ",len(session)," num songs to pick = ",num_songs_to_pick)
		#print ("Session songs ", session)
		for rand_idx in random_indices:
			if session[rand_idx] not in self.test_songs[user_id]:
				self.test_songs[user_id][session[rand_idx]] = 1
				num_songs_picked += 1
				if (num_songs_picked == num_songs_to_pick):
					break

	def gen_user_train_test_sets(self, user_db, song_db, user_id):
		song_list = list(song_db.keys())
		num_test_songs = int(len(user_db[user_id].songs)*Constants.TEST_SET_RATIO)
		print ("User_id=",user_id," num total songs =",len(user_db[user_id].songs)," num test songs = ",num_test_songs," num sessions =", len(user_db[user_id].play_sessions.sessions))

		# loop through all the sessions for this user and find test set songs from the session
		session_idx = 0
		self.test_songs[user_id] = {}
		for session in user_db[user_id].play_sessions.sessions:
			self.add_session_stats(self.all_session_stats, user_id, session)
			session_song_id_ints = [x[1].song_id_int for x in session]
			#print (session_song_id_ints)
			num_songs_to_pick = round(num_test_songs*len(session_song_id_ints)/user_db[user_id].play_sessions.get_total_sessions_songs())

			self.pick_test_songs_equidistantly(user_id, session_idx, session_song_id_ints, num_songs_to_pick)
			#self.pick_test_songs_randomly(user_id, session_idx, session_song_id_ints, num_songs_to_pick)
			session_idx += 1

		print ("Test set size=",len(self.test_songs[user_id]))
		#print ("Test set size=",len(self.test_songs[user_id])," songs ", self.test_songs[user_id].keys())
		self.print_user_session_stats(user_id, self.all_session_stats)

		# loop through all the sessions for this user and create training/test sessions
		start_index = 0
		session_idx = 0
		break_loop = 0
		for session in user_db[user_id].play_sessions.sessions:
			#print ("Session #", session_idx," num songs in session ",len(session_song_id_ints))
			session_idx += 1
			#print ("Session songs ", [x[1].song_id_int for x in session])
			first_test_song = True
			add_index = self.add_session(user_id, [x[1].song_id_int for x in session])
			#breakloop = False
			for song_id_int in self.test_songs[user_id]:
				#print ("Removing song id int ", song_id_int," from the session, start_index=",start_index)
				if first_test_song:
					# del_orig_session = self.create_training_sessions(user_id, song_id_int, self.training_sessions[user_id][add_index])
					del_orig_session = self.create_training_sessions_without_split_on_test_song(user_id, song_id_int, self.training_sessions[user_id][add_index])
					if (del_orig_session==True):
						self.del_session(user_id, add_index)
						first_test_song = False
						#print ("Split sessions ", self.training_sessions[user_id])
				else:
					idx = start_index
					for split_session in self.training_sessions[user_id][start_index:]:
						# del_orig_session = self.create_training_sessions(user_id, song_id_int, split_session)
						del_orig_session = self.create_training_sessions_without_split_on_test_song(user_id, song_id_int, split_session)
						if (del_orig_session==True):
							self.del_session(user_id, idx)
							#print ("Split sessions ", self.training_sessions[user_id])
							#breakloop = True
						else:
							idx += 1
				#print ("Split sessions final", self.training_sessions[user_id])
				# if breakloop == True:
				# 	abd
			start_index = len(self.training_sessions[user_id])

			if self.validate_and_write_training_sessions() != True:
				break
		self.print_user_session_stats(user_id, self.training_session_stats)

	def split_data_into_train_test_sets(self, user_db, song_db, user_id):

		random.seed(9001)

		# executor = concurrent.futures.ProcessPoolExecutor(8)
		# futures = [executor.submit(self.gen_user_train_test_sets, user_db, song_db, user_id) for user_id in user_db]
		# concurrent.futures.wait(futures)

		# loop = 0
		# for user_id in user_db:
		# 	self.gen_user_train_test_sets(user_db, song_db, user_id)
		# 	loop += 1
		# 	if loop==3:
		# 		break

		self.gen_user_train_test_sets(user_db, song_db, user_id)
		#self.validate_training_sessions()
		self.print_session_stats(self.training_session_stats)
