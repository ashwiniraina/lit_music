from Constants import Constants
from Song import Song
from User import User
import json
import datetime as dt
import pickle
import sys
import numpy as np
from scipy.sparse import csc_matrix, diags

class DatasetReader:
	constants = Constants()

	def __init__(self):
		sys.setrecursionlimit(self.constants.MAX_SYS_RECURSION_DEPTH)

	def read(self, dataset_name):
		if dataset_name == self.constants.DATASET_LASTFM_1K:
			return self.read_lastfm_1k_dataset()
		elif dataset_name == self.constants.DATASET_30MUSIC:
			self.read_30music_dataset()
		elif dataset_name == self.constants.MAPS_LASTFM_1K:
			return self.read_lastfm_1k_map_files()
		elif dataset_name == self.constants.MAPSS_30MUSIC:
			self.read_30music_map_files()
		else:
			print ("error: unknown dataset name")

	def print_user_db(self, user_db):
		print ("USER_DB:")
		for user_id in user_db:
			print (user_id, user_db[user_id].gender, user_db[user_id].age, user_db[user_id].country, user_db[user_id].reg_date)

	def print_song_db(self, song_db):
		print ("SONG_DB:")
		index = 0
		total_songs_in_db = 0
		total_songs_played = 0
		for song_id in song_db:
			print ("Index: ",index,"song id:"+song_id, "num times played=",song_db[song_id].num_times_song_played, "num_unique_users_played=",song_db[song_id].get_num_unique_users())
			total_songs_in_db += 1
			total_songs_played += song_db[song_id].num_times_song_played
			index += 1
		print ("Total songs in db=",total_songs_in_db, "total songs played=",total_songs_played)

	def write_map_objects_to_files(self, user_db, song_db):
		with open('../datasets/lastfm-dataset-1K/extracts/user_db.map', 'wb') as user_db_file:
			pickle.dump(user_db, user_db_file)
		print ("user_db.map writing complete")
		with open('../datasets/lastfm-dataset-1K/extracts/song_db.map', 'wb') as song_db_file:
			pickle.dump(song_db, song_db_file)
		print ("song_db.map writing complete")
		with open('../datasets/lastfm-dataset-1K/extracts/song_id_to_int_id.map', 'w') as song_id_to_int_id_file:
			for song_id in song_db:
				song_id_to_int_id_file.write(str(song_id)+","+str(song_db[song_id].get_song_id_int())+"\n")

	def write_play_sessions_to_file(self, user_db):
		for user_id in user_db:
			with open('../datasets/lastfm-dataset-1K/extracts/play_session_'+user_id, 'w') as play_session_user_file:
				for session in user_db[user_id].play_sessions.sessions:
					session_str = ""
					first_event = True
					for event in session:
						if first_event == False:
							session_str += ","
						else:
							first_event = False
						session_str += str(event[0])+","+str(event[1].get_song_id_int())
					play_session_user_file.write(session_str+"\n")
		print ("user_db play sessions writing complete")

	def read_lastfm_1k_dataset(self):

		user_db, infrequent_user_map, infrequent_song_map = self.find_infrequent_users_and_songs()
		loop = 0
		num_interrrupted_sessions = 0
		num_play_sessions = 0
		num_zero_session_users = 0
		# populate the song database
		song_db = {} # maps song_id to song object
		with open('../datasets/lastfm-dataset-1K/userid-timestamp-artid-artname-traid-traname.tsv', 'r') as song_history_file:
			play_session = []
			prev_user_id = ""
			prev_timestamp = ""

			for song_event in song_history_file:
				song_info = song_event.strip("\n").split("\t")
				user_id = song_info[0]
				timestamp = dt.datetime.strptime(song_info[1], "%Y-%m-%dT%H:%M:%SZ")
				song_id = song_info[-2]

				# skip infrequent users and songs
				if user_id in infrequent_user_map or song_id in infrequent_song_map:
					continue

				#print ("Index: ",loop," user_id: ",user_id," timestamp=",timestamp," song id: ",song_id)
				# some entries dont have song_id or artist_id. song_id is the unique key.
				# song_id is the second last element of song_info
				if song_id not in song_db:
					# index 2 is artist_id, 3 is artist_name, 4 is song_id, 5 is song_name
					song_object = Song(song_info[2], song_info[3], song_info[4], song_info[5])
					song_db[song_id] = song_object
				else:
					song_object = song_db[song_id]

				user_object = user_db[user_id]
				song_object.set_song_stats(user_object)
				user_object.set_user_stats(song_object)

				# check if current event user_id is same as prev_user_id
				if (user_id == prev_user_id):
					seconds_elapsed = (prev_timestamp-timestamp).total_seconds()
					if seconds_elapsed > self.constants.SESSION_INTERRUPTION_DURATON_IN_SECS:
						# songs were interrupted for more than 800secs, mark the end of session
						if len(play_session) >= self.constants.MIN_PLAY_SESSION_SONG_COUNT:
							user_object.play_sessions.append_session(play_session)
							num_play_sessions += 1
							#print ("Added session user_id=",user_id," session len=",len(play_session)," num play sessions=",num_play_sessions," sessions=",[(x[0],x[1].song_id_int) for x in play_session])
							# if len(play_session) > 150:
							# 	print ("Long listening session len=",len(play_session), [(x[0],x[1].song_id) for x in play_session])
						# else:
						# 	print ("Short listening session len=",len(play_session), [x[1].song_id for x in play_session])
						play_session = []
						num_interrrupted_sessions += 1
						#print ("Found interrupted session, user_id=",user_id," seconds elapsed=",seconds_elapsed," num songs in session=",len(play_session)," total interrupted sessions=",num_interrrupted_sessions)
				else:
					if prev_user_id != "":
						print ("Processing user_id: ",user_id)
						# previous user song history has been completely processed, switching to new user.
						# mark the end of play session for previous user
						if len(play_session) >= self.constants.MIN_PLAY_SESSION_SONG_COUNT:
							user_db[prev_user_id].play_sessions.append_session(play_session)
							num_play_sessions += 1
							print ("Added session user_id=",user_id," session len=",len(play_session)," num play sessions=",num_play_sessions)
							#user_db[prev_user_id].print_class_state()
							#self.print_song_db(song_db)
						play_session = []
						# if user_id does not have any listening sessions, then delete it
						if user_db[prev_user_id].play_sessions.num_sessions() == 0:
							num_zero_session_users += 1
							#print (prev_user_id," has 0 sessions, total zero sessions=",num_zero_session_users)
							del user_db[prev_user_id]
					else:
						print ("Processing user_id: ",user_id)

				play_session.append((timestamp, song_object))
				prev_user_id = user_id
				prev_timestamp = timestamp
				loop += 1

			# mark the end of play session for the very last user
			if len(play_session) >= self.constants.MIN_PLAY_SESSION_SONG_COUNT:
				user_db[prev_user_id].play_sessions.append_session(play_session)
				num_play_sessions += 1
				print ("Added session user_id=",user_id," session len=",len(play_session)," num play sessions=",num_play_sessions)
				#user_db[prev_user_id].print_class_state()
				#self.print_song_db(song_db)
			play_session = []
			# if user_id does not have any listening sessions, then delete it
			if user_db[prev_user_id].play_sessions.num_sessions() == 0:
				num_zero_session_users += 1
				#print (prev_user_id," has 0 sessions, total zero sessions=",num_zero_session_users)
				del user_db[prev_user_id]

		print ("After post processing num valid users=",len(user_db),"Num valid songs=",len(song_db), "num infrequent users=",len(infrequent_user_map)," num_zero_session_users=",num_zero_session_users)

		total_play_sessions = 0
		total_play_session_songs = 0
		for user_id in user_db:
			for play_session in user_db[user_id].play_sessions.sessions:
				total_play_session_songs += len(play_session)
				total_play_sessions += 1
		print ("Total play sessions=",total_play_sessions," Total play session songs=",total_play_session_songs," Avg len of play sessions=",float(total_play_session_songs)/total_play_sessions)

		# write the user_db and song_db to files
		self.write_map_objects_to_files(user_db, song_db)

		# write the play sessions files
		self.write_play_sessions_to_file(user_db)

		return (user_db, song_db)

	def find_infrequent_users_and_songs(self):

		# populate the user database
		user_db = {} # maps user_id to user object
		with open('../datasets/lastfm-dataset-1K/userid-profile.tsv', 'r') as user_profile_file:
			header_line = True
			for user in user_profile_file:
				if header_line:
					header_line = False
					continue
				user_info = user.split("\t")
				# index 0 is user_id, 1 is gender, 2 is age, 3 is country, 4 is date registered
				user_object = User(user_info[0], user_info[1], user_info[2], user_info[3], user_info[4])
				user_db[user_info[0]] = user_object

		#self.print_user_db(user_db)

		loop = 0
		# populate the song database
		song_db = {} # maps song_id to song object
		with open('../datasets/lastfm-dataset-1K/userid-timestamp-artid-artname-traid-traname.tsv', 'r') as song_history_file:
			for song_event in song_history_file:
				song_info = song_event.strip("\n").split("\t")
				user_id = song_info[0]
				song_id = song_info[-2]
				#print ("Index: ",loop," user_id: ",user_id," timestamp=",timestamp," song id ",song_id)
				if song_id not in song_db:
					# index 2 is artist_id, 3 is artist_name, 4 is song_id, 5 is song_name
					song_object = Song(song_info[2], song_info[3], song_info[4], song_info[5])
					song_db[song_id] = song_object
				else:
					song_object = song_db[song_id]
				user_object = user_db[user_id]
				song_object.set_song_stats(user_object)
				user_object.set_user_stats(song_object)

				if loop%100000 == 0:
					print ("Index: ",loop," user_id: ",user_id,"song name: ",song_info[-1]," song_id: ",song_id)
				loop += 1

		print ("Before post processing num valid users=",len(user_db)," Num valid songs=",len(song_db))

		print (song_db[""].song_name, song_db[""].artist_name)

		infrequent_user_map = {}
		infrequent_song_map = {}
		while True:

			# find infrequent users
			infrequent_user_map_temp = {}
			for user_id in user_db:
				user_object = user_db[user_id]
				if user_object.get_num_unique_songs() < self.constants.MIN_SONGS_COUNT:
					infrequent_user_map[user_id] = 1
					infrequent_user_map_temp[user_id] = 1
					print ("Infrequent user_id=",user_id)
					for song_id in user_object.songs:
						song_object = song_db[song_id]
						count = song_object.users[user_object.user_id]
						song_object.num_times_song_played -= count
						del song_object.users[user_object.user_id]
			for user_id in infrequent_user_map_temp:
				del user_db[user_id]
			print ("Infrequent user list len=",len(infrequent_user_map))

			# find infrequent songs
			infrequent_song_map_temp = {}
			for song_id in song_db:
				song_object = song_db[song_id]
				if song_object.get_num_unique_users() < self.constants.MIN_USERS_COUNT:
					infrequent_song_map[song_id] = 1
					infrequent_song_map_temp[song_id] = 1
					for user_id in song_db[song_id].users:
						user_object = user_db[user_id]
						count = user_object.songs[song_object.song_id]
						user_object.num_songs_played -= count
						del user_object.songs[song_object.song_id]
			for song_id in infrequent_song_map_temp:
				del song_db[song_id]
			print ("Infrequent song map len=",len(infrequent_song_map))

			if len(infrequent_user_map_temp)==0 and len(infrequent_song_map_temp)==0:
				break

		# append the song with missing mb_id
		infrequent_song_map[""] = 1
		Song.clear_song_id_to_int_id_map()
		return user_db, infrequent_user_map, infrequent_song_map

	def read_lastfm_1k_map_files(self):
		user_db = {}
		with open('../datasets/lastfm-dataset-1K/extracts/user_db.map', 'rb') as user_db_file:
			user_db = pickle.load(user_db_file)

		song_db = {}
		with open('../datasets/lastfm-dataset-1K/extracts/song_db.map', 'rb') as song_db_file:
			song_db = pickle.load(song_db_file)

		return (user_db, song_db)

	def read_30music_dataset(self):
		print ("Nothing to do")


	def read_30music_map_files(self):
		print ("Nothing to do")

	def get_ratings_matrix(self, user_db, song_db):
		num_users = len(user_db.keys())
		num_songs = len(song_db.keys())
                 # a matrix of normalized counts
		ratings_mat = csc_matrix((num_users, num_songs), dtype=np.int16)
                max_song_rating = numpy.zeros(1,num_songs)
		for user_id, user_obj in user_db.items():
			user_id_int = int(user_id[5:])
			for song_obj,count in user_obj.songs.items():
				song_id = int(''.join(song_obj.song_id.split('-')), base=16)
                                # normalized count
                                rating = np.log(count/user_obj.num_songs_played + 1)
				ratings_mat[user_id, song_id] = rating
                                max_song_rating = max(max_song_rating[song_id], rating)

                ratings_mat.dot(diags([max_song_rating])) * 5


	        np.save('../datasets/lastfm-dataset-1K/extracts/rating_mat', ratings_mat)
