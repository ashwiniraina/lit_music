
class Song:
	# song_name is used as the unique key as dataset does not have song_id for all songs.
	# song_name = "uninitialized"
	# song_id = "uninitialized"
	# song_id_int = -1
	# artist_id = "uninitialized"
	# artist_name = "uninitialized"

	# # song statistics
	# # all users that played this song
	# users = None
	# num_times_song_played = 0

	# song_id to song_id_int map
	song_id_to_int_id_map = {}
	
	def __init__(self, artist_id, artist_name, song_id, song_name):
		self.artist_id = artist_id
		self.artist_name = artist_name
		self.song_id = song_id
		self.song_name = song_name
		self.users = {} # map of user_id to play count
		self.num_times_song_played = 0

	def set_song_stats(self, user):
		user_id = user.user_id
		if user_id in self.users:
			count = self.users[user_id]
			self.users[user_id] = count+1
		else:
			self.users[user_id] = 1

		self.num_times_song_played += 1

	def get_song_id_int(self):
		return self.song_id_int

	def set_song_id_int(self):
		temp_id = len(Song.song_id_to_int_id_map)
		if song_id not in Song.song_id_to_int_id_map:
			Song.song_id_to_int_id_map[song_id] = temp_id

	def get_num_unique_users(self):
		return (len(self.users))

	@staticmethod
	def clear_song_id_to_int_id_map():
		Song.song_id_to_int_id_map = {}

	def print_class_state(self):
		print ("song_name=",self.song_name,"song_id=",self.song_id,"artist_id=",self.artist_id,"artist_name=",self.artist_name)
		print ("num_times_song_played=",self.num_times_song_played,"num_unique_users_played=",len(self.users))
		print ("users=")
		index = 0
		user_list= ""
		for user_id in self.users:
			user_list += str(index)+":"+user_id+" "
			index += 1

		print (user_list)
