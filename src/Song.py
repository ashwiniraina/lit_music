
class Song:
	# song_name is used as the unique key as dataset does not have song_id for all songs.
	song_name = "uninitialized"
	song_id = "uninitialized"
	artist_id = "uninitialized"
	artist_name = "uninitialized"

	# song statistics
	# all users that played this song
	users = None
	num_times_song_played = 0
	
	def __init__(self, artist_id, artist_name, song_id, song_name):
		self.artist_id = artist_id
		self.artist_name = artist_name
		self.song_id = song_id
		self.song_name = song_name
		self.users = {}

	def set_song_stats(self, user):
		if user in self.users:
			count = self.users[user]
			self.users[user] = count+1
		else:
			self.users[user] = 1

		self.num_times_song_played += 1

	def get_num_unique_users(self):
		return (len(self.users))

	def print_class_state(self):
		print ("song_name=",self.song_name,"song_id=",self.song_id,"artist_id=",self.artist_id,"artist_name=",self.artist_name)
		print ("num_times_song_played=",self.num_times_song_played,"num_unique_users_played=",len(self.users))
		print ("users=")
		index = 0
		user_list= ""
		for user in self.users:
			user_list += str(index)+":"+user_id+" "
			index += 1

		print (user_list)
