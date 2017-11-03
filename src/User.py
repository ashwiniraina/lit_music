from PlaySessions import PlaySessions

class User:
	
	user_id = "uninitialized"
	gender = "uninitialized"
	age = -1
	country = "uninitialized"
	reg_date = "uninitialized"
	play_sessions = None

	# user statistics
	# songs played by this user
	songs = None
	num_songs_played = 0

	def __init__(self, user_id, gender, age, country, reg_date):
		self.user_id = user_id
		self.gender = gender
		self.age = age
		self.country = country
		self.reg_date = reg_date
		self.play_sessions = PlaySessions(self)
		self.songs = {}

	def set_user_stats(self, song):
		if song in self.songs:
			count = self.songs[song]
			self.songs[song] = count+1
		else:
			self.songs[song] = 1
			
		self.num_songs_played += 1

	def get_num_unique_songs(self):
		return(len(self.songs))

	def print_class_state(self):
		print ("user_id=",self.user_id,"gender=",self.gender,"age=",self.age,"country=",self.country,"reg_date=",self.reg_date)
		print ("num_songs_played=",self.num_songs_played,"num_unique_songs_played=",len(self.songs))
		print ("songs=")
		index = 0
		song_list = ""
		for song in self.songs:
			song_list += str(index)+":"+song.song_name+" "
			index += 1
		print(song_list)


