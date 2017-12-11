from Constants import Constants
import numpy as np

class PlaySessions:
	constants = Constants()
	# user = "uninitialized"
	# sessions = None # list of lists

	def __init__(self, user):
		self.user = user
		self.sessions = []
		self.total_sessions_songs = 0

	# input argument session is a list of tuples i.e. [(timestamp1, song_object1), (timestamp2, song_object2)...]
	def append_session(self, session):
		self.sessions.append(session)
		self.total_sessions_songs += len(session)
		for ts, song_object in session:
			song_object.set_song_id_int()

	def num_sessions(self):
		return len(self.sessions)

	def get_total_sessions_songs(self):
		return self.total_sessions_songs

