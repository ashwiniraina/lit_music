from Constants import Constants
import numpy as np

class PlaySessions:
	constants = Constants()
	user = "uninitialized"
	#song_transition_matrix = np.zeros((constants.MAX_SONGS_LASTFM_1K, constants.MAX_SONGS_LASTFM_1K))
	sessions = None # list of lists

	def __init__(self, user):
		self.user = user
		self.sessions = []

	# input argument session is a list of tuples i.e. [(timestamp1, song1), (timestamp2, song2)...]
	def append_session(self, session):
		self.sessions.append(session)

	def num_sessions(self):
		return len(self.sessions)

