from Constants import Constants
import numpy as np

class PlaySessions:
	constants = Constants()
	# user = "uninitialized"
	# sessions = None # list of lists

	def __init__(self, user):
		self.user = user
		self.sessions = []

	# input argument session is a list of tuples i.e. [(timestamp1, song_int_id), (timestamp2, song_int_id)...]
	def append_session(self, session):
		self.sessions.append(session)

	def num_sessions(self):
		return len(self.sessions)

