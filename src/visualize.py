from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy
from scipy.linalg import svd

# each column is a song
def plot_songs(X, type='2D'):
    meanCol = np.mean(X, axis=1)
    X = (X.T - meanCol).T # center the data
    U,S,Vh = svd(X, full_matrices=False) # columns of U are what we need
    if type=='2D':
        num_bases = 2 # only keeping these many `eigen_faces`
        eigen_songs = U[:,:num_bases]
        plt.plot(eigen_songs[:,0], eigen_songs[:,1], 'o')
    else:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(eigen_songs[:,0], eigen_songs[:,1], eigen_songs[:,2])
        plt.show()
