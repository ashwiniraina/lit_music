import numpy as np
from numpy.random import rand

class Song2vecMF:
    def __init__(self, num_users, num_items, initialQ, train_matrix, simMatrix, num_factors=128):
        self.userBias = rand(num_users)
        self.itemBias = rand(num_items)
        self.globalMean = 1 # for now
        self.P = rand(1,num_factors)
        self.Q = initialQ
        self.trainMatrix = train_matrix
        self.num_factors = num_factors
        self.simMatrix = simMatrix

    def predict (self, u, j):
        return self.globalMean + self.userBias[u] + self.itemBias[j] + self.P[u,:].dot(self.Q[j,:])


    def buildModel(self, lRate=0.001, regB=0.001, regAlpha=0.001,
                   regU=0.001, regS=0.001, regI=0.001):
        # itemCache = simMatrix.rowColumnsCache(cacheSpec)
        numIters = 1000
        for iter in range(1, numIters+1):
            loss = 0
            me = self.trainMatrix.tocoo()
            for u, j, ruj in zip(me.row, me.col, me.data):
                pred = self.predict(u, j)
                # error = (actual rating - predicted)
                euj = ruj - pred
                loss += euj * euj # squared loss function

                # update factors
                bu = self.userBias[u] # userbias
                sgd = euj - regB * bu # regB is lambda
                self.userBias[u] += lRate * sgd # update userbias

                loss += regB * bu * bu
                # update itembias
                bj = self.itemBias[j]
                sgd = euj - regB * bj
                self.itemBias[j] += lRate * sgd

                loss += regB * bj * bj;
                # f is the number of dimensions for a user vector
                for f in range(self.num_factors):
                    puf = self.P[u, f] # P is the matrix for users.
                    qjf = self.Q[j, f] # Q is the matrix for songs
                    delta_u = euj * qjf - regU * puf # -1*grad of loss wrt p_u
                    delta_j = euj * puf - regI * qjf #
                    self.P[u, f] += lRate * delta_u # update P
                    self.Q[j, f] += lRate * delta_j # update Q

                    loss += regU * puf * puf + regI * qjf * qjf


                # i think looping over nearest neighbors here
                # and simMatrix is probably a sparseMatrix of
                # song similarites with nearest neighbors.
                tj = self.simMatrix.getrow(j)
                if tj.nnz > 0:
                    tj = tj.toarray()
                    for k,sim_jk in enumerate(tj):
                        if (sim_jk > 0):
                            # \alpha * (s_ij - q_i^Tq_j)
                            ejk = regAlpha * (sim_jk - self.Q[j,:].dot(self.Q[k,:]))
                            for f in range(numFactors):
                                delta_j = ejk * self.Q[k, f]
                                self.Q[j, f] +=  lRate * delta_j

                            loss += regS * ejk * ejk;

            loss *= 0.5;
            if (isConverged(iter)):
                break
