import numpy as np
from numpy.random import rand

class Song2vecMF:
    def __init__(self, num_users, num_items, initialQ, train_matrix, num_factors=128):
        self.userBias = rand(num_users, 1)
        self.itemBias = rand(num_items, 1)
        self.globalMean = 1 # for now
        self.P = rand(1,num_factors)
        self.Q = initialQ

    def predict (u, j):
        return self.globalMean + self.userBias.get[u] + self.itemBias.get[j] + P[u,:].dot(Q[j,:])


    def buildModel(lrate=0.001, regB=0.001, regAlpha=0.001, regU=0.001, regS=0.001, regI=0.001):
        # itemCache = simMatrix.rowColumnsCache(cacheSpec)
        numIters = 1000
        for iter in range(1, numIters+1):
            loss = 0
            me = trainMatrix.tocoo()
            for u, j, ruj in zip(me.row, me.col, me.data):
                pred = self.predict(u, j, false)
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
                for f in range(num_factors):
                    puf = P[u, f] # P is the matrix for users.
                    qjf = Q[j, f] # Q is the matrix for songs
                    delta_u = euj * qjf - regU * puf # -1*grad of loss wrt p_u
                    delta_j = euj * puf - regI * qjf #
                    P[u, f] += lRate * delta_u # update P
                    Q[j, f] += lRate * delta_j # update Q

                    loss += regU * puf * puf + regI * qjf * qjf


                # i think looping over nearest neighbors here
                # and simMatrix is probably a sparseMatrix of
                # song similarites with nearest neighbors.
                tj = simMatrix.getrow(j);
                if tj.nnz > 0:
                    for k in tj.indices:
                        sim_jk = simMatrix[j,k]
                        if (sim_jk > 0):
                            # \alpha * (s_ij - q_i^Tq_j)
                            ejk = regAlpha * (sim_jk - Q[j,:].dot(Q[k,:]))
                            for f in range(numFactors):
                                delta_j = ejk * Q[k, f]
                                Q[j, f] +=  lRate * delta_j

                            loss += regS * ejk * ejk;

            loss *= 0.5;
            if (isConverged(iter)):
                break
