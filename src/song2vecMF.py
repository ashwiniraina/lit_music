import numpy as np
from numpy.random import rand
import timeit

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
        numIters = 5
        me = self.trainMatrix.tocoo()
        me_zip = zip(me.row, me.col, me.data)
        for iter in range(1, numIters+1):
            loss = 0
            print(iter)
            for u, j, ruj in me_zip:
                start_time = timeit.default_timer()
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
                delta_u = (euj * self.Q[j,:]) - (regU *  self.P[u,:])
                delta_j = (euj * self.P[u,:]) - (regI * self.Q[j,:])
                self.P[u,:] += lRate * delta_u
                self.Q[j,:] += lRate * delta_j
                # elapsed = timeit.default_timer() - start_time
                # print('1:',elapsed)
                # for f in range(self.num_factors):
                #     puf = self.P[u, f] # P is the matrix for users.
                #     qjf = self.Q[j, f] # Q is the matrix for songs
                #     delta_u = euj * qjf - regU * puf # -1*grad of loss wrt p_u
                #     delta_j = euj * puf - regI * qjf #
                #     self.P[u, f] += lRate * delta_u # update P
                #     self.Q[j, f] += lRate * delta_j # update Q

                    # loss += regU * puf * puf + regI * qjf * qjf


                # i think looping over nearest neighbors here
                # and simMatrix is probably a sparseMatrix of
                # song similarites with nearest neighbors.
                start_time = timeit.default_timer()
                tj = self.simMatrix.getrow(j).tocoo()
                if tj.nnz > 0:
                    for k,sim_jk in zip(tj.col, tj.data):
                        if (sim_jk > 0):
                            # \alpha * (s_ij - q_i^Tq_j)
                            ejk = regAlpha * (sim_jk - self.Q[j,:].dot(self.Q[k,:]))
                            delta_j = ejk * self.Q[k,:]
                            self.Q[j,:] += lRate * delta_j
                            for f in range(self.num_factors):
                                delta_j = ejk * self.Q[k, f]
                                self.Q[j, f] +=  lRate * delta_j


            loss *= 0.5;
