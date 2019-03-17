import pickle
import numpy as np
import matplotlib.pyplot as pyplot
from sortedcontainers import SortedList
from datetime import datetime

movie_data_path = "/usr/src/app/data/movielens-20m-dataset/"
movie_dicts_path = movie_data_path + "dicts/"
user_to_movie_path = movie_dicts_path + 'user_to_movie.json'
movie_to_user_path = movie_dicts_path + 'movie_to_user.json'
user_movie_to_rating_train_path = movie_dicts_path + 'user_movie_to_rating.json'
user_movie_to_rating_test_path = movie_dicts_path + 'user_movie_to_rating_test.json'

class Mfnp():
    def __init__(self):
        self.N = 0
        self.M = 0
        self.K = 0
        self.W = 0
        self.U = 0
        self.b = 0
        self.c = 0
        self.mu = 0

        self.epochs = 0
        self.reg = 0

        self.user_to_movie = []
        self.movie_to_user = []
        self.user_movie_to_rating_train = []
        self.user_movie_to_rating_test = []

    def load_data(self):
        with open(user_to_movie_path, 'rb') as f:
            self.user_to_movie = pickle.load(f)

        with open(movie_to_user_path, 'rb') as f:
            self.movie_to_user = pickle.load(f)

        with open(user_movie_to_rating_train_path, 'rb') as f:
            self.user_movie_to_rating_train = pickle.load(f)

        with open(user_movie_to_rating_test_path, 'rb') as f:
            self.user_movie_to_rating_test = pickle.load(f)

        # find max number of users
        self.N = np.max(list(self.user_to_movie.keys())) + 1

        # movies might appear in the test set, but not the train set, check that here
        # we want to make sure that we get ALL the movie ids
        # movie to user is empty
        # print("this is movie to user keys", list(movie_to_user.keys()))

        m1 = np.max(list(self.movie_to_user.keys()))
        m2 = np.max([m for (u, m), r in self.user_movie_to_rating_test.items()])
        self.M = max(m1, m2) + 1

        print("N:", self.N, "M:", self.M)

    def initialize_variables(self):
        self.K = 10
        # W (N x K) - users matrix
        self.W = np.random.randn(self.N, self.K)
        # U (M x K) - movies matrix
        self.U = np.random.randn(self.M, self.K)

        # b and c are regularization, frobenius norm
        # b is user bias and c is movie bias
        # mu is the global average
        self.b = np.zeros(self.N)
        self.c = np.zeros(self.M)
        self.mu = np.mean(list(self.user_movie_to_rating_train.values()))

        self.epochs = 25
        # regularization penalty
        self.reg = 0.01

    def get_loss(self, d):
        # given a ratings dictionary d, calculate the loss
        N = float(len(d))
        sum_squared_err = 0
        for k, r in d.items():
            i, j = k
            p = self.W[i].dot(self.U[j]) + self.b[i] + self.c[j] + self.mu
            sum_squared_err += (p - r) * (p - r)
        # mean squared error
        return sum_squared_err/N

    def train(self):
        train_losses = []
        test_losses = []

        for epoch in range(self.epochs):
            print("epoch:", epoch)
            epoch_start = datetime.now()

            # update W and b
            t0 = datetime.now()
            # loop through each user
            for i in range(self.N):
                # W
                # np.eye returns a 2D array with ones on the diagonal and zeros everywhere else
                # matrix variable will accumulate terms which will help calculate Wi
                matrix = np.eye(self.K) * self.reg
                vector = np.zeros(self.K)

                # b
                bi = 0
                # loop through each movie j that user has rated
                for j in self.user_to_movie[i]:
                    r = self.user_movie_to_rating_train[(i, j)]
                    matrix += np.outer(self.U[j], self.U[j])
                    vector += (r - self.b[i] - self.c[j] - self.mu) * self.U[j]
                    bi += (r - self.W[i].dot(self.U[j]) - self.c[j] - self.mu)

                # now update as a result of alternating squares
                self.W[i] = np.linalg.solve(matrix, vector)
                self.b[i] = bi / ((1 + self.reg) * len(self.user_to_movie[i]))

                if i % (self.N//10) == 0:
                    print("i:", i, "N:", self.N)

            print("updated W and b:", datetime.now() - t0)

            # now update U and C
            t0 = datetime.now()
            for j in range(self.M):
                matrix = np.eye(self.K) * self.reg
                vector = np.zeros(self.K)

                # for c
                cj = 0
                try:
                    for i in self.movie_to_user[j]:
                        r = self.user_movie_to_rating_train[(i, j)]
                        matrix += np.outer(self.W[i], self.W[i])
                        vector += (r - self.W[i].dot(self.U[j]) - self.b[i] - self.mu)
                        cj += (r - self.W[i].dot(self.U[j]) - self.b[i] - self.mu)

                    # set updates
                    self.U[j] = np.linalg.solve(matrix, vector)
                    self.c[j] = cj / ((1 + self.reg) * len(self.movie_to_user[j]))

                    if j % (self.M//10) == 0:
                        print("j:", j, "M:", self.M)
                
                except KeyError:
                    pass
            
            print("updated U and c:", datetime.now() - t0)
            print("epoch duration:", datetime.now() - epoch_start)

            # store train loss
            t0 = datetime.now()
            # get loss is where predictions are being made i think
            train_losses.append(self.get_loss(self.user_movie_to_rating_train))

            # store test loss
            test_losses.append(self.get_loss(self.user_movie_to_rating_test))
            print("calculate cost:", datetime.now() - t0)
            print("train loss:", train_losses[-1])
            print("test loss:", test_losses[-1])
        
        print("train losses:", train_losses)
        print("test losses:", test_losses)

    def run(self):
        self.load_data()
        self.initialize_variables()
        self.train()

