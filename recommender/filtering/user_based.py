import pickle
import numpy as np
from sortedcontainers import SortedList

movie_data_path = "/usr/src/app/data/movielens-20m-dataset/"
movie_dicts_path = movie_data_path + "dicts/"
user_to_movie_path = movie_dicts_path + 'user_to_movie.json'
movie_to_user_path = movie_dicts_path + 'movie_to_user.json'
user_movie_to_rating_train_path = movie_dicts_path + 'user_movie_to_rating.json'
user_movie_to_rating_test_path = movie_dicts_path + 'user_movie_to_rating_test.json'

def user_based_filtering():
    with open(user_to_movie_path, 'rb') as f:
        user_to_movie = pickle.load(f)

    with open(movie_to_user_path, 'rb') as f:
        movie_to_user = pickle.load(f)

    with open(user_movie_to_rating_train_path, 'rb') as f:
        user_movie_to_rating_train = pickle.load(f)

    with open(user_movie_to_rating_test_path, 'rb') as f:
        user_movie_to_rating_test = pickle.load(f)

    # find max number of users
    N = np.max(list(user_to_movie.keys())) + 1

    # movies might appear in the test set, but not the train set, check that here
    # we want to make sure that we get ALL the movie ids
    # movie to user is empty
    # print("this is movie to user keys", list(movie_to_user.keys()))

    m1 = np.max(list(movie_to_user.keys()))
    m2 = np.max([m for (u, m), r in user_movie_to_rating_test.items()])
    M = max(m1, m2) + 1

    print("N:", N, "M:", M)

    if N > 10000:
        print("this is a really big quadratic run time, exiting!")

    # find user similarities
    # first, find user's average rating
    # then, for each user i, find close neighbors that have already rated movie j: get prediction for user i, or rate movie j
    # to know which neighbors to choose, calculate all the weights between all the users, that's the O(n2) calculation runtime
    # only want to keep the highest weights, so have to find all weights first, then sort them in decreasing order
    # also, store deviations of how much better a user's rating for a movie j is compared to his/her average rating

    # number of neighbors to look for
    K = 25

    # number of movies that users need to have in common to be considered for filtering
    # if the number of movies that i and i' have in common are too few or 0, there is a low correlation, thus NG
    limit = 5

    # list of neighbors for each user
    neighbors = []

    # list of each user's average rating
    averages = []

    # list of deviations for each user
    deviations = []

    for i in range(N):
        # find the closest K users to user i
        movies_i = user_to_movie[i]

        # get unique movies
        movies_i_set = set(movies_i)

        # calculate average and deviation
        # want a movies to rating dictionary
        # remember this is a dictionary of ratings, ex. given user i and movie j, what is the rating?
        ratings_i = {movie:user_movie_to_rating_train[(i, movie)] for movie in movies_i}

        # now calculate users average rating
        avg_i = np.mean(list(ratings_i.values()))
        
        # once average is known, deviations can be calculated
        # this gives a dict of movie ids as key and deviations as value
        dev_i = {movie:(rating - avg_i) for movie, rating in ratings_i.items()}

        # for computational reasons, convert this dict into a numpy array
        dev_i_values = np.array(list(dev_i.values()))

        # the denominator in the pearson correlation is the square root of the sum of squares of the deviations
        # denote that as sigma_i
        sigma_i = np.sqrt(dev_i_values.dot(dev_i_values))

        # save these values
        averages.append(avg_i)
        deviations.append(dev_i)

        # create a sorted list to keep track of all users calculated so far
        sl = SortedList()

        # for each other user besides the current user, calculate the user-user weight, then add it to this list
        for j in range(N):
            # don't include current user, no user is a neighbor to themselves!
            if j != i:
                # basically do the same thing as user i for user j aka neighbor, but if we do that, we compute the user i and j weights twice
                # an improvement here could be to store previous user in memory, but then we have to use up memory, storage vs. memory tradeoff
                movies_j = user_to_movie[j]
                # the set of user movies we created earlier is now useful as we can find the set including user i and j movies
                movies_j_set = set(movies_j)
                common_movies = (movies_i_set & movies_j_set)
                if len(common_movies) > limit:
                    # set a new dict with key being movie id and the value being all of the users rating for that movie that user j has rated
                    ratings_j = {movie: user_movie_to_rating_train[(j, movie)] for movie in movies_j}
                    avg_j = np.mean(list(ratings_j.values()))
                    dev_j = {movie:(rating - avg_j) for movie, rating in ratings_j.items()}
                    dev_j_values = np.array(list(dev_j.values()))
                    sigma_j = np.sqrt(dev_j_values.dot(dev_j_values))

                    # calculate correlation
                    numerator = sum(dev_i[m] * dev_j[m] for m in common_movies)
                    w_ij = numerator/(sigma_i + sigma_j)

                    # insert into sorted list
                    # notable gotcha is that sorted list sorts in ascending order
                    # pearson coefficient with higher value means higher level of correlation
                    # ex. maximum value (1) represents highest correlation
                    # thus, store value as negative because we want the most correlated at the fron of the list
                    sl.add((-w_ij, j))

                    # we want to keep the size of the list less than K neighbors
                    if len(sl) > K:
                        del sl[-1]

        neighbors.append(sl)

        if i % 1 == 0:
            print("this is i", i)
    



    train_predictions = []
    train_targets = []
    test_prediction = []
    test_targets.append(target)


    for (i, m), target in user_movie_to_rating_train.items():
        prediction = predict(i, m, neighbors)
        # save
        train_predictions.append(prediction)
        train_targets.append(target)
    

    for (i, m), target in user_movie_to_rating_test.items():
        prediction = predict(i, m, neighbors)
        # save
        test_predictions.append(prediction)
        test_targets.append(target)
    

    print('train mse:', mse(train_predictions, train_targets))
    print('test mse:', mse(test_predictions, test_targets))


def predict(i, m, neighbors):
    numerator = 0
    denominator = 0

    for neg_w, j in neighbors[i]:
        # the weight is stored negative, so turn it positive for the calculation
        try:
            numerator += -neg_w * deviations[j][m]
            denominator += abs(neg_w)
        except KeyError:
            # if neighbor didn't rated the same movie
            pass

    if denominator == 0:
        # in the case that the denominator of pearson calculation is 0, just average it
        prediction = averages[i]

    else:
        prediction = numerator/denominator + averages[i]

    # bound the values since you know rating is 0-5
    prediction = min(5, prediction)
    prediction = max(0.5, prediction)
    return prediction


def mse(p, t):
    p = np.array(p)
    t = np.array(t)
    return np.mean((p - t)**2)