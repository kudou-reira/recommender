import pandas as pd
import os.path
from collections import Counter
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import pickle
import urllib

# https://www.kaggle.com/CooperUnion/anime-recommendations-database
# https://github.com/Mayank-Bhatia/Anime-Recommender

movie_zip_path = "/usr/src/app/data/movielens-20m-dataset.zip"
movie_data_path = "/usr/src/app/data/movielens-20m-dataset/"
movie_rating_path = movie_data_path + "rating.csv"
movie_edited_path = movie_data_path + "edited_rating.csv"
movie_truncated_path = movie_data_path + "truncate_rating.csv"
movie_dicts_path = movie_data_path + "dicts/"

user_to_movie_path = movie_dicts_path + 'user_to_movie.json'
movie_to_user_path = movie_dicts_path + 'movie_to_user.json'
user_movie_to_rating_train_path = movie_dicts_path + 'user_movie_to_rating.json'
user_movie_to_rating_test_path = movie_dicts_path + 'user_movie_to_rating_test.json'

def process_movie_data():
    # edit and save the data if it doesn't exist yet
    if not os.path.isfile(movie_edited_path):
        df = pd.read_csv(movie_rating_path)
        print("head", df.head())
        print("tail", df.tail())

        df = df.drop(columns=['timestamp'])
        df.userId = df.userId - 1

        # create mapping for movie ids
        unique_movie_ids = set(df.movieId.values)
        movie_idx = {}
        count = 0

        for movie_id in unique_movie_ids:
            # 1st id = 0, 2nd id = 1, 3rd id = 2, etc
            movie_idx[movie_id] = count
            count += 1

        # add this movie_idx as a dataframe column
        df['movie_idx'] = df.apply(lambda row: movie_idx[row.movieId], axis=1)

        df.to_csv(movie_edited_path)
    
    else:
        print("edited file exists")


def truncate_movie_data():
    if not os.path.isfile(movie_truncated_path):
        df = pd.read_csv(movie_edited_path, index_col=0)
        print("original dataframe size:", len(df))
        print("this is df", df)

        # n is users, m is movies
        N = df.userId.max() + 1
        M = df.movie_idx.max() + 1

        # counter creates a dictionary of counts for each occurence
        user_ids_count = Counter(df.userId)
        movie_ids_count = Counter(df.movie_idx)

        # set number of users and movies we want to preserve for dataset
        n = 10000
        m = 2000

        # get tuples back, only choose first of the tuple for id
        user_ids = [u for u, c in user_ids_count.most_common(n)]
        movie_ids = [m for m, c in movie_ids_count.most_common(m)]

        # select * from dataframe where user_id in user_ids and movie_id in movie_ids
        # make copy of dataframe, returns boolean series for each ids, use & operator to keep
        df_truncated = df[df.userId.isin(user_ids) & df.movie_idx.isin(movie_ids)].copy()
        
        # want user id and movie id to count from (new index, n - 1) and (new index, m - 1)
        # so, have to make new id mapping to old for both user and movie ids
        new_user_id_map = {}
        new_movie_id_map = {}
        i = 0
        j = 0

        for old_id in user_ids:
            new_user_id_map[old_id] = i
            i += 1
        print("max i: ", i)

        for old_id in movie_ids:
            new_movie_id_map[old_id] = j
            j += 1
        print("max j: ", j)

        df_truncated.loc[:, 'userId'] = df_truncated.apply(lambda row: new_user_id_map[row.userId], axis=1)
        df_truncated.loc[:, 'movie_idx'] = df_truncated.apply(lambda row: new_movie_id_map[row.movie_idx], axis=1)

        print("max user id: ", df_truncated.userId.max())
        print("max movie id: ", df_truncated.movie_idx.max())

        print("length of dataframe: ", len(df_truncated))
        df_truncated.to_csv(movie_truncated_path, index=False)

    else:
        print("truncated file exists")

def create_dicts():
    # the point of this function is to preprocess the truncated data into dicts
    # for the sake of reducing computational time, if loop through the dataset, it's O(n2)
    # if made into a dict, it's a hash table 

    if check_file_exists(user_to_movie_path) and check_file_exists(movie_to_user_path) and check_file_exists(user_movie_to_rating_train_path) and check_file_exists(user_movie_to_rating_test_path):
        print("those json files already exist")
        pass

    else:
        print("else block in create dicts")
        df = pd.read_csv(movie_truncated_path)

        # same as above, get number of users and movies
        N = df.userId.max() + 1
        M = df.movie_idx.max() + 1

        # split into train and test set
        df = shuffle(df)
        cutoff_idx = int(0.8 * len(df))
        df_train = df.iloc[:cutoff_idx]
        df_test = df.iloc[cutoff_idx:]

        # now create the dictionaries
        # want 3
        # which users rate which movies
        user_to_movie = {}

        # which movies have been rated by which users
        movie_to_user = {}

        # dictionary of ratings, ex. given user i and movie j, what is the rating?
        user_movie_to_rating = {}

        print("calling: update_user_movie_dicts train")

        # print("this is df_train", df_train)

        # fails on the first try, whatever index that may be

        # creating a nested function because want to use apply on 
        # count = 0
        def update_user_movie_dicts(row):
            # global count
            # print("this is count inside update", count)
            # count += 1
            # if count % 100000 == 0:
            #     print("processed: {}".format(float(count)/cutoff_idx))

            i = int(row.userId)
            j = int(row.movie_idx)

            if i not in user_to_movie:
                # if it doesn't exist, create a list with the current id of movie
                user_to_movie[i] = [j]
            else:
                # if it does exist, append it to the list that should already be there
                user_to_movie[i].append(j) 

            if j not in movie_to_user:
                movie_to_user[j] = [i]
            else:
                movie_to_user[j].append(i)

            # lastly, add to user movie to rating dict with the key being the tuple of the user id and movie id
            user_movie_to_rating[(i, j)] = row.rating
        
        df_train.apply(update_user_movie_dicts, axis=1)

        # now we have to create the test dictionaries
        user_movie_to_rating_test = {}
        print("calling: update_user_movie_dicts test")

        # count = 0
        def update_user_movie_dicts_test(row):
            # global count
            # count += 1
            # if count % 100000 == 0:
            #     print("processed: {}".format(float(count)/len(df_test)))

            i = int(row.userId)
            j = int(row.movie_idx)
            user_movie_to_rating_test[(i, j)] = row.rating
        
        df_test.apply(update_user_movie_dicts_test, axis=1)

        # save these files
        # json can't have tuples of integers as dict key, but use pickle to write as binary

        with open(user_to_movie_path, 'wb') as f:
            pickle.dump(user_to_movie, f)

        with open(movie_to_user_path, 'wb') as f:
            pickle.dump(movie_to_user, f)

        with open(user_movie_to_rating_train_path, 'wb') as f:
            pickle.dump(user_movie_to_rating, f)

        with open(user_movie_to_rating_test_path, 'wb') as f:
            pickle.dump(user_movie_to_rating_test, f)
    

def check_sequential(df):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    numeric_df = df.select_dtypes(include=numerics)
    numeric_df.diff().dropna().eq(1).all()
    print("this is sequential", numeric_df)

def check_file_exists(path_name):
    return os.path.isfile(path_name)

def download_dataset():
    url = 'https://codeload.github.com/fogleman/Minecraft/zip/master'
 
    # downloading with urllib
    # Copy a network object to a local file
    # probably don't want to name this
    urllib.urlretrieve(url, "minemaster.zip")