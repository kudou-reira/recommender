import tensorflow as tf
from utils.utils import process_movie_data, truncate_movie_data, create_dicts
from filtering.user_based import user_based_filtering
from matrix_factorization.mf_np import Mfnp
from matrix_factorization.mf_keras import Mfkeras

process_movie_data()
truncate_movie_data()
create_dicts()

# if you want to try out user_based_filtering, uncomment the below
# user_based_filtering()

# if you want to try out the theoretical np implementation of matrix factorization, uncomment the below
# mf = Mfnp()
# mf.run()

mf = Mfkeras()
mf.create_and_train_model()

