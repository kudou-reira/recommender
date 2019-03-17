import pickle
import numpy as np
import pandas as pd
from sklearn.utils import shuffle

from keras.models import Model, load_model, model_from_json
from keras.layers import Input, Embedding, Dot, Add, Flatten
from keras.regularizers import l2
from keras.optimizers import SGD, Adam

movie_data_path = "/usr/src/app/data/movielens-20m-dataset/"
movie_edited_path = movie_data_path + "edited_rating.csv"
loss_plot_path = movie_data_path + "loss_plot.png"
mse_plot_path = movie_data_path + "mse_plot.png"

model_path = "/usr/src/app/data/matrix_factorization/models/"

# implementation of matrix factorization with keras
class Mfkeras():
    def __init__(self):
        self.df = pd.read_csv(movie_edited_path)
        self.N = self.df.userId.max() + 1
        self.M = self.df.movie_idx.max() + 1
        
        self.u = 0
        self.m = 0

        self.K = 0
        # global moving average
        self.mu = 0
        self.epochs = 0
        # regularization penalty
        self.reg = 0

    def run(self):
        print("this is self.N", self.N)
        print("this is self.M", self.M)

    def create_and_train_model(self):
        self.df = shuffle(self.df)

        cutoff = int(0.8 * len(self.df))
        df_train = self.df.iloc[:cutoff]
        df_test = self.df.iloc[cutoff:]

        # initialize variables
        self.K = 10
        self.mu = df_train.rating.mean()
        self.epochs = 25
        self.reg = 0

        u = Input(shape=(1,))
        m = Input(shape=(1,))

        # will be batch size x 1 x K size because embeddings expect a sequence
        u_embedding = Embedding(self.N, self.K, embeddings_regularizer=l2(self.reg))(u)
        m_embedding = Embedding(self.M, self.K, embeddings_regularizer=l2(self.reg))(m)

        # bias shapes, N x 1 and M x 1
        u_bias = Embedding(self.N, 1, embeddings_regularizer=l2(self.reg))(u)
        m_bias = Embedding(self.M, 1, embeddings_regularizer=l2(self.reg))(m)
        
        # sum over k-size axis, which is 2 because remember shape is batch_size x 1 x K
        x = Dot(axes=2)([u_embedding, m_embedding])

        x = Add()([x, u_bias, m_bias])
        x = Flatten()(x)

        # add mse as a metrics because if you include regularization, there is a problem where the regularization penalty will be added to the mse
        model = Model(inputs=[u, m], outputs=x)
        model.compile(
            loss='mse',
            optimizer=SGD(lr=0.01, momentum=0.9),
            metrics=['mse']
        )

        r = model.fit(
            x=[df_train.userId.values, df_train.movie_idx.values],
            y=[df_train.rating.values - self.mu],
            epochs=self.epochs,
            batch_size=128,
            validation_data=(
                [df_test.userId.values, df_test.movie_idx.values],
                df_test.rating.values - self.mu
            )
        )

        # https://stackoverflow.com/questions/42763094/how-to-save-final-model-using-keras
        model_json = model.to_json()

        with open(model_path + "model_num.json", "w") as json_file:
            json_file.write(model_json)

        model.save_weights(model_path + 'model_num.h5')

        # plot losses
        plt.plot(r.history['loss'], label="train_loss")
        plt.plot(r.history['val_loss'], label="test_loss")
        plt.legend()
        plt.savefig(loss_plot_path)

        # plot mse
        plt.plot(r.history['mean_squared_error'], label="train mse")
        plt.plot(r.history['val_mean_squared_error'], label="test mse")
        plt.legend()
        plt.savefig(mse_plot_path)