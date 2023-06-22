# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import tensorflow.keras as keras
from tensorflow.keras import layers
import tensorflow as tf


from recommenders.models.newsrec.models.base_model import BaseModel
from recommenders.models.newsrec.models.layers import AttLayer2, SelfAttention

from keras import backend as K
from keras.layers import Lambda

from keras.utils.np_utils import to_categorical
from keras.layers import *
from keras.models import Model, load_model
from keras import backend as K
from sklearn.metrics import *
from keras.optimizers import *


__all__ = ["NRMSModel"]


useFastFormer = 0

class Fastformer(layers.Layer):

    def __init__(self, nb_head, size_per_head, **kwargs):
        self.nb_head = nb_head
        self.size_per_head = size_per_head
        self.output_dim = nb_head*size_per_head
        self.now_input_shape=None
        super(Fastformer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.now_input_shape=input_shape
        self.WQ = self.add_weight(name='WQ',
                                  shape=(input_shape[0][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.WK = self.add_weight(name='WK',
                                  shape=(input_shape[1][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.Wq = self.add_weight(name='Wq',
                                  shape=(self.output_dim,self.nb_head),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.Wk = self.add_weight(name='Wk',
                                  shape=(self.output_dim,self.nb_head),
                                  initializer='glorot_uniform',
                                  trainable=True)

        self.WP = self.add_weight(name='WP',
                                  shape=(self.output_dim,self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)


        super(Fastformer, self).build(input_shape)

    def call(self, x):
        if len(x) == 2:
            Q_seq,K_seq = x
        elif len(x) == 4:
            Q_seq,K_seq,Q_mask,K_mask = x #different mask lengths, reserved for cross attention

        Q_seq = K.dot(Q_seq, self.WQ)
        Q_seq_reshape = K.reshape(Q_seq, (-1, self.now_input_shape[0][1], self.nb_head*self.size_per_head))

        Q_att=  K.permute_dimensions(K.dot(Q_seq_reshape, self.Wq),(0,2,1))/ self.size_per_head**0.5

        if len(x)  == 4:
            Q_att = Q_att-(1-K.expand_dims(Q_mask,axis=1))*1e8

        Q_att = K.softmax(Q_att)
        Q_seq = K.reshape(Q_seq, (-1,self.now_input_shape[0][1], self.nb_head, self.size_per_head))
        Q_seq = K.permute_dimensions(Q_seq, (0,2,1,3))

        K_seq = K.dot(K_seq, self.WK)
        K_seq = K.reshape(K_seq, (-1,self.now_input_shape[1][1], self.nb_head, self.size_per_head))
        K_seq = K.permute_dimensions(K_seq, (0,2,1,3))

        Q_att = Lambda(lambda x: K.repeat_elements(K.expand_dims(x,axis=3),self.size_per_head,axis=3))(Q_att)
        global_q = K.sum(multiply([Q_att, Q_seq]),axis=2)

        global_q_repeat = Lambda(lambda x: K.repeat_elements(K.expand_dims(x,axis=2), self.now_input_shape[1][1],axis=2))(global_q)

        QK_interaction = multiply([K_seq, global_q_repeat])
        QK_interaction_reshape = K.reshape(QK_interaction, (-1, self.now_input_shape[0][1], self.nb_head*self.size_per_head))
        K_att = K.permute_dimensions(K.dot(QK_interaction_reshape, self.Wk),(0,2,1))/ self.size_per_head**0.5

        if len(x)  == 4:
            K_att = K_att-(1-K.expand_dims(K_mask,axis=1))*1e8

        K_att = K.softmax(K_att)

        K_att = Lambda(lambda x: K.repeat_elements(K.expand_dims(x,axis=3),self.size_per_head,axis=3))(K_att)

        global_k = K.sum(multiply([K_att, QK_interaction]),axis=2)

        global_k_repeat = Lambda(lambda x: K.repeat_elements(K.expand_dims(x,axis=2), self.now_input_shape[0][1],axis=2))(global_k)
        #Q=V
        QKQ_interaction = multiply([global_k_repeat, Q_seq])
        QKQ_interaction = K.permute_dimensions(QKQ_interaction, (0,2,1,3))
        QKQ_interaction = K.reshape(QKQ_interaction, (-1,self.now_input_shape[0][1], self.nb_head*self.size_per_head))
        QKQ_interaction = K.dot(QKQ_interaction, self.WP)
        QKQ_interaction = K.reshape(QKQ_interaction, (-1,self.now_input_shape[0][1], self.nb_head,self.size_per_head))
        QKQ_interaction = K.permute_dimensions(QKQ_interaction, (0,2,1,3))
        QKQ_interaction = QKQ_interaction+Q_seq
        QKQ_interaction = K.permute_dimensions(QKQ_interaction, (0,2,1,3))
        QKQ_interaction = K.reshape(QKQ_interaction, (-1,self.now_input_shape[0][1], self.nb_head*self.size_per_head))

        #many operations can be optimized if higher versions are used.

        return QKQ_interaction

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], self.output_dim)


class NRMSModel(BaseModel):
    """NRMS model(Neural News Recommendation with Multi-Head Self-Attention)

    Chuhan Wu, Fangzhao Wu, Suyu Ge, Tao Qi, Yongfeng Huang,and Xing Xie, "Neural News
    Recommendation with Multi-Head Self-Attention" in Proceedings of the 2019 Conference
    on Empirical Methods in Natural Language Processing and the 9th International Joint Conference
    on Natural Language Processing (EMNLP-IJCNLP)

    Attributes:
        word2vec_embedding (numpy.ndarray): Pretrained word embedding matrix.
        hparam (object): Global hyper-parameters.
    """

    def __init__(
        self,
        hparams,
        iterator_creator,
        seed=None,
    ):
        """Initialization steps for NRMS.
        Compared with the BaseModel, NRMS need word embedding.
        After creating word embedding matrix, BaseModel's __init__ method will be called.

        Args:
            hparams (object): Global hyper-parameters. Some key setttings such as head_num and head_dim are there.
            iterator_creator_train (object): NRMS data loader class for train data.
            iterator_creator_test (object): NRMS data loader class for test and validation data
        """
        self.word2vec_embedding = self._init_embedding(hparams.wordEmb_file)

        super().__init__(
            hparams,
            iterator_creator,
            seed=seed,
        )

    def _get_input_label_from_iter(self, batch_data):
        """get input and labels for trainning from iterator

        Args:
            batch data: input batch data from iterator

        Returns:
            list: input feature fed into model (clicked_title_batch & candidate_title_batch)
            numpy.ndarray: labels
        """
        input_feat = [
            batch_data["clicked_title_batch"],
            batch_data["candidate_title_batch"],
        ]
        input_label = batch_data["labels"]
        return input_feat, input_label

    def _get_user_feature_from_iter(self, batch_data):
        """get input of user encoder
        Args:
            batch_data: input batch data from user iterator

        Returns:
            numpy.ndarray: input user feature (clicked title batch)
        """
        return batch_data["clicked_title_batch"]

    def _get_news_feature_from_iter(self, batch_data):
        """get input of news encoder
        Args:
            batch_data: input batch data from news iterator

        Returns:
            numpy.ndarray: input news feature (candidate title batch)
        """
        return batch_data["candidate_title_batch"]

    def _build_graph(self):
        """Build NRMS model and scorer.

        Returns:
            object: a model used to train.
            object: a model used to evaluate and inference.
        """
        model, scorer = self._build_nrms()
        return model, scorer

    def _build_userencoder(self, titleencoder):
        """The main function to create user encoder of NRMS.

        Args:
            titleencoder (object): the news encoder of NRMS.

        Return:
            object: the user encoder of NRMS.
        """
        hparams = self.hparams
        his_input_title = keras.Input(
            shape=(hparams.his_size, hparams.title_size), dtype="int32"
        )



        click_title_presents = layers.TimeDistributed(titleencoder)(his_input_title)

        if (useFastFormer == 22):
            print('use faster former for user encoder')
            y = Fastformer(20,20)([click_title_presents,click_title_presents,qmask,qmask])
        else:
            y = SelfAttention(hparams.head_num, hparams.head_dim, seed=self.seed)(
                [click_title_presents] * 3
            )

        user_present = AttLayer2(hparams.attention_hidden_dim, seed=self.seed)(y)

        model = keras.Model(his_input_title, user_present, name="user_encoder")
        print(model.summary())
        return model

    def _build_newsencoder(self, embedding_layer):




        """The main function to create news encoder of NRMS.

        Args:
            embedding_layer (object): a word embedding layer.

        Return:
            object: the news encoder of NRMS.
        """
        hparams = self.hparams
        sequences_input_title = keras.Input(shape=(hparams.title_size,), dtype="int32")


        qmask=Lambda(lambda x:  K.cast(K.cast(x,'bool'),'float32'))(sequences_input_title)

        embedded_sequences_title = embedding_layer(sequences_input_title)
        print("embedded_sequences_title.shape")
        print(embedded_sequences_title.shape)

        y = layers.Dropout(hparams.dropout)(embedded_sequences_title)

        print('looks good')


        print('Use Fast Former: ')
        print(str(useFastFormer))

        if (useFastFormer == 1):
            # This one doesn't work'
            mask = tf.ones([1, 300], dtype=tf.bool)
            # x = tf.experimental.numpy.random.randint(0, 20000, (1, 4096))
            # fast_former_layer = model(x)

            y = model(y)
        elif (useFastFormer == 2):
            print('use fast former')
            y = Fastformer(20,20)([y,y,qmask,qmask])
        else:
            y = SelfAttention(hparams.head_num, hparams.head_dim, seed=self.seed)([y, y, y])
            print('use nrms')

        y = layers.Dropout(hparams.dropout)(y)
        pred_title = AttLayer2(hparams.attention_hidden_dim, seed=self.seed)(y)

        model = keras.Model(sequences_input_title, pred_title, name="news_encoder")
        print(model.summary())
        return model

    def _build_nrms(self):
        """The main function to create NRMS's logic. The core of NRMS
        is a user encoder and a news encoder.

        Returns:
            object: a model used to train.
            object: a model used to evaluate and inference.
        """
        hparams = self.hparams

        his_input_title = keras.Input(
            shape=(hparams.his_size, hparams.title_size), dtype="int32"
        )
        pred_input_title = keras.Input(
            shape=(hparams.npratio + 1, hparams.title_size), dtype="int32"
        )
        pred_input_title_one = keras.Input(
            shape=(
                1,
                hparams.title_size,
            ),
            dtype="int32",
        )
        pred_title_one_reshape = layers.Reshape((hparams.title_size,))(
            pred_input_title_one
        )

        embedding_layer = layers.Embedding(
            self.word2vec_embedding.shape[0],
            hparams.word_emb_dim,
            weights=[self.word2vec_embedding],
            trainable=True,
        )

        titleencoder = self._build_newsencoder(embedding_layer)
        self.userencoder = self._build_userencoder(titleencoder)
        self.newsencoder = titleencoder

        user_present = self.userencoder(his_input_title)
        news_present = layers.TimeDistributed(self.newsencoder)(pred_input_title)
        news_present_one = self.newsencoder(pred_title_one_reshape)

        preds = layers.Dot(axes=-1)([news_present, user_present])
        preds = layers.Activation(activation="softmax")(preds)

        pred_one = layers.Dot(axes=-1)([news_present_one, user_present])
        pred_one = layers.Activation(activation="sigmoid")(pred_one)

        model = keras.Model([his_input_title, pred_input_title], preds)
        scorer = keras.Model([his_input_title, pred_input_title_one], pred_one)

        return model, scorer
