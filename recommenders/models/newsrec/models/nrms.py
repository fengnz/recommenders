# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import tensorflow.keras as keras
from tensorflow.keras import layers
import tensorflow as tf
tf.compat.v1.experimental.output_all_intermediates(True)
import numpy as np


from recommenders.models.newsrec.models.base_model import BaseModel
from recommenders.models.newsrec.models.layers import AttLayer2, SelfAttention
#from fast_transformer import FastTransformer

from keras import backend as K
from keras.layers import Lambda

#from keras.utils.np_utils import to_categorical
from keras.layers import *
from keras.models import Model, load_model
from keras import backend as K
from sklearn.metrics import *
from keras.optimizers import *


__all__ = ["NRMSModel"]

import os
import shutil

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
#from official.nlp import optimization  # to create AdamW optimizer

import matplotlib.pyplot as plt

tf.get_logger().setLevel('ERROR')

#@title Choose a BERT model to fine-tune

bert_model_name = 'small_bert/bert_en_uncased_L-4_H-512_A-8'  #@param ["bert_en_uncased_L-12_H-768_A-12", "bert_en_cased_L-12_H-768_A-12", "bert_multi_cased_L-12_H-768_A-12", "small_bert/bert_en_uncased_L-2_H-128_A-2", "small_bert/bert_en_uncased_L-2_H-256_A-4", "small_bert/bert_en_uncased_L-2_H-512_A-8", "small_bert/bert_en_uncased_L-2_H-768_A-12", "small_bert/bert_en_uncased_L-4_H-128_A-2", "small_bert/bert_en_uncased_L-4_H-256_A-4", "small_bert/bert_en_uncased_L-4_H-512_A-8", "small_bert/bert_en_uncased_L-4_H-768_A-12", "small_bert/bert_en_uncased_L-6_H-128_A-2", "small_bert/bert_en_uncased_L-6_H-256_A-4", "small_bert/bert_en_uncased_L-6_H-512_A-8", "small_bert/bert_en_uncased_L-6_H-768_A-12", "small_bert/bert_en_uncased_L-8_H-128_A-2", "small_bert/bert_en_uncased_L-8_H-256_A-4", "small_bert/bert_en_uncased_L-8_H-512_A-8", "small_bert/bert_en_uncased_L-8_H-768_A-12", "small_bert/bert_en_uncased_L-10_H-128_A-2", "small_bert/bert_en_uncased_L-10_H-256_A-4", "small_bert/bert_en_uncased_L-10_H-512_A-8", "small_bert/bert_en_uncased_L-10_H-768_A-12", "small_bert/bert_en_uncased_L-12_H-128_A-2", "small_bert/bert_en_uncased_L-12_H-256_A-4", "small_bert/bert_en_uncased_L-12_H-512_A-8", "small_bert/bert_en_uncased_L-12_H-768_A-12", "albert_en_base", "electra_small", "electra_base", "experts_pubmed", "experts_wiki_books", "talking-heads_base"]

map_name_to_handle = {
    'bert_en_uncased_L-12_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3',
    'bert_en_cased_L-12_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_en_cased_L-12_H-768_A-12/3',
    'bert_multi_cased_L-12_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_multi_cased_L-12_H-768_A-12/3',
    'small_bert/bert_en_uncased_L-2_H-128_A-2':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-128_A-2/1',
    'small_bert/bert_en_uncased_L-2_H-256_A-4':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-256_A-4/1',
    'small_bert/bert_en_uncased_L-2_H-512_A-8':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-512_A-8/1',
    'small_bert/bert_en_uncased_L-2_H-768_A-12':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-768_A-12/1',
    'small_bert/bert_en_uncased_L-4_H-128_A-2':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-128_A-2/1',
    'small_bert/bert_en_uncased_L-4_H-256_A-4':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-256_A-4/1',
    'small_bert/bert_en_uncased_L-4_H-512_A-8':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/1',
    'small_bert/bert_en_uncased_L-4_H-768_A-12':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-768_A-12/1',
    'small_bert/bert_en_uncased_L-6_H-128_A-2':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-128_A-2/1',
    'small_bert/bert_en_uncased_L-6_H-256_A-4':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-256_A-4/1',
    'small_bert/bert_en_uncased_L-6_H-512_A-8':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-512_A-8/1',
    'small_bert/bert_en_uncased_L-6_H-768_A-12':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-768_A-12/1',
    'small_bert/bert_en_uncased_L-8_H-128_A-2':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-128_A-2/1',
    'small_bert/bert_en_uncased_L-8_H-256_A-4':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-256_A-4/1',
    'small_bert/bert_en_uncased_L-8_H-512_A-8':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-512_A-8/1',
    'small_bert/bert_en_uncased_L-8_H-768_A-12':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-768_A-12/1',
    'small_bert/bert_en_uncased_L-10_H-128_A-2':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-128_A-2/1',
    'small_bert/bert_en_uncased_L-10_H-256_A-4':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-256_A-4/1',
    'small_bert/bert_en_uncased_L-10_H-512_A-8':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-512_A-8/1',
    'small_bert/bert_en_uncased_L-10_H-768_A-12':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-768_A-12/1',
    'small_bert/bert_en_uncased_L-12_H-128_A-2':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-128_A-2/1',
    'small_bert/bert_en_uncased_L-12_H-256_A-4':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-256_A-4/1',
    'small_bert/bert_en_uncased_L-12_H-512_A-8':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-512_A-8/1',
    'small_bert/bert_en_uncased_L-12_H-768_A-12':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-768_A-12/1',
    'albert_en_base':
        'https://tfhub.dev/tensorflow/albert_en_base/2',
    'electra_small':
        'https://tfhub.dev/google/electra_small/2',
    'electra_base':
        'https://tfhub.dev/google/electra_base/2',
    'experts_pubmed':
        'https://tfhub.dev/google/experts/bert/pubmed/2',
    'experts_wiki_books':
        'https://tfhub.dev/google/experts/bert/wiki_books/2',
    'talking-heads_base':
        'https://tfhub.dev/tensorflow/talkheads_ggelu_bert_en_base/1',
}

map_model_to_preprocess = {
    'bert_en_uncased_L-12_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'bert_en_cased_L-12_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_en_cased_preprocess/3',
    'small_bert/bert_en_uncased_L-2_H-128_A-2':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-2_H-256_A-4':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-2_H-512_A-8':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-2_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-4_H-128_A-2':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-4_H-256_A-4':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-4_H-512_A-8':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-4_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-6_H-128_A-2':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-6_H-256_A-4':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-6_H-512_A-8':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-6_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-8_H-128_A-2':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-8_H-256_A-4':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-8_H-512_A-8':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-8_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-10_H-128_A-2':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-10_H-256_A-4':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-10_H-512_A-8':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-10_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-12_H-128_A-2':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-12_H-256_A-4':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-12_H-512_A-8':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-12_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'bert_multi_cased_L-12_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_multi_cased_preprocess/3',
    'albert_en_base':
        'https://tfhub.dev/tensorflow/albert_en_preprocess/3',
    'electra_small':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'electra_base':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'experts_pubmed':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'experts_wiki_books':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'talking-heads_base':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
}

tfhub_handle_encoder = map_name_to_handle[bert_model_name]
tfhub_handle_preprocess = map_model_to_preprocess[bert_model_name]

print(f'BERT model selected           : {tfhub_handle_encoder}')
print(f'Preprocess model auto-selected: {tfhub_handle_preprocess}')


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
            batch_data["clicked_title_bert_batch"],
            batch_data["candidate_title_bert_batch"],
            # np.asarray(batch_data["clicked_title_string_batch"]),
            # np.asarray(batch_data["candidate_title_string_batch"]),
        ]

        # print(batch_data["clicked_title_bert_batch"])
        input_label = batch_data["labels"]
        return input_feat, input_label

    def _get_user_feature_from_iter(self, batch_data):
        """get input of user encoder
        Args:
            batch_data: input batch data from user iterator

        Returns:
            numpy.ndarray: input user feature (clicked title batch)
        """
        # return batch_data["clicked_title_string_batch"]
        return batch_data["clicked_title_bert_batch"]

    def _get_news_feature_from_iter(self, batch_data):
        """get input of news encoder
        Args:
            batch_data: input batch data from news iterator

        Returns:
            numpy.ndarray: input news feature (candidate title batch)
        """
        return batch_data["candidate_title_bert_batch"]

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
        #his_input_title = keras.Input(
        #    shape=(hparams.his_size, hparams.title_size), dtype="int32"
        #)

        his_input_title_bert = keras.Input(
        #   shape=(hparams.his_size, hparams.deberta_states_num, 1537), dtype="float32"
            shape=(hparams.his_size,  hparams.deberta_states_num , 1537), dtype="float32"
        )

        # his_input_string_title = keras.Input(
        #     shape=(hparams.his_size,), dtype=tf.string
        # )
        #
        # reshaped_his_input_string_title = tf.expand_dims(his_input_string_title, axis=-1)

        click_title_presents = layers.TimeDistributed(titleencoder)(his_input_title_bert)
        y = SelfAttention(hparams.head_num, hparams.head_dim, seed=self.seed)(
            [click_title_presents] * 3
        )
        user_present = AttLayer2(hparams.attention_hidden_dim, seed=self.seed)(y)

        model = keras.Model(his_input_title_bert, user_present, name="user_encoder")
        model.summary()
        return model


    def _build_news_title_encoder(self, title_encoder, bert_title_encoder):
        hparams = self.hparams
        concated_sequences_input_title = keras.Input(shape=(hparams.deberta_states_num, 1537,), dtype="float32", name="news_title_bert_input")
        y = bert_title_encoder(concated_sequences_input_title)

        model = keras.Model(concated_sequences_input_title, y, name="news_encoder")
        print(model.summary())
        return model


    def _build_bert_newsencoder(self):

        """The main function to create news encoder of NRMS.

        Args:
            embedding_layer (object): a word embedding layer.

        Return:
            object: the news encoder of NRMS.
        """
        hparams = self.hparams
        #sequences_input_title = keras.Input(shape=(1536,), dtype="float32", name="news_title_bert_input")
        concated_sequences_input_title = keras.Input(shape=(hparams.deberta_states_num, 1537,), dtype="float32", name="news_title_bert_input")

        # split the input masks back

        input_mask = concated_sequences_input_title[:, :, 1536:]
        input_mask = tf.squeeze(input_mask, axis=-1)
        sequences_input_title = concated_sequences_input_title[:, :, :1536]

        # try, take the first 30 only
        def take_first_30(x):
            return x[:, :30]


        chain_bert = False

        if chain_bert:
            sequences_input_string_title = keras.Input(shape=(), dtype=tf.string, name="news_title_input")
            preprocessing_layer = hub.KerasLayer(tfhub_handle_preprocess, name='preprocessing')
            encoder_inputs = preprocessing_layer(sequences_input_string_title)
            encoder = hub.KerasLayer(tfhub_handle_encoder, trainable=True, name='BERT_encoder')
            outputs = encoder(encoder_inputs)
            y = outputs['pooled_output']
            y = tf.keras.layers.Dropout(0.1)(y)

        else:
            # the title has been processed by bert, do nothing, just return
            # convert int 32 to float 32
            #y = tf.keras.layers.Lambda(lambda x: tf.cast(x, 'float32'))(y)
            #y = tf.keras.layers.Lambda(take_first_30)(sequences_input_title)

            # reduced_sequences_input_title = tf.keras.layers.Dense(30)(sequences_input_title)
            # y = tf.expand_dims(reduced_sequences_input_title, axis=-1)
            reduced_sequences_input_title = sequences_input_title

            use_fast_former = True
            if use_fast_former:
                # qmask=Lambda(lambda x:  K.cast(K.cast(x,'bool'),'float32'))(reduced_sequences_input_title)
                qmask = input_mask
                y = sequences_input_title
                y = Fastformer(hparams.head_num,hparams.head_dim)([y, y, qmask, qmask])
            else:
                y = sequences_input_title
                y = SelfAttention(hparams.head_num, hparams.head_dim, seed=self.seed)([y, y, y])


        y = layers.Dropout(hparams.dropout)(y)
        y = AttLayer2(hparams.attention_hidden_dim, seed=self.seed)(y)

        #sequences_input_title = net

        #qmask=Lambda(lambda x:  K.cast(K.cast(x,'bool'),'float32'))(sequences_input_title)

        #embedded_sequences_title = embedding_layer(sequences_input_title)
        #print("embedded_sequences_title.shape")
        #print(embedded_sequences_title.shape)

        #y = layers.Dropout(hparams.dropout)(embedded_sequences_title)

        #print('looks good')

        #useFastFormer = 2

        #print('Use Fast Former: ')
        #print(str(useFastFormer))

        #if (useFastFormer == 1):
        #    # This one doesn't work'
        #    mask = tf.ones([1, 300], dtype=tf.bool)
        #    model = FastTransformer(
        #        num_tokens = hparams.head_num,
        #        dim = hparams.head_dim,
        #        depth = 2,
        #        max_seq_len = 300,
        #        absolute_pos_emb = None, # Absolute positional embeddings
        #        mask = mask
        #    )
        #    # x = tf.experimental.numpy.random.randint(0, 20000, (1, 4096))
        #    # fast_former_layer = model(x)

        #    y = model(y)
        #elif (useFastFormer == 2):
        #    y = Fastformer(20,20)([y,y,qmask,qmask])
        #else:
        #    y = SelfAttention(hparams.head_num, hparams.head_dim, seed=self.seed)([y, y, y])

        #y = layers.Dropout(hparams.dropout)(y)
        #pred_title = AttLayer2(hparams.attention_hidden_dim, seed=self.seed)(y)

        model = keras.Model(concated_sequences_input_title, y, name="news_encoder")
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
        #sequences_input_title = keras.Input(shape=(1536,), dtype="float32", name="news_title_bert_input")
        concated_sequences_input_title = keras.Input(shape=(hparams.deberta_states_num, 1537,), dtype="float32", name="news_title_bert_input")

        # split the input masks back

        input_mask = concated_sequences_input_title[:, :, 1536:]
        input_mask = tf.squeeze(input_mask, axis=-1)
        sequences_input_title = concated_sequences_input_title[:, :, :1536]

        # try, take the first 30 only
        def take_first_30(x):
            return x[:, :30]


        chain_bert = False

        if chain_bert:
            sequences_input_string_title = keras.Input(shape=(), dtype=tf.string, name="news_title_input")
            preprocessing_layer = hub.KerasLayer(tfhub_handle_preprocess, name='preprocessing')
            encoder_inputs = preprocessing_layer(sequences_input_string_title)
            encoder = hub.KerasLayer(tfhub_handle_encoder, trainable=True, name='BERT_encoder')
            outputs = encoder(encoder_inputs)
            y = outputs['pooled_output']
            y = tf.keras.layers.Dropout(0.1)(y)

        else:
            # the title has been processed by bert, do nothing, just return
            # convert int 32 to float 32
            #y = tf.keras.layers.Lambda(lambda x: tf.cast(x, 'float32'))(y)
            #y = tf.keras.layers.Lambda(take_first_30)(sequences_input_title)

            # reduced_sequences_input_title = tf.keras.layers.Dense(30)(sequences_input_title)
            # y = tf.expand_dims(reduced_sequences_input_title, axis=-1)
            reduced_sequences_input_title = sequences_input_title

            use_fast_former = True
            if use_fast_former:
                # qmask=Lambda(lambda x:  K.cast(K.cast(x,'bool'),'float32'))(reduced_sequences_input_title)
                qmask = input_mask
                y = sequences_input_title
                y = Fastformer(hparams.head_num,hparams.head_dim)([y, y, qmask, qmask])
            else:
                y = sequences_input_title
                y = SelfAttention(hparams.head_num, hparams.head_dim, seed=self.seed)([y, y, y])


        y = layers.Dropout(hparams.dropout)(y)
        y = AttLayer2(hparams.attention_hidden_dim, seed=self.seed)(y)

        #sequences_input_title = net

        #qmask=Lambda(lambda x:  K.cast(K.cast(x,'bool'),'float32'))(sequences_input_title)

        #embedded_sequences_title = embedding_layer(sequences_input_title)
        #print("embedded_sequences_title.shape")
        #print(embedded_sequences_title.shape)

        #y = layers.Dropout(hparams.dropout)(embedded_sequences_title)

        #print('looks good')

        #useFastFormer = 2

        #print('Use Fast Former: ')
        #print(str(useFastFormer))

        #if (useFastFormer == 1):
        #    # This one doesn't work'
        #    mask = tf.ones([1, 300], dtype=tf.bool)
        #    model = FastTransformer(
        #        num_tokens = hparams.head_num,
        #        dim = hparams.head_dim,
        #        depth = 2,
        #        max_seq_len = 300,
        #        absolute_pos_emb = None, # Absolute positional embeddings
        #        mask = mask
        #    )
        #    # x = tf.experimental.numpy.random.randint(0, 20000, (1, 4096))
        #    # fast_former_layer = model(x)

        #    y = model(y)
        #elif (useFastFormer == 2):
        #    y = Fastformer(20,20)([y,y,qmask,qmask])
        #else:
        #    y = SelfAttention(hparams.head_num, hparams.head_dim, seed=self.seed)([y, y, y])

        #y = layers.Dropout(hparams.dropout)(y)
        #pred_title = AttLayer2(hparams.attention_hidden_dim, seed=self.seed)(y)

        model = keras.Model(concated_sequences_input_title, y, name="news_encoder")
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

        # his_input_title_bert = keras.Input(
        #     shape=(hparams.his_size, 1536), dtype="float32"
        # )
        his_input_title_bert = keras.Input(
            shape=(hparams.his_size, hparams.deberta_states_num, 1537), dtype="float32"

        )
        pred_input_title_bert = keras.Input(
            shape=(hparams.npratio + 1, hparams.deberta_states_num, 1537), dtype="float32"
        )

        # his_input_string_title= tf.keras.layers.Input(shape=(hparams.his_size,), dtype=tf.string, name='his_input_text')
        # pred_input_string_title= tf.keras.layers.Input(shape=(hparams.npratio+1,), dtype=tf.string, name='pre_input_text')

        pred_input_title_one = keras.Input(
            shape=(
                1,
                hparams.title_size,
            ),
            dtype="int32",
        )

        # pred_input_title_bert_one = keras.Input(
        #     shape=(
        #         1,
        #         1536,
        #     ),
        #     dtype="float32",
        # )

        pred_input_title_bert_one = keras.Input(
            shape=(
                1,
                hparams.deberta_states_num,
                1537,
            ),
            dtype="float32",
        )

        pred_input_title_string_one = keras.Input(
            shape=(
                1,
            ),
            dtype=tf.string,
        )
        pred_title_one_reshape = layers.Reshape((hparams.title_size,))(
            pred_input_title_one
        )

        pred_title_bert_one_reshape = layers.Reshape((hparams.deberta_states_num, 1537,))(
            pred_input_title_bert_one
        )

        pred_string_title_one_reshape = layers.Reshape((1,))(
            pred_input_title_string_one
        )

        embedding_layer = layers.Embedding(
            self.word2vec_embedding.shape[0],
            hparams.word_emb_dim,
            weights=[self.word2vec_embedding],
            trainable=True,
        )

        bert_title_encoder = self._build_bert_newsencoder()

        news_encoder = self._build_news_title_encoder(bert_title_encoder, bert_title_encoder)

        self.userencoder = self._build_userencoder(news_encoder)
        self.newsencoder = news_encoder

        # reshaped_pred_input_string_title = tf.expand_dims(pred_input_string_title, axis=-1)

        user_present = self.userencoder(his_input_title_bert)
        # news_present = layers.TimeDistributed(self.newsencoder)(reshaped_pred_input_string_title)
        news_present = layers.TimeDistributed(self.newsencoder)(pred_input_title_bert)

        # news_present_one = self.newsencoder(pred_string_title_one_reshape)
        news_present_one = self.newsencoder(pred_title_bert_one_reshape)

        preds = layers.Dot(axes=-1)([news_present, user_present])
        preds = layers.Activation(activation="softmax")(preds)

        pred_one = layers.Dot(axes=-1)([news_present_one, user_present])
        pred_one = layers.Activation(activation="sigmoid")(pred_one)

        # model = keras.Model([his_input_title, pred_input_title, his_input_string_title, pred_input_string_title], preds)
        # scorer = keras.Model([his_input_title, pred_input_title_one, his_input_string_title, pred_input_title_string_one], pred_one)

        model = keras.Model([his_input_title, pred_input_title, his_input_title_bert, pred_input_title_bert], preds)
        scorer = keras.Model([his_input_title, pred_input_title_one, his_input_title_bert, pred_input_title_bert_one], pred_one)

        model.summary()
        scorer.summary()
        return model, scorer
