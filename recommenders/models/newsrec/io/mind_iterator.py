# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import math
import os
import tensorflow as tf
import numpy as np
import pickle

from recommenders.models.deeprec.io.iterator import BaseIterator
from recommenders.models.newsrec.newsrec_utils import word_tokenize, newsample
import tensorflow_hub as hub

#from recommenders.models.newsrec.io.bert_model import bert_preprocess_model, bert_model

from transformers import TFAutoModel, AutoTokenizer


__all__ = ["MINDIterator"]


class MINDIterator(BaseIterator):
    """Train data loader for NAML model.
    The model require a special type of data format, where each instance contains a label, impresion id, user id,
    the candidate news articles and user's clicked news article. Articles are represented by title words,
    body words, verts and subverts.

    Iterator will not load the whole data into memory. Instead, it loads data into memory
    per mini-batch, so that large files can be used as input data.

    Attributes:
        col_spliter (str): column spliter in one line.
        ID_spliter (str): ID spliter in one line.
        batch_size (int): the samples num in one batch.
        title_size (int): max word num in news title.
        his_size (int): max clicked news num in user click history.
        npratio (int): negaive and positive ratio used in negative sampling. -1 means no need of negtive sampling.
    """

    def __init__(
            self,
            hparams,
            npratio=-1,
            col_spliter="\t",
            ID_spliter="%",
    ):
        """Initialize an iterator. Create necessary placeholders for the model.

        Args:
            hparams (object): Global hyper-parameters. Some key setttings such as head_num and head_dim are there.
            npratio (int): negaive and positive ratio used in negative sampling. -1 means no need of negtive sampling.
            col_spliter (str): column spliter in one line.
            ID_spliter (str): ID spliter in one line.
        """
        self.col_spliter = col_spliter
        self.ID_spliter = ID_spliter
        self.batch_size = hparams.batch_size
        self.title_size = hparams.title_size
        self.his_size = hparams.his_size
        self.npratio = npratio

        self.word_dict = self.load_dict(hparams.wordDict_file)
        self.uid2index = self.load_dict(hparams.userDict_file)

    def load_dict(self, file_path):
        """load pickle file

        Args:
            file path (str): file path

        Returns:
            object: pickle loaded object
        """
        with open(file_path, "rb") as f:
            return pickle.load(f)

    import tensorflow as tf


    def init_news(self, news_file):
        """init news information given news file, such as news_title_index and nid2index.
        Args:
            news_file: path of news file
        """
        def combine_CLS(a_inputs, b_outputs):
            hidden_states = b_outputs.hidden_states  # List of hidden states (including embeddings)

            # Extract CLS token from all layers except the last one
            cls_tokens = [layer[:, 0, :] for layer in hidden_states[:]]

            # the cls_tokens has a shape of (batch_size, attention_size) with length of 24 (layer number -1).
            # reshape it to (batch_size, length -1, attention_size)
            cls_tokens_concatenated = tf.stack(cls_tokens, axis=1)

            c_input_masks = a_inputs.data["attention_mask"]
            # Create a tensor of ones with the shape of (batch_size, num_cls_tokens, 1)
            ones_mask = tf.ones((tf.shape(c_input_masks)[0], cls_tokens_concatenated.shape[1], 1), dtype=tf.float32)

            # Concatenate the combined_hidden_state and input_masks_extended
            combined_tensor = tf.concat([cls_tokens_concatenated, ones_mask], axis=2)

            return combined_tensor

        def combine_hidden_states(a_inputs, b_outputs):
            hidden_states = b_outputs.hidden_states  # List of hidden states (including embeddings)

            # Extract CLS token from all layers except the last one
            cls_tokens = [layer[:, 0, :] for layer in hidden_states[:-1]]

            # the cls_tokens has a shape of (batch_size, attention_size) with length of 24 (layer number -1).
            # reshape it to (batch_size, length -1, attention_size)
            cls_tokens_concatenated = tf.stack(cls_tokens, axis=1)

            # Concatenate the extracted CLS tokens to the last hidden state
            #cls_tokens_concatenated = cls_tokens
            last_hidden_state = hidden_states[-1]

            # Prepend the concatenated CLS tokens to the last hidden state
            combined_hidden_state = tf.concat([cls_tokens_concatenated, last_hidden_state], axis=1)

            c_input_masks = a_inputs.data["attention_mask"]
            # Create a tensor of ones with the shape of (batch_size, num_cls_tokens, 1)
            ones_mask = tf.ones((tf.shape(c_input_masks)[0], cls_tokens_concatenated.shape[1], 1), dtype=tf.float32)
            c_input_masks_reshapped = tf.reshape(c_input_masks, (-1, 30, 1))
            c_input_masks_reshapped = tf.cast(c_input_masks_reshapped, tf.float32)

        # Concatenate the ones_mask with the input_masks
            input_masks_extended = tf.concat([ones_mask, c_input_masks_reshapped], axis=1)

            # Concatenate the combined_hidden_state and input_masks_extended
            combined_tensor = tf.concat([combined_hidden_state, input_masks_extended], axis=2)

            return combined_tensor




        self.nid2index = {}
        self.news_title = [""]
        news_title = [""]

        text_test = ['this is such an amazing movie!']
        text_test = text_test
        # text_preprocessed = bert_preprocess_model(text_test)
        #
        # print(f'Keys       : {list(text_preprocessed.keys())}')
        # print(f'Shape      : {text_preprocessed["input_word_ids"].shape}')
        # print(f'Word Ids   : {text_preprocessed["input_word_ids"][0, :12]}')
        # print(f'Input Mask : {text_preprocessed["input_mask"][0, :12]}')
        # print(f'Type Ids   : {text_preprocessed["input_type_ids"][0, :12]}')
        #
        # bert_results = bert_model(text_preprocessed)
        #
        # print(f'Pooled Outputs Shape:{bert_results["pooled_output"].shape}')
        # print(f'Pooled Outputs Values:{bert_results["pooled_output"][0, :12]}')
        # print(f'Sequence Outputs Shape:{bert_results["sequence_output"].shape}')
        # print(f'Sequence Outputs Values:{bert_results["sequence_output"][0, :12]}')


        count = 0

        batch_size_for_bert = 64
        batch_count_for_bert = 0
        title_batch_prepared_for_bert = []
        maxElementsPerFile = batch_size_for_bert * 100


        fileLinesDict = {
            'demo': 26740,
            'small': 51282,
            'large': 101527
        }


        fileLinesKey = os.environ.get("MIND_type")

        fileLines = fileLinesDict[fileLinesKey] + 1

        use_saved_bert = False

        bert_model_names = {
            "tf_hub_bert_1": "tf_hub_bert_1",
            "deberta": "deberta"
        }

        deberta_tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v2-xlarge")
        deberta_model = TFAutoModel.from_pretrained("microsoft/deberta-v2-xlarge")

        selected_bert_model = "deberta"
        use_CLS_token_only = "use_CLS_token_only"
        add_CLS_of_each_layer = "add_CLS_of_each_layer"
        use_all_CLS_tokens_only = "use_all_CLS_tokens_only"

        bertMode = use_all_CLS_tokens_only

        cached_bert_pooled_output_file_name = news_file + "_" + selected_bert_model + "_" + bertMode + "_bert_index.npy"

        if os.path.exists(cached_bert_pooled_output_file_name):
            use_saved_bert = True
        else:
            print("File does not exist, will create new bert cache")

        memory_map_writer = None
        if use_saved_bert:
            #self.news_title_bert_index = np.load(cached_bert_pooled_output_file_name)
            # number_of_files = math.floor(fileLines)/maxElementsPerFile
            # number_of_elements_last_file = fileLines - number_of_files * maxElementsPerFile
            # memory_map = None
            # for i in range(number_of_files + 1):
            #     shape1 = maxElementsPerFile
            #     if i == number_of_files:
            #         shape1 = number_of_elements_last_file
            #     mmapped_array = np.memmap(cached_bert_pooled_output_file_name + "." + fileLinesKey + ".npy", dtype='float32', mode='r', shape=(shape1,))
            #     if memory_map is None:
            #         memory_map = mmapped_array
            #     else:
            #         memory_map = np.hstack((memory_map, mmapped_array))

            memory_map = np.memmap(cached_bert_pooled_output_file_name, dtype='float32', mode='r', shape=(fileLines,25, 1537))
            self.news_title_bert_index = memory_map
        else:
            memory_map_writer = np.memmap(cached_bert_pooled_output_file_name, dtype='float32', mode='w+', shape=(fileLines,25, 1537))


        use_fake_data = True
        if (not use_saved_bert) and selected_bert_model == "tf_hub_bert_1":
            pass
            # self.news_title_bert_index = bert_results["pooled_output"]
            # self.news_title_bert_index = np.asarray(self.news_title_bert_index)
        if (not use_saved_bert) and selected_bert_model == "deberta":

            if (use_fake_data):
                firstLine = np.zeros((1, 25, 1537), dtype="float32")
                #self.news_title_bert_index = firstLine
                memory_map_writer[0] = firstLine
                memory_map_writer.flush()

            else:
                inputs = deberta_tokenizer(
                    text_test,
                    padding='max_length',
                    truncation=True,
                    max_length=30,
                    return_tensors="tf")
                outputs = deberta_model(**inputs, output_hidden_states=True)
                if bertMode == use_CLS_token_only:
                    bert_title = outputs.last_hidden_state[:, 0:1, :]
                    bert_title = tf.reshape(bert_title, (-1, 1536))
                elif bertMode == add_CLS_of_each_layer:
                    # bert_title = outputs.last_hidden_state[:, :, :]
                    # input_masks = inputs.data["attention_mask"]
                    # input_masks = tf.reshape(input_masks, (-1, 30, 1))
                    # input_masks = tf.cast(input_masks, tf.float32)
                    # # concat the bert_title and masks together
                    # # will split them back to original in the news encoder
                    # bert_title = tf.concat([bert_title, input_masks], axis=2)

                    bert_title = combine_hidden_states(inputs, outputs)
                elif bertMode == use_all_CLS_tokens_only:
                    bert_title = combine_CLS(inputs, outputs)

                # comment these two lines out when using saved bert
                # self.news_title_bert_index = bert_title
                # self.news_title_bert_index = np.asarray(self.news_title_bert_index)
                memory_map_writer[0] = bert_title
                memory_map_writer.flush()


        with tf.io.gfile.GFile(news_file, "r") as rd:
            for line in rd:
                nid, vert, subvert, title, ab, url, _, _ = line.strip("\n").split(
                    self.col_spliter
                )

                if nid in self.nid2index:
                    continue

                self.nid2index[nid] = len(self.nid2index) + 1
                self.news_title.append(title)
                # text_preprocessed = bert_preprocess_model([title])
                # bert_results = bert_model(text_preprocessed)
                # bert_title = bert_results["pooled_output"]
                # print(f'Pooled Outputs Shape:{bert_title.shape}')
                count = count + 1
                if not use_saved_bert:
                    batch_count_for_bert = batch_count_for_bert + 1
                    title_batch_prepared_for_bert.append(title)

                if len(title_batch_prepared_for_bert) != batch_count_for_bert:
                    print('fatal error: queue size does not match')
                    exit()

                if not use_saved_bert and batch_count_for_bert == batch_size_for_bert:
                    if selected_bert_model == "deberta":
                        if (use_fake_data):
                            title_batch_prepared_for_bert = []
                            batch_count_for_bert = 0

                            fake_title = np.ones((batch_size_for_bert, 25, 1537), dtype="float32")
                            #self.news_title_bert_index = np.concatenate((self.news_title_bert_index, fake_title), axis=0)

                            # need to add 1, as the first element is taken
                            begin = count + 1 - batch_size_for_bert
                            end = count + 1
                            for i in range(begin, end):
                                memory_map_writer[i] = fake_title[i - begin]
                            memory_map_writer.flush()

                        else:
                            inputs = deberta_tokenizer(
                                title_batch_prepared_for_bert,
                                padding='max_length',
                                truncation=True,
                                max_length=30,
                                return_tensors="tf")
                            outputs = deberta_model(**inputs, output_hidden_states=True)
                            # clear the queue
                            title_batch_prepared_for_bert = []
                            batch_count_for_bert = 0
                            if bertMode == use_CLS_token_only:
                                bert_title = outputs.last_hidden_state[:, 0:1, :]
                                bert_title = tf.reshape(bert_title, (-1, 1536))
                            elif bertMode == add_CLS_of_each_layer:
                                # bert_title = outputs.last_hidden_state[:, :, :]
                                # input_masks = inputs.data["attention_mask"]
                                # input_masks = tf.reshape(input_masks, (-1, 30, 1))
                                # input_masks = tf.cast(input_masks, tf.float32)
                                # # concat the bert_title and masks together
                                # # will split them back to original in the news encoder
                                # bert_title = tf.concat([bert_title, input_masks], axis=2)

                                bert_title = combine_hidden_states(inputs, outputs)
                            elif bertMode == use_all_CLS_tokens_only:
                                bert_title = combine_CLS(inputs, outputs)

                            print(f'type of bert title:{type(bert_title)}')
                            print(f'Pooled Outputs Shape:{bert_title.shape}')

                            #self.news_title_bert_index = np.concatenate((self.news_title_bert_index, bert_title.numpy()), axis=0)
                            begin = count + 1 - batch_size_for_bert
                            end = count + 1
                            for i in range(begin, end):
                                memory_map_writer[i] = bert_title[i - begin]
                            memory_map_writer.flush()


                        print(count)
                        # print(f'self bert titles Shape:{self.news_title_bert_index.shape}')
                    if selected_bert_model == "tf_hub_bert_1":
                        pass
                        # text queue is full, start process
                        # text_preprocessed = bert_preprocess_model(title_batch_prepared_for_bert)
                        # bert_results = bert_model(text_preprocessed)
                        # bert_title = bert_results["pooled_output"]
                        # print(f'Pooled Outputs Shape:{bert_title.shape}')
                        # # clear the queue
                        # title_batch_prepared_for_bert = []
                        # batch_count_for_bert = 0
                        # print(f'type of self.bert_titles:{type(self.news_title_bert_index)}')
                        # print(f'type of bert title:{type(bert_title)}')
                        # self.news_title_bert_index = np.concatenate((self.news_title_bert_index, bert_title.numpy()),
                        #                                             axis=0)
                        # print(f'self bert titles Shape:{self.news_title_bert_index.shape}')

                # print(count)
                title = word_tokenize(title)
                news_title.append(title)
                if count > 500:
                    pass
                    # break

        if not use_saved_bert and batch_count_for_bert > 0:
            print('Finish processing the news file, check the batch queue')
            print(f'The batch queue still have {len(title_batch_prepared_for_bert)}')
            if len(title_batch_prepared_for_bert) != batch_count_for_bert:
                print('fatal error: queue size does not match')
                exit()
            if selected_bert_model == "tf_hub_bert_1":
                pass
                # text queue is full, start process
                # text_preprocessed = bert_preprocess_model(title_batch_prepared_for_bert)
                # bert_results = bert_model(text_preprocessed)
                # bert_title = bert_results["pooled_output"]
                # print(f'Pooled Outputs Shape:{bert_title.shape}')
                # # clear the queue
                # title_batch_prepared_for_bert = []
                # batch_count_for_bert = 0
                # print(f'type of self.bert_titles:{type(self.news_title_bert_index)}')
                # print(f'type of bert title:{type(bert_title)}')
                # self.news_title_bert_index = np.concatenate((self.news_title_bert_index, bert_title.numpy()), axis=0)
                # print(f'self bert titles Shape:{self.news_title_bert_index.shape}')
            if selected_bert_model == "deberta":
                inputs = deberta_tokenizer(
                    title_batch_prepared_for_bert,
                    padding='max_length',
                    truncation=True,
                    max_length=30,
                    return_tensors="tf")
                outputs = deberta_model(**inputs, output_hidden_states=True)
                # clear the queue
                if bertMode == use_CLS_token_only:
                    bert_title = outputs.last_hidden_state[:, 0:1, :]
                    bert_title = tf.reshape(bert_title, (-1, 1536))
                elif bertMode == add_CLS_of_each_layer:
                    # bert_title = outputs.last_hidden_state[:, :, :]
                    # input_masks = inputs.data["attention_mask"]
                    # input_masks = tf.reshape(input_masks, (-1, 30, 1))
                    # input_masks = tf.cast(input_masks, tf.float32)
                    # # concat the bert_title and masks together
                    # # will split them back to original in the news encoder
                    # bert_title = tf.concat([bert_title, input_masks], axis=2)

                    bert_title = combine_hidden_states(inputs, outputs)
                elif bertMode == use_all_CLS_tokens_only:
                    bert_title = combine_CLS(inputs, outputs)

                #self.news_title_bert_index = np.concatenate((self.news_title_bert_index, bert_title.numpy()), axis=0)
                begin = count + 1 - batch_count_for_bert
                end = count + 1
                for i in range(begin, end):
                    memory_map_writer[i] = bert_title[i - begin]
                memory_map_writer.flush()
                title_batch_prepared_for_bert = []
                batch_count_for_bert = 0
                # print(f'self bert titles Shape:{self.news_title_bert_index.shape}')

        # convert news_title list to ndarray
        self.news_title = np.asarray(self.news_title)
        # file_path = 'output.txt'

        # Save the array to a text file
        # with open(file_path, 'w') as f:
        #     for item in self.news_title:
        #         f.write("%s\n" % item)

        print("the news title shape is ")
        print(self.news_title.shape)

        print("the bert title shape is ")
        # print(self.news_title_bert_index.shape)

        if memory_map_writer is not None:
            memory_map_writer.flush()


        if not use_saved_bert:
            #np.save(cached_bert_pooled_output_file_name, self.news_title_bert_index)
            memory_map = np.memmap(cached_bert_pooled_output_file_name, dtype='float32', mode='r', shape=(fileLines, 25, 1357))
            self.news_title_bert_index = memory_map


        print(f'self bert titles Shape:{self.news_title_bert_index.shape}')

        self.news_title_index = np.zeros(
            (len(news_title), self.title_size), dtype="int32"
        )

        for news_index in range(len(news_title)):
            title = news_title[news_index]
            for word_index in range(min(self.title_size, len(title))):
                if title[word_index] in self.word_dict:
                    self.news_title_index[news_index, word_index] = self.word_dict[
                        title[word_index].lower()
                    ]

        print("the news title index shape is ")
        print(self.news_title_index.shape)

    def init_behaviors(self, behaviors_file):
        """init behavior logs given behaviors file.

        Args:
        behaviors_file: path of behaviors file
        """
        self.histories = []
        self.imprs = []
        self.labels = []
        self.impr_indexes = []
        self.uindexes = []

        with tf.io.gfile.GFile(behaviors_file, "r") as rd:
            impr_index = 0
            for line in rd:
                uid, time, history, impr = line.strip("\n").split(self.col_spliter)[-4:]

                history = [self.nid2index[i] for i in history.split()]
                history = [0] * (self.his_size - len(history)) + history[
                                                                 : self.his_size
                                                                 ]

                impr_news = [self.nid2index[i.split("-")[0]] for i in impr.split()]
                label = [int(i.split("-")[1]) for i in impr.split()]
                uindex = self.uid2index[uid] if uid in self.uid2index else 0

                self.histories.append(history)
                self.imprs.append(impr_news)
                self.labels.append(label)
                self.impr_indexes.append(impr_index)
                self.uindexes.append(uindex)
                impr_index += 1

    def parser_one_line(self, line):
        """Parse one behavior sample into feature values.
        if npratio is larger than 0, return negtive sampled result.

        Args:
            line (int): sample index.

        Yields:
            list: Parsed results including label, impression id , user id,
            candidate_title_index, clicked_title_index.
        """
        if self.npratio > 0:
            impr_label = self.labels[line]
            impr = self.imprs[line]

            poss = []
            negs = []

            for news, click in zip(impr, impr_label):
                if click == 1:
                    poss.append(news)
                else:
                    negs.append(news)

            for p in poss:
                candidate_title_index = []
                impr_index = []
                user_index = []
                candidate_title = []
                click_title = []
                label = [1] + [0] * self.npratio

                n = newsample(negs, self.npratio)
                candidate_title_index = self.news_title_index[[p] + n]
                candidate_title_bert_index = self.news_title_bert_index[[p] + n]
                candidate_title.append(self.news_title[p])
                for i in n:
                    candidate_title.append(self.news_title[i])
                click_title_index = self.news_title_index[self.histories[line]]
                click_title_bert_index = self.news_title_bert_index[self.histories[line]]
                for i in self.histories[line]:
                    click_title.append(self.news_title[i])
                impr_index.append(self.impr_indexes[line])
                user_index.append(self.uindexes[line])

                candidate_title = np.asarray(candidate_title)
                click_title = np.asarray(click_title)

                # print("candidate_title_index shape")
                # print(candidate_title_index.shape)
                # print("click_title_index shape")
                # print(click_title_index.shape)
                # print("candidate_title shape")
                # print(candidate_title.shape)
                # print("click_title shape <====")
                # print(click_title.shape)
                #
                yield (
                    label,
                    impr_index,
                    user_index,
                    candidate_title_index,
                    click_title_index,
                    candidate_title_bert_index,
                    click_title_bert_index,
                    candidate_title,
                    click_title
                )

        else:
            impr_label = self.labels[line]
            impr = self.imprs[line]

            for news, label in zip(impr, impr_label):
                candidate_title_index = []
                candidate_title_bert_index = []
                impr_index = []
                user_index = []
                candidate_title = []
                label = [label]

                candidate_title_index.append(self.news_title_index[news])
                candidate_title_bert_index.append(self.news_title_bert_index[news])
                candidate_title.append(self.news_title[news])
                click_title_index = self.news_title_index[self.histories[line]]
                click_title_bert_index = self.news_title_bert_index[self.histories[line]]
                impr_index.append(self.impr_indexes[line])
                user_index.append(self.uindexes[line])
                click_title = self.news_title[self.histories[line]]

                yield (
                    label,
                    impr_index,
                    user_index,
                    candidate_title_index,
                    click_title_index,
                    candidate_title_bert_index,
                    click_title_bert_index,
                    candidate_title,
                    click_title
                )

    def load_data_from_file(self, news_file, behavior_file):
        """Read and parse data from news file and behavior file.

        Args:
            news_file (str): A file contains several informations of news.
            beahaviros_file (str): A file contains information of user impressions.

        Yields:
            object: An iterator that yields parsed results, in the format of dict.
        """

        if not hasattr(self, "news_title_index"):
            self.init_news(news_file)

        if not hasattr(self, "impr_indexes"):
            self.init_behaviors(behavior_file)

        label_list = []
        imp_indexes = []
        user_indexes = []
        candidate_title_indexes = []
        click_title_indexes = []
        candidate_title_bert_indexes = []
        click_title_bert_indexes = []
        candidate_titles = []
        click_titles = []
        cnt = 0

        indexes = np.arange(len(self.labels))

        if self.npratio > 0:
            np.random.shuffle(indexes)

        for index in indexes:
            for (
                    label,
                    imp_index,
                    user_index,
                    candidate_title_index,
                    click_title_index,
                    candidate_title_bert_index,
                    click_title_bert_index,
                    candidate_title,
                    click_title,
            ) in self.parser_one_line(index):
                candidate_title_indexes.append(candidate_title_index)
                click_title_indexes.append(click_title_index)
                candidate_title_bert_indexes.append(candidate_title_bert_index)
                click_title_bert_indexes.append(click_title_bert_index)
                candidate_titles.append(candidate_title)
                click_titles.append(click_title)
                imp_indexes.append(imp_index)
                user_indexes.append(user_index)
                label_list.append(label)

                cnt += 1
                if cnt >= self.batch_size:
                    yield self._convert_data(
                        label_list,
                        imp_indexes,
                        user_indexes,
                        candidate_title_indexes,
                        click_title_indexes,
                        candidate_title_bert_indexes,
                        click_title_bert_indexes,
                        candidate_titles,
                        click_titles,
                    )
                    label_list = []
                    imp_indexes = []
                    user_indexes = []
                    candidate_title_indexes = []
                    click_title_indexes = []
                    candidate_title_bert_indexes = []
                    click_title_bert_indexes = []
                    candidate_titles = []
                    click_titles = []
                    cnt = 0

        if cnt > 0:
            yield self._convert_data(
                label_list,
                imp_indexes,
                user_indexes,
                candidate_title_indexes,
                click_title_indexes,
                candidate_title_bert_indexes,
                click_title_bert_indexes,
                candidate_titles,
                click_titles
            )

    def _convert_data(
            self,
            label_list,
            imp_indexes,
            user_indexes,
            candidate_title_indexes,
            click_title_indexes,
            candidate_title_bert_indexes,
            click_title_bert_indexes,
            candidate_titles,
            click_titles
    ):
        """Convert data into numpy arrays that are good for further model operation.

        Args:
            label_list (list): a list of ground-truth labels.
            imp_indexes (list): a list of impression indexes.
            user_indexes (list): a list of user indexes.
            candidate_title_indexes (list): the candidate news titles' words indices.
            click_title_indexes (list): words indices for user's clicked news titles.

        Returns:
            dict: A dictionary, containing multiple numpy arrays that are convenient for further operation.
        """

        labels = np.asarray(label_list, dtype=np.float32)
        imp_indexes = np.asarray(imp_indexes, dtype=np.int32)
        user_indexes = np.asarray(user_indexes, dtype=np.int32)
        candidate_title_index_batch = np.asarray(
            candidate_title_indexes, dtype=np.int64
        )
        click_title_index_batch = np.asarray(click_title_indexes, dtype=np.int64)
        candidate_title_bert_index_batch = np.asarray(
            candidate_title_bert_indexes, dtype=np.float32
        )
        click_title_bert_index_batch = np.asarray(click_title_bert_indexes, dtype=np.float32)
        return {
            "impression_index_batch": imp_indexes,
            "user_index_batch": user_indexes,
            "clicked_title_batch": click_title_index_batch,
            "candidate_title_batch": candidate_title_index_batch,
            "clicked_title_bert_batch": click_title_bert_index_batch,
            "candidate_title_bert_batch": candidate_title_bert_index_batch,
            "labels": labels,
            "candidate_title_string_batch": np.asarray(candidate_titles),
            "clicked_title_string_batch": np.asarray(click_titles),
        }

    def load_user_from_file(self, news_file, behavior_file):
        """Read and parse user data from news file and behavior file.

        Args:
            news_file (str): A file contains several informations of news.
            beahaviros_file (str): A file contains information of user impressions.

        Yields:
            object: An iterator that yields parsed user feature, in the format of dict.
        """

        if not hasattr(self, "news_title_index"):
            self.init_news(news_file)

        if not hasattr(self, "impr_indexes"):
            self.init_behaviors(behavior_file)

        user_indexes = []
        impr_indexes = []
        click_title_indexes = []
        click_title_bert_indexes = []
        click_title_string_batch = []
        cnt = 0

        for index in range(len(self.impr_indexes)):
            click_title_indexes.append(self.news_title_index[self.histories[index]])
            click_title_bert_indexes.append(self.news_title_bert_index[self.histories[index]])
            click_title_string_batch.append(self.news_title[self.histories[index]])
            # for i in self.histories[index]:
            #     click_title_string_batch.append(self.news_title[i])

            user_indexes.append(self.uindexes[index])
            impr_indexes.append(self.impr_indexes[index])

            cnt += 1
            if cnt >= self.batch_size:
                yield self._convert_user_data(
                    user_indexes,
                    impr_indexes,
                    click_title_indexes,
                    click_title_bert_indexes,
                    click_title_string_batch
                )
                user_indexes = []
                impr_indexes = []
                click_title_indexes = []
                click_title_bert_indexes = []
                click_title_string_batch = []
                cnt = 0

        if cnt > 0:
            yield self._convert_user_data(
                user_indexes,
                impr_indexes,
                click_title_indexes,
                click_title_bert_indexes,
                click_title_string_batch,
            )

    def _convert_user_data(
            self,
            user_indexes,
            impr_indexes,
            click_title_indexes,
            click_title_bert_indexes,
            click_title_string_batch,
    ):
        """Convert data into numpy arrays that are good for further model operation.

        Args:
            user_indexes (list): a list of user indexes.
            click_title_indexes (list): words indices for user's clicked news titles.

        Returns:
            dict: A dictionary, containing multiple numpy arrays that are convenient for further operation.
        """

        user_indexes = np.asarray(user_indexes, dtype=np.int32)
        impr_indexes = np.asarray(impr_indexes, dtype=np.int32)
        click_title_index_batch = np.asarray(click_title_indexes, dtype=np.int64)
        click_title_bert_index_batch = np.asarray(click_title_bert_indexes, dtype=np.float32)

        return {
            "user_index_batch": user_indexes,
            "impr_index_batch": impr_indexes,
            "clicked_title_batch": click_title_index_batch,
            "clicked_title_bert_batch": click_title_bert_index_batch,
            "clicked_title_string_batch": np.asarray(click_title_string_batch),
        }

    def load_news_from_file(self, news_file):
        """Read and parse user data from news file.

        Args:
            news_file (str): A file contains several informations of news.

        Yields:
            object: An iterator that yields parsed news feature, in the format of dict.
        """
        if not hasattr(self, "news_title_index"):
            self.init_news(news_file)

        news_indexes = []
        candidate_title_indexes = []
        candidate_title_bert_indexes = []
        candidate_title_string_batch = []
        cnt = 0

        for index in range(len(self.news_title_index)):
            news_indexes.append(index)
            candidate_title_indexes.append(self.news_title_index[index])
            candidate_title_bert_indexes.append(self.news_title_bert_index[index])
            candidate_title_string_batch.append(self.news_title[index])

            cnt += 1
            if cnt >= self.batch_size:
                yield self._convert_news_data(
                    news_indexes,
                    candidate_title_indexes,
                    candidate_title_bert_indexes,
                    candidate_title_string_batch,
                )
                news_indexes = []
                candidate_title_indexes = []
                candidate_title_bert_indexes = []
                candidate_title_string_batch = []
                cnt = 0

        if cnt > 0:
            yield self._convert_news_data(
                news_indexes,
                candidate_title_indexes,
                candidate_title_bert_indexes,
                candidate_title_string_batch,
            )

    def _convert_news_data(
            self,
            news_indexes,
            candidate_title_indexes,
            candidate_title_bert_indexes,
            candidate_title_string_batch,
    ):
        """Convert data into numpy arrays that are good for further model operation.

        Args:
            news_indexes (list): a list of news indexes.
            candidate_title_indexes (list): the candidate news titles' words indices.

        Returns:
            dict: A dictionary, containing multiple numpy arrays that are convenient for further operation.
        """

        news_indexes_batch = np.asarray(news_indexes, dtype=np.int32)
        candidate_title_index_batch = np.asarray(
            candidate_title_indexes, dtype=np.int32
        )

        return {
            "news_index_batch": news_indexes_batch,
            "candidate_title_batch": candidate_title_index_batch,
            "candidate_title_bert_batch": np.asarray(candidate_title_bert_indexes),
            "candidate_title_string_batch": np.asarray(candidate_title_string_batch),
        }

    def load_impression_from_file(self, behaivors_file):
        """Read and parse impression data from behaivors file.

        Args:
            behaivors_file (str): A file contains several informations of behaviros.

        Yields:
            object: An iterator that yields parsed impression data, in the format of dict.
        """

        if not hasattr(self, "histories"):
            self.init_behaviors(behaivors_file)

        indexes = np.arange(len(self.labels))

        for index in indexes:
            impr_label = np.array(self.labels[index], dtype="int32")
            impr_news = np.array(self.imprs[index], dtype="int32")

            yield (
                self.impr_indexes[index],
                impr_news,
                self.uindexes[index],
                impr_label,
            )
