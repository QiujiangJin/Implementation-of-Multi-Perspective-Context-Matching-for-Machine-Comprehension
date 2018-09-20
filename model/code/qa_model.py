from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import logging

import numpy as np
from six.moves import xrange
import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs

from evaluate import exact_match_score, f1_score

logging.basicConfig(level=logging.INFO)

FLAGS = tf.app.flags.FLAGS

def get_optimizer(opt):
    if opt == "adam":
        optfn = tf.train.AdamOptimizer
    elif opt == "sgd":
        optfn = tf.train.GradientDescentOptimizer
    else:
        assert (False)
    return optfn

def load_dataset(dataset):
    fd1, fd2, fd3 = open(dataset[0]), open(dataset[1]), open(dataset[2])
    batch_1 = []
    batch_2 = []
    batch_3 = []
    batch_4 = []
    while True:
        line1, line2, line3 = fd1.readline(), fd2.readline(), fd3.readline()
        if not line1:
            break
        line1_list = map(int, line1.split(" "))
        line2_list = map(int, line2.split(" "))
        line3_list = map(int, line3.split(" "))
        if line3_list[1] >= FLAGS.paragraph_output_size:
            continue
        batch_1.append(line1_list)
        batch_2.append(line2_list)
        batch_3.append(line3_list)
        batch_4.append(line2)
        if len(batch_1) == FLAGS.batch_size:
            yield (batch_1, batch_2, batch_3, batch_4)
            batch_1 = []
            batch_2 = []
            batch_3 = []
            batch_4 = []

def pad_sequences(data_bacth, max_length):
    result = []
    for data in data_bacth:
        n = max_length - len(data)
        last = min(len(data), max_length)
        data_padding = data[:last] + [0]*max(n, 0)
        masks = [1]*last + [0]*max(n, 0)
        result.append((data_padding, masks))
    return result

def matching_function_full(q_state, p_out, scope):
    with vs.variable_scope(scope):
        xavier = tf.contrib.layers.xavier_initializer()
        W = tf.get_variable("W", shape=(FLAGS.matching_units, FLAGS.state_size), initializer=xavier)
        q_state = tf.reshape(q_state, shape=[-1, 1, 1, FLAGS.state_size])
        p_out = tf.reshape(p_out, shape=[-1, FLAGS.paragraph_output_size, 1, FLAGS.state_size])
        q = q_state*W
        p = p_out*W
        q = tf.nn.l2_normalize(q, dim=3)
        p = tf.nn.l2_normalize(p, dim=3)
        result = tf.reduce_sum(q*p, axis=3)
        return result

def matching_function_max(q_out, p_out, scope):
    with vs.variable_scope(scope):
        xavier = tf.contrib.layers.xavier_initializer()
        W = tf.get_variable("W", shape=(FLAGS.matching_units, FLAGS.state_size), initializer=xavier)
        q_out = tf.reshape(q_out, shape=[-1, FLAGS.question_output_size, 1, FLAGS.state_size])
        p_out = tf.reshape(p_out, shape=[-1, FLAGS.paragraph_output_size, 1, FLAGS.state_size])
        q = q_out*W
        p = p_out*W
        q = tf.reshape(q, shape=[-1, FLAGS.question_output_size, 1, FLAGS.matching_units, FLAGS.state_size])
        p = tf.reshape(p, shape=[-1, 1, FLAGS.paragraph_output_size, FLAGS.matching_units, FLAGS.state_size])
        q = tf.nn.l2_normalize(q, dim=4)
        p = tf.nn.l2_normalize(p, dim=4)
        dot_factor = tf.reduce_sum(q*p, axis=4)
        result = tf.reduce_max(dot_factor, axis=1)
        return result

def matching_function_mean(q_out, p_out, scope):
    with vs.variable_scope(scope):
        xavier = tf.contrib.layers.xavier_initializer()
        W = tf.get_variable("W", shape=(FLAGS.matching_units, FLAGS.state_size), initializer=xavier)
        q_out = tf.reshape(q_out, shape=[-1, FLAGS.question_output_size, 1, FLAGS.state_size])
        p_out = tf.reshape(p_out, shape=[-1, FLAGS.paragraph_output_size, 1, FLAGS.state_size])
        q = q_out*W
        p = p_out*W
        q = tf.reshape(q, shape=[-1, FLAGS.question_output_size, 1, FLAGS.matching_units, FLAGS.state_size])
        p = tf.reshape(p, shape=[-1, 1, FLAGS.paragraph_output_size, FLAGS.matching_units, FLAGS.state_size])
        q = tf.nn.l2_normalize(q, dim=4)
        p = tf.nn.l2_normalize(p, dim=4)
        dot_factor = tf.reduce_sum(q*p, axis=4)
        result = tf.reduce_mean(dot_factor, axis=1)
        return result

class Encoder(object):
    def __init__(self, size, vocab_dim):
        self.size = size
        self.vocab_dim = vocab_dim
        self.cell_q_fw = tf.nn.rnn_cell.BasicLSTMCell(self.size)
        self.cell_q_bw = tf.nn.rnn_cell.BasicLSTMCell(self.size)
        self.cell_p_fw = tf.nn.rnn_cell.BasicLSTMCell(self.size)
        self.cell_p_bw = tf.nn.rnn_cell.BasicLSTMCell(self.size)
        self.cell_a_fw = tf.nn.rnn_cell.BasicLSTMCell(self.size)
        self.cell_a_bw = tf.nn.rnn_cell.BasicLSTMCell(self.size)

    def encode_filter(self, question, paragraph):
        q = tf.nn.l2_normalize(question, dim=2)
        p = tf.nn.l2_normalize(paragraph, dim=2)
        q_reshape = tf.reshape(q, shape=[-1, FLAGS.question_output_size, 1, FLAGS.embedding_size])
        p_reshape = tf.reshape(p, shape=[-1, 1, FLAGS.paragraph_output_size, FLAGS.embedding_size])
        cosine = tf.reduce_sum(q_reshape*p_reshape, axis=3)
        relevancy = tf.reduce_max(cosine, axis=1)
        relevancy = tf.reshape(relevancy, shape=[-1, FLAGS.paragraph_output_size, 1])
        paragraph_filters = relevancy*paragraph
        return paragraph_filters

    def encode(self, question, paragraph, masks_q, masks_p):
        with vs.variable_scope("question"):
            srclen_q = tf.reduce_sum(masks_q, axis=1)
            (q_fw_out, q_bw_out), (q_fw_state, q_bw_state) = tf.nn.bidirectional_dynamic_rnn(self.cell_q_fw, self.cell_q_bw, question, sequence_length=srclen_q, initial_state_fw=None, initial_state_bw=None, dtype=tf.float32)
        with vs.variable_scope("paragraph"):
            srclen_p = tf.reduce_sum(masks_p, axis=1)
            (p_fw_out, p_bw_out), _ = tf.nn.bidirectional_dynamic_rnn(self.cell_p_fw, self.cell_p_bw, paragraph, sequence_length=srclen_p, initial_state_fw=q_fw_state, initial_state_bw=q_bw_state, dtype=tf.float32)
        return q_fw_out, q_bw_out, q_fw_state[1], q_bw_state[1], p_fw_out, p_bw_out

    def encode_matching(self, q_fw_out, q_bw_out, q_fw_state, q_bw_state, p_fw_out, p_bw_out):
        matching_full_fw = matching_function_full(q_fw_state, p_fw_out, "full_fw")
        matching_full_bw = matching_function_full(q_bw_state, p_bw_out, "full_bw")
        matching_max_fw = matching_function_max(q_fw_out, p_fw_out, "max_fw")
        matching_max_bw = matching_function_max(q_bw_out, p_bw_out, "max_bw")
        matching_mean_fw = matching_function_mean(q_fw_out, p_fw_out, "mean_fw")
        matching_mean_bw = matching_function_mean(q_bw_out, p_bw_out, "mean_bw")
        matching_vector = tf.concat(2, [matching_full_fw, matching_full_bw, matching_max_fw, matching_max_bw, matching_mean_fw, matching_mean_bw])
        return matching_vector

    def encode_aggregation(self, matching_inputs, masks_p):
        with vs.variable_scope("aggregation"):
            srclen = tf.reduce_sum(masks_p, axis=1)
            aggre_out, _ = tf.nn.bidirectional_dynamic_rnn(self.cell_a_fw, self.cell_a_bw, matching_inputs, sequence_length=srclen, initial_state_fw=None, initial_state_bw=None, dtype=tf.float32)
        return aggre_out

class Decoder(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def decode(self, representation):
        xavier = tf.contrib.layers.xavier_initializer()
        with vs.variable_scope("a_start"):
            U_s = tf.get_variable(name="U_s", shape=(1, 1, 2*FLAGS.state_size), initializer=xavier)
            b_s = tf.Variable(tf.zeros((1, 1, )))
            a_start = tf.reduce_sum(representation*U_s, axis=2) + b_s
        with vs.variable_scope("a_end"):
            U_e = tf.get_variable(name="U_e", shape=(1, 1, 2*FLAGS.state_size), initializer=xavier)
            b_e = tf.Variable(tf.zeros((1, 1, )))
            a_end = tf.reduce_sum(representation*U_e, axis=2) + b_e
        return a_start, a_end

class QASystem(object):
    def __init__(self, encoder, decoder):
        # ==== set up placeholder tokens ========
        self.question = tf.placeholder(tf.int32, shape=(None, FLAGS.question_output_size))
        self.paragraph = tf.placeholder(tf.int32, shape=(None, FLAGS.paragraph_output_size))
        self.answer_start = tf.placeholder(tf.int32, shape=(None, ))
        self.answer_end = tf.placeholder(tf.int32, shape=(None, ))
        self.masks_question = tf.placeholder(tf.int32, shape=(None, FLAGS.question_output_size))
        self.masks_paragraph = tf.placeholder(tf.int32, shape=(None, FLAGS.paragraph_output_size))
        self.learning_rate = tf.placeholder(tf.float32)
        self.question_embeddings = None
        self.paragraph_embeddings = None
        self.a_start = None
        self.a_end = None
        self.loss = None
        self.encoder = encoder
        self.decoder = decoder

        # ==== assemble pieces ====
        with tf.variable_scope("qa", initializer=tf.uniform_unit_scaling_initializer(1.0)):
            self.setup_embeddings()
            self.setup_system()
            self.setup_loss()

        # ==== set up training/updating procedure ====
        self.optimizer = get_optimizer("adam")(self.learning_rate)
        grads, var = zip(*self.optimizer.compute_gradients(self.loss, tf.trainable_variables()))
        self.grad_norm = tf.global_norm(grads)
        self.train_op = self.optimizer.apply_gradients(zip(grads, var))

    def setup_embeddings(self):
        with vs.variable_scope("embeddings"):
            glove_matrix = np.load(FLAGS.embed_path)
            glove_information = tf.constant(glove_matrix['glove'], dtype=tf.float32)
            self.question_embeddings = tf.nn.embedding_lookup(glove_information, self.question)
            self.question_embeddings = tf.reshape(self.question_embeddings, shape=(-1, FLAGS.question_output_size, FLAGS.embedding_size))
            self.paragraph_embeddings = tf.nn.embedding_lookup(glove_information, self.paragraph)
            self.paragraph_embeddings = tf.reshape(self.paragraph_embeddings, shape=(-1, FLAGS.paragraph_output_size, FLAGS.embedding_size))

    def setup_system(self):
        paragraph_embeddings_filters = self.encoder.encode_filter(self.question_embeddings, self.paragraph_embeddings)
        q_fw_out, q_bw_out, q_fw_state, q_bw_state, p_fw_out, p_bw_out = self.encoder.encode(self.question_embeddings, paragraph_embeddings_filters, self.masks_question, self.masks_paragraph)
        matching_inputs = self.encoder.encode_matching(q_fw_out, q_bw_out, q_fw_state, q_bw_state, p_fw_out, p_bw_out)
        aggre_out = self.encoder.encode_aggregation(matching_inputs, self.masks_paragraph)
        aggre_representation = tf.concat(2, aggre_out)
        self.a_start, self.a_end = self.decoder.decode(aggre_representation)

    def setup_loss(self):
        with vs.variable_scope("loss"):
            loss_1 = tf.nn.sparse_softmax_cross_entropy_with_logits(self.a_start, self.answer_start)
            loss_2 = tf.nn.sparse_softmax_cross_entropy_with_logits(self.a_end, self.answer_end)
            self.loss = tf.reduce_mean(loss_1 + loss_2)

    def optimize(self, session, q_batch, p_batch, a_s_batch, a_e_batch, m_q_batch, m_p_batch, learning_rate):
        input_feed = {}
        input_feed[self.question] = q_batch
        input_feed[self.paragraph] = p_batch
        input_feed[self.answer_start] = a_s_batch
        input_feed[self.answer_end] = a_e_batch
        input_feed[self.masks_question] = m_q_batch
        input_feed[self.masks_paragraph] = m_p_batch
        input_feed[self.learning_rate] = learning_rate

        output_feed = []
        output_feed.append(self.train_op)
        output_feed.append(self.loss)
        output_feed.append(self.grad_norm)

        outputs = session.run(output_feed, input_feed)

        return outputs

    def decode(self, session, q_batch, p_batch, m_q_batch, m_p_batch):
        input_feed = {}
        input_feed[self.question] = q_batch
        input_feed[self.paragraph] = p_batch
        input_feed[self.masks_question] = m_q_batch
        input_feed[self.masks_paragraph] = m_p_batch

        output_feed = []
        output_feed.append(self.a_start)
        output_feed.append(self.a_end)

        outputs = session.run(output_feed, input_feed)

        return outputs

    def answer(self, session, q_batch, p_batch, m_q_batch, m_p_batch):

        yp1, yp2 = self.decode(session, q_batch, p_batch, m_q_batch, m_p_batch)

        a_s = np.argmax(yp1, axis=1)
        a_e = np.argmax(yp2, axis=1)

        return (a_s, a_e)

    def evaluate_answer(self, session, dataset, sample=100, log=False):
        f1 = 0.
        em = 0.

        num_iter = 0
        flag = False
        for q_id, p_id, a_span, paragraph in load_dataset(dataset):
            q_batch, m_q_batch = zip(*pad_sequences(q_id, FLAGS.question_output_size))
            p_batch, m_p_batch = zip(*pad_sequences(p_id, FLAGS.paragraph_output_size))
            a_s, a_e = self.answer(session, q_batch, p_batch, m_q_batch, m_p_batch)
            for i in range(len(a_span)):
                answer = paragraph[i][a_s[i]:a_e[i] + 1]
                true_answer = paragraph[i][a_span[i][0]:a_span[i][1] + 1]
                f1 += f1_score(answer, true_answer)
                em += exact_match_score(answer, true_answer)
                num_iter += 1
                if num_iter >= sample:
                    flag = True
                    break
            if flag:
                break

        f1 = f1/sample
        em = em/sample
        
        if log:
            logging.info("F1: {}, EM: {}, for {} samples".format(f1, em, sample))        

        return f1, em

    def train(self, session, dataset, train_dir):
        tic = time.time()
        params = tf.trainable_variables()
        num_params = sum(map(lambda t: np.prod(tf.shape(t.value()).eval()), params))
        toc = time.time()
        logging.info("Number of params: %d (retreival took %f secs)" % (num_params, toc - tic))

        for e in range(FLAGS.epochs):
            count = 0
            learning_rate = FLAGS.learning_rate*(0.8**e)
            for q_id, p_id, a_span, _ in load_dataset(dataset[0:3]):
                q_batch, m_q_batch = zip(*pad_sequences(q_id, FLAGS.question_output_size))
                p_batch, m_p_batch = zip(*pad_sequences(p_id, FLAGS.paragraph_output_size))
                a_s_batch, a_e_batch = zip(*a_span)
                _, loss, grad_norm = self.optimize(session, q_batch, p_batch, a_s_batch, a_e_batch, m_q_batch, m_p_batch, learning_rate)
                print("loss is {} and the norm of the gradient is {}".format(loss, grad_norm))
                count += 1
                if count >= 100:
                    f1_train, em_train = self.evaluate_answer(session, dataset[0:3], sample=100, log=True)
                    print("F1: {}, EM: {} of training in epoch: {}".format(f1_train, em_train, e))
                    f1_val, em_val = self.evaluate_answer(session, dataset[3:6], sample=100, log=True)
                    print("F1: {}, EM: {} of validation in epoch: {}".format(f1_val, em_val, e))
                    count = 0
            saver = tf.train.Saver(tf.trainable_variables())
            saver.save(session, train_dir+"/savings.ckpt")
            f1_train, em_train = self.evaluate_answer(session, dataset[0:3], sample=100, log=True)
            print("F1: {}, EM: {} of training after epoch: {}".format(f1_train, em_train, e))
            f1_val, em_val = self.evaluate_answer(session, dataset[3:6], sample=100, log=True)
            print("F1: {}, EM: {} of validation after epoch: {}".format(f1_val, em_val, e))