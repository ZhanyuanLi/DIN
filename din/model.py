# coding=gbk
import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()

from Dice import dice


class Model(object):

    def __init__(self, user_count, item_count, cate_count, cate_list, predict_batch_size, predict_ads_num):
        # tf.placeholder：占位符函数，建立session后再通过feed_dict()喂入数据，返回Tensor类型
        self.u = tf.placeholder(tf.int32, [None, ])  # [B]。user id。 (B：batch size)
        self.i = tf.placeholder(tf.int32, [None, ])  # [B]。i: 正样本的item
        self.j = tf.placeholder(tf.int32, [None, ])  # [B]。j: 负样本的item
        self.y = tf.placeholder(tf.float32, [None, ])  # [B]。y: label
        self.hist_i = tf.placeholder(tf.int32, [None, None])  # [B, T]。用户行为特征(User Behavior)中的item序列。T为序列长度
        self.sl = tf.placeholder(tf.int32, [None, ])  # [B]。sl：sequence length，User Behavior中序列的真实序列长度
        self.lr = tf.placeholder(tf.float64, [])  # learning rate

        hidden_units = 128

        # tf.get_variable：创建新的tensorflow变量
        # shape: [U, H], user_id的embedding weight，整个代码没用到。 U是user_id的超出设定参数（hash bucket size）
        user_emb_w = tf.get_variable("user_emb_w", [user_count, hidden_units])
        # shape: [I, H//2], item_id的embedding weight。 I是item_id的hash bucket size
        item_emb_w = tf.get_variable("item_emb_w", [item_count, hidden_units // 2])
        # shape: [I], bias
        item_b = tf.get_variable("item_b", [item_count],
                                 initializer=tf.constant_initializer(0.0))

        # shape: [C, H//2], cate_id的embedding weight。
        cate_emb_w = tf.get_variable("cate_emb_w", [cate_count, hidden_units // 2])
        # shape: [C, H//2]。tf.convert_to_tensor：将定值转换为张量
        cate_list = tf.convert_to_tensor(cate_list, dtype=tf.int64)

        # 从cate_list中取出正样本的cate。tf.gather：用一个一维的索引数组i，将张量cate_list中对应索引的向量提取出来
        ic = tf.gather(cate_list, self.i)
        # 正样本的embedding，正样本包括item和cate
        i_emb = tf.concat(values=[
            tf.nn.embedding_lookup(item_emb_w, self.i),
            tf.nn.embedding_lookup(cate_emb_w, ic),
        ], axis=1)
        # 偏置b
        i_b = tf.gather(item_b, self.i)

        # 从cate_list中取出负样本的cate
        jc = tf.gather(cate_list, self.j)
        # 负样本的embedding，负样本包括item和cate
        j_emb = tf.concat([
            tf.nn.embedding_lookup(item_emb_w, self.j),
            tf.nn.embedding_lookup(cate_emb_w, jc),
        ], axis=1)
        # 偏置b
        j_b = tf.gather(item_b, self.j)

        # 用户行为序列(User Behavior)中的cate序列
        hc = tf.gather(cate_list, self.hist_i)
        # 用户行为序列(User Behavior)的embedding，包括item序列和cate序列
        h_emb = tf.concat([
            tf.nn.embedding_lookup(item_emb_w, self.hist_i),
            tf.nn.embedding_lookup(cate_emb_w, hc),
        ], axis=2)

        # attention操作
        hist_i = attention(i_emb, h_emb, self.sl)
        # -- attention end ---

        hist_i = tf.layers.batch_normalization(inputs=hist_i)
        hist_i = tf.reshape(hist_i, [-1, hidden_units], name='hist_bn')
        # 添加一层全连接层，hist_i为输入，hidden_units为输出维数
        hist_i = tf.layers.dense(hist_i, hidden_units, name='hist_fcn')

        u_emb_i = hist_i

        hist_j = attention(j_emb, h_emb, self.sl)
        # -- attention end ---

        # hist_j = tf.layers.batch_normalization(inputs = hist_j)
        hist_j = tf.layers.batch_normalization(inputs=hist_j, reuse=True)
        hist_j = tf.reshape(hist_j, [-1, hidden_units], name='hist_bn')
        hist_j = tf.layers.dense(hist_j, hidden_units, name='hist_fcn', reuse=True)

        u_emb_j = hist_j
        print(u_emb_i.get_shape().as_list())
        print(u_emb_j.get_shape().as_list())
        print(i_emb.get_shape().as_list())
        print(j_emb.get_shape().as_list())

        # -- fcn begin -------
        # 正负样本的两个全连接层计算
        # u_emb_i: 由 Attention 层得到的输出结果, 表示用户兴趣
        # i_emb：候选广告对应的 embedding
        # u_emb_i * i_emb：用户兴趣和候选广告的交叉特征
        din_i = tf.concat([u_emb_i, i_emb, u_emb_i * i_emb], axis=-1)
        din_i = tf.layers.batch_normalization(inputs=din_i, name='b1')
        d_layer_1_i = tf.layers.dense(din_i, 80, activation=tf.nn.sigmoid, name='f1')
        # if u want try dice change sigmoid to None and add dice layer like following two lines. You can also find model_dice.py in this folder.
        # d_layer_1_i = tf.layers.dense(din_i, 80, activation=None, name='f1')
        # d_layer_1_i = dice(d_layer_1_i, name='dice_1_i')
        d_layer_2_i = tf.layers.dense(d_layer_1_i, 40, activation=tf.nn.sigmoid, name='f2')
        # d_layer_2_i = tf.layers.dense(d_layer_1_i, 40, activation=None, name='f2')
        # d_layer_2_i = dice(d_layer_2_i, name='dice_2_i')
        d_layer_3_i = tf.layers.dense(d_layer_2_i, 1, activation=None, name='f3')

        din_j = tf.concat([u_emb_j, j_emb, u_emb_j * j_emb], axis=-1)
        din_j = tf.layers.batch_normalization(inputs=din_j, name='b1', reuse=True)
        d_layer_1_j = tf.layers.dense(din_j, 80, activation=tf.nn.sigmoid, name='f1', reuse=True)
        # d_layer_1_j = tf.layers.dense(din_j, 80, activation=None, name='f1', reuse=True)
        # d_layer_1_j = dice(d_layer_1_j, name='dice_1_j')
        d_layer_2_j = tf.layers.dense(d_layer_1_j, 40, activation=tf.nn.sigmoid, name='f2', reuse=True)
        # d_layer_2_j = tf.layers.dense(d_layer_1_j, 40, activation=None, name='f2', reuse=True)
        # d_layer_2_j = dice(d_layer_2_j, name='dice_2_j')
        d_layer_3_j = tf.layers.dense(d_layer_2_j, 1, activation=None, name='f3', reuse=True)
        d_layer_3_i = tf.reshape(d_layer_3_i, [-1])
        d_layer_3_j = tf.reshape(d_layer_3_j, [-1])

        # 预测的（y正-y负）
        x = i_b - j_b + d_layer_3_i - d_layer_3_j  # [B]
        # 预测的（y正）
        self.logits = i_b + d_layer_3_i

        # prediciton for selected items
        # logits for selected item:
        # 将所有的除u_emb_all外的embedding，concat到一起
        item_emb_all = tf.concat([
            item_emb_w,
            tf.nn.embedding_lookup(cate_emb_w, cate_list)
        ], axis=1)
        item_emb_sub = item_emb_all[:predict_ads_num, :]
        item_emb_sub = tf.expand_dims(item_emb_sub, 0)
        item_emb_sub = tf.tile(item_emb_sub, [predict_batch_size, 1, 1])
        hist_sub = attention_multi_items(item_emb_sub, h_emb, self.sl)
        # -- attention end ---

        hist_sub = tf.layers.batch_normalization(inputs=hist_sub, name='hist_bn', reuse=tf.AUTO_REUSE)
        # print hist_sub.get_shape().as_list()
        hist_sub = tf.reshape(hist_sub, [-1, hidden_units])
        hist_sub = tf.layers.dense(hist_sub, hidden_units, name='hist_fcn', reuse=tf.AUTO_REUSE)

        u_emb_sub = hist_sub
        item_emb_sub = tf.reshape(item_emb_sub, [-1, hidden_units])
        # 将所有的embedding，concat到一起
        din_sub = tf.concat([u_emb_sub, item_emb_sub, u_emb_sub * item_emb_sub], axis=-1)
        din_sub = tf.layers.batch_normalization(inputs=din_sub, name='b1', reuse=True)
        d_layer_1_sub = tf.layers.dense(din_sub, 80, activation=tf.nn.sigmoid, name='f1', reuse=True)
        # d_layer_1_sub = dice(d_layer_1_sub, name='dice_1_sub')
        d_layer_2_sub = tf.layers.dense(d_layer_1_sub, 40, activation=tf.nn.sigmoid, name='f2', reuse=True)
        # d_layer_2_sub = dice(d_layer_2_sub, name='dice_2_sub')
        d_layer_3_sub = tf.layers.dense(d_layer_2_sub, 1, activation=None, name='f3', reuse=True)
        d_layer_3_sub = tf.reshape(d_layer_3_sub, [-1, predict_ads_num])
        self.logits_sub = tf.sigmoid(item_b[:predict_ads_num] + d_layer_3_sub)
        self.logits_sub = tf.reshape(self.logits_sub, [-1, predict_ads_num, 1])
        # -- fcn end -------

        self.mf_auc = tf.reduce_mean(tf.to_float(x > 0))
        self.score_i = tf.sigmoid(i_b + d_layer_3_i)
        self.score_j = tf.sigmoid(j_b + d_layer_3_j)
        self.score_i = tf.reshape(self.score_i, [-1, 1])
        self.score_j = tf.reshape(self.score_j, [-1, 1])
        self.p_and_n = tf.concat([self.score_i, self.score_j], axis=-1)
        print(self.p_and_n.get_shape().as_list())

        # Step variable
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.global_epoch_step = \
            tf.Variable(0, trainable=False, name='global_epoch_step')
        self.global_epoch_step_op = \
            tf.assign(self.global_epoch_step, self.global_epoch_step + 1)

        self.loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.logits,
                labels=self.y)
        )

        trainable_params = tf.trainable_variables()
        self.opt = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
        gradients = tf.gradients(self.loss, trainable_params)
        clip_gradients, _ = tf.clip_by_global_norm(gradients, 5)
        self.train_op = self.opt.apply_gradients(
            zip(clip_gradients, trainable_params), global_step=self.global_step)

    def train(self, sess, uij, l):
        loss, _ = sess.run([self.loss, self.train_op], feed_dict={
            self.u: uij[0],  # reviewerID
            self.i: uij[1],  # pos/neg_list[i]
            self.y: uij[2],  # 1/0
            self.hist_i: uij[3],  # 该轮epoch内train_set的hist数组（短的向量后面已补零）
            self.sl: uij[4],  # 每个reviewerID的实际hist长度
            self.lr: l,  # 学习率
        })
        return loss

    def eval(self, sess, uij):
        u_auc, socre_p_and_n = sess.run([self.mf_auc, self.p_and_n], feed_dict={
            self.u: uij[0],
            self.i: uij[1],  # 正样本
            self.j: uij[2],  # 负样本
            self.hist_i: uij[3],
            self.sl: uij[4],
        })
        return u_auc, socre_p_and_n

    def test(self, sess, uij):
        return sess.run(self.logits_sub, feed_dict={
            self.u: uij[0],
            self.i: uij[1],
            self.j: uij[2],
            self.hist_i: uij[3],
            self.sl: uij[4],
        })

    def save(self, sess, path):
        saver = tf.train.Saver()
        saver.save(sess, save_path=path)

    def restore(self, sess, path):
        saver = tf.train.Saver()
        saver.restore(sess, save_path=path)


def extract_axis_1(data, ind):
    batch_range = tf.range(tf.shape(data)[0])
    indices = tf.stack([batch_range, ind], axis=1)
    res = tf.gather_nd(data, indices)
    return res


def attention(queries, keys, keys_length):
    """
    queries:     [B, H]
    keys:        [B, T, H]
    keys_length: [B], 保存着用户历史行为序列的真实长度
    """
    queries_hidden_units = queries.get_shape().as_list()[-1]  # queries.get_shape().as_list()[-1]就是H
    queries = tf.tile(queries, [1, tf.shape(keys)[1]])  # tf.shape(keys)[1] 结果就是 T
    queries = tf.reshape(queries, [-1, tf.shape(keys)[1], queries_hidden_units])  # queries 先 reshape 成和 keys 相同的大小: [B, T, H]

    # Local Activation Unit 的输入, 候选广告 queries 对应的 emb 以及用户历史行为序列 keys
    # 对应的 embed, 再加上它们之间的交叉特征, 进行 concat 后, 再输入到一个 DNN 网络中
    # DNN 网络的输出节点为 1
    din_all = tf.concat([queries, keys, queries - keys, queries * keys], axis=-1)
    d_layer_1_all = tf.layers.dense(din_all, 80, activation=tf.nn.sigmoid, name='f1_att', reuse=tf.AUTO_REUSE)
    d_layer_2_all = tf.layers.dense(d_layer_1_all, 40, activation=tf.nn.sigmoid, name='f2_att', reuse=tf.AUTO_REUSE)
    d_layer_3_all = tf.layers.dense(d_layer_2_all, 1, activation=None, name='f3_att', reuse=tf.AUTO_REUSE)
    # 上一层 d_layer_3_all 的 shape 为 [B, T, 1]
    # 下一步 reshape 为 [B, 1, T], axis=2 这一维表示 T 个用户行为序列分别对应的权重参数
    d_layer_3_all = tf.reshape(d_layer_3_all, [-1, 1, tf.shape(keys)[1]])
    outputs = d_layer_3_all
    # Mask
    # 由于一个 Batch 中的用户行为序列不一定都相同, 其真实长度保存在 keys_length 中
    # 所以这里要产生 masks 来选择真正的历史行为
    key_masks = tf.sequence_mask(keys_length, tf.shape(keys)[1])  # [B, T]
    key_masks = tf.expand_dims(key_masks, 1)  # [B, 1, T]
    paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
    # 选出真实的历史行为, 而对于那些填充的结果, 适用 paddings 中的值来表示
    # padddings 中使用巨大的负值, 后面计算 softmax 时, e^{x} 结果就约等于 0
    outputs = tf.where(key_masks, outputs, paddings)  # [B, 1, T]

    # Scale
    outputs = outputs / (keys.get_shape().as_list()[-1] ** 0.5)

    # Activation
    outputs = tf.nn.softmax(outputs)  # [B, 1, T]

    # Weighted sum
    # outputs 的大小为 [B, 1, T], 表示每条历史行为的权重,
    # keys 为历史行为序列, 大小为 [B, T, H];
    # 两者用矩阵乘法做, 得到的结果就是 [B, 1, H]
    outputs = tf.matmul(outputs, keys)  # [B, 1, H]

    return outputs


# 和上面attention的处理逻辑一样，只是一次处理N个候选广告
def attention_multi_items(queries, keys, keys_length):
    """
    queries:     [B, N, H] N is the number of ads
    keys:        [B, T, H]
    keys_length: [B]
    """
    queries_hidden_units = queries.get_shape().as_list()[-1]
    queries_nums = queries.get_shape().as_list()[1]
    queries = tf.tile(queries, [1, 1, tf.shape(keys)[1]])
    queries = tf.reshape(queries, [-1, queries_nums, tf.shape(keys)[1], queries_hidden_units])  # shape : [B, N, T, H]
    max_len = tf.shape(keys)[1]
    keys = tf.tile(keys, [1, queries_nums, 1])
    keys = tf.reshape(keys, [-1, queries_nums, max_len, queries_hidden_units])  # shape : [B, N, T, H]
    din_all = tf.concat([queries, keys, queries - keys, queries * keys], axis=-1)
    d_layer_1_all = tf.layers.dense(din_all, 80, activation=tf.nn.sigmoid, name='f1_att', reuse=tf.AUTO_REUSE)
    d_layer_2_all = tf.layers.dense(d_layer_1_all, 40, activation=tf.nn.sigmoid, name='f2_att', reuse=tf.AUTO_REUSE)
    d_layer_3_all = tf.layers.dense(d_layer_2_all, 1, activation=None, name='f3_att', reuse=tf.AUTO_REUSE)
    d_layer_3_all = tf.reshape(d_layer_3_all, [-1, queries_nums, 1, max_len])
    outputs = d_layer_3_all
    # Mask
    key_masks = tf.sequence_mask(keys_length, max_len)  # [B, T]
    key_masks = tf.tile(key_masks, [1, queries_nums])
    key_masks = tf.reshape(key_masks, [-1, queries_nums, 1, max_len])  # shape : [B, N, 1, T]
    paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
    outputs = tf.where(key_masks, outputs, paddings)  # [B, N, 1, T]

    # Scale
    outputs = outputs / (keys.get_shape().as_list()[-1] ** 0.5)

    # Activation
    outputs = tf.nn.softmax(outputs)  # [B, N, 1, T]
    outputs = tf.reshape(outputs, [-1, 1, max_len])
    keys = tf.reshape(keys, [-1, max_len, queries_hidden_units])
    # print outputs.get_shape().as_list()
    # print keys.get_sahpe().as_list()
    # Weighted sum
    outputs = tf.matmul(outputs, keys)
    outputs = tf.reshape(outputs, [-1, queries_nums, queries_hidden_units])  # [B, N, 1, H]
    print(outputs.get_shape().as_list())
    return outputs
