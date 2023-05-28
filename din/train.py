# coding=gbk
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # 可以指定多卡，强行设置程序可见某几块板
import time
import pickle
import random
import numpy as np
import tensorflow as tf
import sys
from input import DataInput, DataInputTest
from model import Model


def calc_auc(raw_arr):
    """Summary

    Args:
        raw_arr (TYPE): Description

    Returns:
        TYPE: Description
    """
    # sort by pred value, from small to big
    arr = sorted(raw_arr, key=lambda d: d[2])

    auc = 0.0
    fp1, tp1, fp2, tp2 = 0.0, 0.0, 0.0, 0.0
    for record in arr:
        fp2 += record[0]  # noclick
        tp2 += record[1]  # click
        auc += (fp2 - fp1) * (tp2 + tp1)
        fp1, tp1 = fp2, tp2

    # if all nonclick or click, disgard
    threshold = len(arr) - 1e-3
    if tp2 > threshold or fp2 > threshold:
        return -0.5

    if tp2 * fp2 > 0.0:  # normal auc
        return (1.0 - auc / (2.0 * tp2 * fp2))
    else:
        return None


def _auc_arr(score):
    score_p = score[:, 0]
    score_n = score[:, 1]
    # print "============== p ============="
    # print score_p
    # print "============== n ============="
    # print score_n
    score_arr = []
    for s in score_p.tolist():
        score_arr.append([0, 1, s])
    for s in score_n.tolist():
        score_arr.append([1, 0, s])
    return score_arr


def _eval(sess, model):
    auc_sum = 0.0
    score_arr = []
    for _, uij in DataInputTest(test_set, test_batch_size):
        auc_, score_ = model.eval(sess, uij)
        score_arr += _auc_arr(score_)
        auc_sum += auc_ * len(uij[0])
    test_gauc = auc_sum / len(test_set)
    Auc = calc_auc(score_arr)
    global best_auc
    if best_auc < test_gauc:
        best_auc = test_gauc
        model.save(sess, 'save_path/ckpt')
    return test_gauc, Auc


def _test(sess, model):
    auc_sum = 0.0
    score_arr = []
    predicted_users_num = 0
    print('test sub items')
    for _, uij in DataInputTest(test_set, predict_batch_size):
        if predicted_users_num >= predict_users_num:
            break
        score_ = model.test(sess, uij)
        score_arr.append(score_)
        predicted_users_num += predict_batch_size
    return score_[0]  # 测试集损失


if __name__ == '__main__':


    random.seed(1234)  # 控制随机数生成，有效期只有1次
    np.random.seed(1234)
    # tf.set_random_seed(1234)
    tf.random.set_seed(1234)

    train_batch_size = 32
    test_batch_size = 512
    predict_batch_size = 32
    predict_users_num = 1000
    predict_ads_num = 100

    with open('dataset.pkl', 'rb') as f:
        train_set = pickle.load(f)
        test_set = pickle.load(f)
        cate_list = pickle.load(f)
        user_count, item_count, cate_count = pickle.load(f)

    best_auc = 0.0

    gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)  # tf在训练时默认占用所有GPU的显存，因此设置动态按需取显存
    with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options)) as sess:  # 创建session时的参数配置
        model = Model(user_count, item_count, cate_count, cate_list, predict_batch_size, predict_ads_num)
        sess.run(tf.compat.v1.global_variables_initializer())
        sess.run(tf.compat.v1.local_variables_initializer())

        print('test_gauc: %.4f\t test_auc: %.4f' % _eval(sess, model))
        # test_gauc: 0.5124	 test_auc: 0.5124
        sys.stdout.flush()
        lr = 1.0
        start_time = time.time()
        for _ in range(50):

            random.shuffle(train_set)

            epoch_size = round(len(train_set) / train_batch_size)  # round函数返回一个浮点数的四舍五入值
            loss_sum = 0.0
            for _, uij in DataInput(train_set, train_batch_size):
                loss = model.train(sess, uij, lr)
                loss_sum += loss

                if model.global_step.eval() % 1000 == 0:  # global_step作为梯度更新次数控制整个训练过程何时停止；.eval()：在一个Seesion里面计算tensor的值
                    test_gauc, Auc = _eval(sess, model)  # GAUC和AUC
                    print('Epoch %d Global_step %d\tTrain_loss: %.4f\tEval_GAUC: %.4f\tEval_AUC: %.4f' %
                          (model.global_epoch_step.eval(), model.global_step.eval(), loss_sum / 1000, test_gauc, Auc))
                    # Epoch 0 Global_step 1000	Train_loss: 0.6975	Eval_GAUC: 0.6740	Eval_AUC: 0.6742
                    sys.stdout.flush()
                    loss_sum = 0.0

                if model.global_step.eval() % 336000 == 0:
                    lr = 0.1

            print('Epoch %d DONE\tCost time: %.2f' % (model.global_epoch_step.eval(), time.time() - start_time))
            # Epoch 0 DONE	Cost time: 4690.95
            sys.stdout.flush()  # 刷新缓冲区
            model.global_epoch_step_op.eval()  # global_epoch_step_op: 定义global_epoch_step加一的算子

        print('best test_gauc:', best_auc)
        sys.stdout.flush()
