# coding=gbk
import pickle
import random

random.seed(1234)


def gen_neg():
    neg = pos_list[0]  # 该用户（'reviewerID'）点击的第一个商品（'asin'）
    while neg in pos_list:
        neg = random.randint(0, item_count - 1)  # 给该用户随机初始化一个新商品
    return neg


if __name__ == '__main__':
    with open('../raw_data/remap.pkl', 'rb') as f:
        reviews_df = pickle.load(f)
        cate_list = pickle.load(f)
        user_count, item_count, cate_count, example_count = pickle.load(f)

    # 构建训练集和测试集
    train_set = []
    test_set = []
    for reviewerID, hist in reviews_df.groupby('reviewerID'):  # 根据'reviewerID'进行聚合分组
        # 将hist的'asin'列作为每个'reviewerID'的正样本列表
        pos_list = hist['asin'].tolist()  # .tolist(): Series转列表。pos_list：一个评论者ID下的所有产品ID
        # 负样本列表为在item_count内产生不在pos_list中的随机数列表
        neg_list = [gen_neg() for i in range(len(pos_list))]  # 根据该用户点击的商品数量，创造一个新的随机商品列表作为负样本列表

        for i in range(1, len(pos_list)):  # 如果用户点击大商品数大于1，则循环，去掉点击商品数量小于等于1的用户
            hist = pos_list[:i]  # pos_list前i个组成的新列表
            if i != len(pos_list) - 1:
                train_set.append((reviewerID, hist, pos_list[i], 1))
                train_set.append((reviewerID, hist, neg_list[i], 0))
            else:
                label = (pos_list[i], neg_list[i])
                test_set.append((reviewerID, hist, label))
        '''
        reviewerID为0时(注意，hist并不包含pos_list[i]，只包含pos_list[i]之前点击的商品，因为DIN采用attention机制，只有历史行为的attention才对后来的有影响)：
        train_set: [(0, [13179], 17993, 1), (0, [13179], 28883, 0),..., (0, [13179, 17993, 28326], 29247, 1), (0, [13179, 17993, 28326], 490, 0)]
        test_set:  [(0, [13179, 17993, 28326, 29247], (62275, 5940))]
        '''

    random.shuffle(train_set)  # 用于将列表中的元素打乱
    random.shuffle(test_set)

    assert len(test_set) == user_count  # 在条件不满足程序运行的情况下直接返回错误，而不必等待程序运行后出现崩溃的情况
    # assert(len(test_set) + len(train_set) // 2 == reviews_df.shape[0])

    with open('dataset.pkl', 'wb') as f:
        pickle.dump(train_set, f, pickle.HIGHEST_PROTOCOL)  # 2608764 (15558, [4965, 14018, 748, 44950, 22680, 19107, 55540], 12128, 0)
        pickle.dump(test_set, f, pickle.HIGHEST_PROTOCOL)  # 192403 (15558, [4965, 14018, 748, 44950, 22680, 19107, 55540], 12128, 0)
        pickle.dump(cate_list, f, pickle.HIGHEST_PROTOCOL)  # 63001 (15558, [4965, 14018, 748, 44950, 22680, 19107, 55540], 12128, 0)
        pickle.dump((user_count, item_count, cate_count), f, pickle.HIGHEST_PROTOCOL)  # user_count: 192403

