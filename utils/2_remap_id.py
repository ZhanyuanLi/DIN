import random
import pickle
import numpy as np

random.seed(1234)


def build_map(df, col_name):
    key = sorted(df[col_name].unique().tolist())
    ''' Series.unique()：按顺序返回Series中的唯一值，以Numpy数组的形式。.tolist()：让数据或列矩阵转为列表
    <class 'pandas.core.series.Series'>
    0        0528881469
    1        0594451647
    2        0594481813
    ...
    63000    B00LGQ6HL8
    -->
    <class 'numpy.ndarray'>
    ['0528881469' '0594451647' '0594481813' ... 'B00L21HC7A' 'B00L3YHF6O' 'B00LGQ6HL8']
    -->
    <class 'list'>
    ['0528881469', '0594451647', '0594481813',..., 'B00LGQ6HL8']
    '''
    m = dict(zip(key, range(len(key))))  # zip转dict，key: 字符串, value: index
    df[col_name] = df[col_name].map(lambda x: m[x])  # index替换col_name中原本的字符串
    '''
    <class 'pandas.core.series.Series'>
    0            0
    1            1
    2            2
    3            3
    ...
    63000    63000
    '''
    return m, key


if __name__ == '__main__':
    with open('../raw_data/reviews.pkl', 'rb') as f:  # rb：读取二进制文件
        reviews_df = pickle.load(f)  # pickle.load(f)反序列化对象，将文件中的数据解析为一个python对象
        reviews_df = reviews_df[['reviewerID', 'asin', 'unixReviewTime']]  # 提取列
    with open('../raw_data/meta.pkl', 'rb') as f:
        meta_df = pickle.load(f)
        meta_df = meta_df[['asin', 'categories']]
        meta_df['categories'] = meta_df['categories'].map(lambda x: x[-1][-1])  # 根据提供的函数对指定序列做映射
        # meta_df['categories'][0]: [['Electronics', 'GPS & Navigation', 'Vehicle GPS', 'Trucking GPS']]

    asin_map, asin_key = build_map(meta_df,
                                   'asin')  # 'asin'列已被替换成inedx。asin_map：字典形式，"商品id(字符串):index(int, 按商品id中唯一值出现的顺序)"。asin_key：唯一的商品id
    cate_map, cate_key = build_map(meta_df, 'categories')  # 目录类别
    revi_map, revi_key = build_map(reviews_df, 'reviewerID')  # 评论者ID。reviews_df.shape: (1689188, 9)

    user_count, item_count, cate_count, example_count = \
        len(revi_map), len(asin_map), len(cate_map), reviews_df.shape[0]  # 192403, 63001, 801, 1689188
    print('user_count: %d\titem_count: %d\tcate_count: %d\texample_count: %d' %
          (user_count, item_count, cate_count, example_count))

    meta_df = meta_df.sort_values('asin')  # DataFrame根据'asin'列排序，索引会跟着排序
    meta_df = meta_df.reset_index(drop=True)  # 重新设置索引，丢弃原索引
    reviews_df['asin'] = reviews_df['asin'].map(lambda x: asin_map[x])  # reviews数据中的asin字符串变为index
    reviews_df = reviews_df.sort_values(
        ['reviewerID', 'unixReviewTime'])  # DataFrame根据'reviewerID'和'unixReviewTime'列排序，索引会跟着排序
    reviews_df = reviews_df.reset_index(drop=True)
    reviews_df = reviews_df[['reviewerID', 'asin', 'unixReviewTime']]  # 只保留'reviewerID', 'asin', 'unixReviewTime'

    cate_list = [meta_df['categories'][i] for i in range(len(asin_map))]  # 获取meta中的categories列的信息，63001个
    '''
    <class 'list'>
    [738, 157, 571, 707,... , 83, 194, 258]
    '''
    cate_list = np.array(cate_list, dtype=np.int32)  # np.array: 列表转数组
    '''
    <class 'numpy.ndarray'>
    [738 157 571 707 799 798 798 714 714 714 798 798 714 798 571 339 385 541
     516 142 188 770 210 142 498 543 214 537 113 462 541 484   1 744 484 484
     ...
     543  54  86 215 216 215 342 249 660 142 558 399 100 558 558 611   8 558
     524 792 158 766 194 215  83 194 258]
    '''

    with open('../raw_data/remap.pkl', 'wb') as f:
        pickle.dump(reviews_df, f, pickle.HIGHEST_PROTOCOL)  # uid, iid
        pickle.dump(cate_list, f, pickle.HIGHEST_PROTOCOL)  # cid of iid line
        pickle.dump((user_count, item_count, cate_count, example_count),
                    f, pickle.HIGHEST_PROTOCOL)
        pickle.dump((asin_key, cate_key, revi_key), f, pickle.HIGHEST_PROTOCOL)
