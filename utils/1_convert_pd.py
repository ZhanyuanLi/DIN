# 将原始数据json格式转换为pandas dataframe格式
import pickle
import pandas as pd


def to_df(file_path):
    with open(file_path, 'r') as fin:
        df = {}
        i = 0
        for line in fin:  # line: <class 'str'>, {...}
            df[i] = eval(line)  # eval返回传入的字符串的表达的结果。df[i]: <class 'dict'>, {...}
            i += 1
        df = pd.DataFrame.from_dict(df, orient='index')
        '''
              reviewerID        asin helpful reviewerName
        0  AO94DHGC771SJ  0528881469     NaN          NaN
        1  AO94DHGC771SJ  0528881469  [0, 0]          NaN
        2  AO94DHGC771SJ  0528881469  [0, 0]      2Cents!
        '''
        return df


if __name__ == '__main__':

    reviews_df = to_df('../raw_data/reviews_Electronics_5.json')
    with open('../raw_data/reviews.pkl', 'wb') as f:  # wb：二进制打开一个文件只用于写入，存在则覆写，不存在则创建
        pickle.dump(reviews_df, f, pickle.HIGHEST_PROTOCOL)  # pickle.dump：序列化对象，对象reviews_df保存到文件f中
    '''reviews.pkl: 
    <class 'pandas.core.frame.DataFrame'>
           reviewerID        asin  ... unixReviewTime   reviewTime
    0   AO94DHGC771SJ  0528881469  ...     1370131200   06 2, 2013
    1   AMO214LNFCEI4  0528881469  ...     1290643200  11 25, 2010
    ...
    (1689188, 9)
    '''

    meta_df = to_df('../raw_data/meta_Electronics.json')
    meta_df = meta_df[meta_df['asin'].isin(reviews_df['asin'].unique())]  # isin返回具有条件的dataframe
    meta_df = meta_df.reset_index(drop=True)  # reset_index(drop=True)：重置索引，丢弃原本index
    with open('../raw_data/meta.pkl', 'wb') as f:
        pickle.dump(meta_df, f, pickle.HIGHEST_PROTOCOL)
    '''meta.pkl: 
    <class 'pandas.core.frame.DataFrame'>
             asin  ...               brand
    0  0528881469  ...                 NaN
    1  0594451647  ...                 NaN
    ...
    (63001, 9)
    '''


