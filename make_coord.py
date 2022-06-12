# """Scriptの更新を随時読み込む"""
# %load_ext autoreload
# %autoreload 2

import warnings
warnings.filterwarnings('ignore')
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set()
from scipy.interpolate import interp1d

np.random.seed(seed=2022)



class GenerateCoordinateVector(object):
    """
    ## 形状の代表点から座標数列ベクトルを生成
    ## 手順: １形状ごとに以下を実行
        - 時計周りルールで代表点のペアを作成
        - 代表点ペアの長さを算出
        - 長さの総和と分割数から最小長さを算出
        - 座標数列ベクトルを作成

    Generate a coordinate sequence vector
    from the representative points of the shape.
    """

    def __init__(self, num_point, df):
        """
        Args:
            num_point(int): 形状全体の分割数
            df(DataFrame): 形状１つの代表点データ
        """
        super(GenerateCoordinateVector, self).__init__()
        # 分割数: 最終的なベクトルの成分数は(num_point - 2)になる
        self.num_point = num_point
        self.df = df

    def __call__(self):
        """
        Return: 
            vec_1dim(NumPy array): 1次元の座標数列ベクトル
                                   ex. (x1, y1, x2, y2, x3, y3, ..)
        """
        return self.process_for_one_shape(self.num_point, self.df)

    def process_for_one_shape(self, num_point, df):
        """
        1形状単位での処理関数（座標数列ベクトルを作成）
            Note: 代表点間ペアの各辺を分割し、座標数列ベクトルを作成
        
        Args:
            num_point(int): 形状全体の分割数
            df(DataFrame): 形状１つの代表点データ
        
        Returns:
            vec_1dim(NumPy array): 1次元の座標数列ベクトル
                                   ex. (x1, y1, x2, y2, x3, y3, ..)
        """
        # 代表点のペア(辺)を作成
        pairs = self.get_pairs(df)
        # 代表点間の長さを算出
        lengths = self.get_length(pairs)
        # 各辺の分割数を計算
        num_p_edge = self.assign_division_number_at_each_edge(num_point, lengths)
        # 各辺ごとの座標ベクトル
        vectors = self.get_coord_vector_each_edge(pairs, num_p_edge)
        # 各辺の座標ベクトルを結合
        vec_x, vec_y = self.combine_vectors_of_each_side(vectors)
        # xとy成分ベクトルを結合
        vec_1dim = self.rearrange_vector(vec_x, vec_y)

        return vec_1dim

    def get_pairs(self, df):
        """
        代表点のペアを作成
        Create a pair list of each coordinate points
        
        Args:
            df(DataFrame): table data of shape information
        Return:
            pairs(NumPy array): pair list of each coordinate points,
                                shape(number of pairs, start or end points, x or y)
        """
        pairs = []
        # 最終ペアは除く（今回の対象形状を踏まえて）
        for i in range(len(df) - 1):  # range(len(df)):
            pair = []
            # start point
            x_s = df['x'].iloc[i]  # [i - 1]
            y_s = df['y'].iloc[i]  # [i - 1]
            # end point
            x_e = df['x'].iloc[i + 1]  # [i]
            y_e = df['y'].iloc[i + 1]  # [i]
            # Store to the list
            pairs.append([[x_s, y_s], [x_e, y_e]])

        return np.array(pairs)

    def get_length(self, pairs):
        """
        ### 代表点間の長さを算出
        Calculate a length of each point pair
        
        Args:
            pairs(ndarray): pair list of each coordinate points,
                                shape(number of pairs, start or end points, x or y)
        Return:
            lengths(ndarray): length of each edge
        """
        lengths = []
        for i in range(pairs.shape[0]):
            # start point
            x_s = pairs[i][0][0]
            y_s = pairs[i][0][1]
            # end point
            x_e = pairs[i][1][0]
            y_e = pairs[i][1][1]
            # Calculate a length
            length = np.sqrt(np.power(x_e - x_s, 2) + np.power(y_e - y_s, 2))
            # Store to the list
            lengths.append(length)

        return np.array(lengths)

    def assign_division_number_at_each_edge(self, num_point, lengths):
        """
        Args:
            num_point(int): 補間後のノード数
            lengths(list): 各辺の長さ
        Return:
            num_p_edge(ndarray): 各辺の分割数
            --> np.sum(num_p_edge) == num_point が必要
        """
        # 各辺の長さの全体和に対する比率
        weight_lengths = lengths / np.sum(lengths)
        # 各辺の分割数の概算での初期値を設定
        num_p_edge = []
        for weight in weight_lengths:
            num_p_edge.append(int(weight * num_point))
        num_p_edge = np.array(num_p_edge)
        # 初期値の和が想定数(num_point)に対して不当でないか確認
        if np.sum(num_p_edge) > num_point:
            # 初期値の和は想定数(num_point)以下が必要
            print("ERROR occured!!")
            print("各辺の分割数の概算での初期値で想定外の状況が発生しました..")
            sys.exit()
        # 不足分の分割数を追加し、総数がnum_pointに一致するようにする
        while np.sum(num_p_edge) != num_point:
            _num_p_edge = num_p_edge
            # 各辺の分割数を仮想的に+1し、追加後で最小長さが最も小さい辺に分割数を+1とする
            _num_p_edge = _num_p_edge + 1
            _unit_lengths = lengths / _num_p_edge
            _idx = np.argmax(_unit_lengths)
            num_p_edge[_idx] += 1

        return np.array(num_p_edge)

    def get_coord_vector_each_edge(self, pairs, num_p_edge):
        """
        Create point-sequence vectors for each edge
        
        Args:
            pairs(ndarray): pair list of each coordinate points,
                                shape(number of pairs, start or end points, x or y)
            num_p_edge(ndarray): ***
        Return:
            vectors(list): number of division for each edge
                Note: numbers of division of each edge are different,
                      so we can't use NumPy array.
        """
        vectors = []
        for i in range(pairs.shape[0]):
            # start point
            x_s = pairs[i][0][0]
            y_s = pairs[i][0][1]
            # end point
            x_e = pairs[i][1][0]
            y_e = pairs[i][1][1]
            # unit length
            unit_x = (x_e - x_s) / num_p_edge[i]
            unit_y = (y_e - y_s) / num_p_edge[i]
            """
            各辺の点列ベクトルを作成
            * 最初の辺の始点が含まれないアルゴリズムとなっている
            --> if文で処理して考慮
            """
            # create verctor at each edge
            if i == 0:
                _x, _y = x_s, y_s
                x_tmp, y_tmp = [_x], [_y]
            else:
                _x, _y = x_s, y_s
                x_tmp, y_tmp = [], []
            # 補間点を追加
            for _ in range(num_p_edge[i]):
                _x += unit_x
                _y += unit_y
                x_tmp.append(_x)
                y_tmp.append(_y)
            vectors.append([x_tmp, y_tmp])

        return vectors

    def combine_vectors_of_each_side(self, vectors):
        """
        各辺の座標ベクトルを結合
        Combine the coordinate vectors of each side
        
        Args:
            aaa
        Returns:
            bbb
        """
        vec_x, vec_y = [], []
        for i in range(len(vectors)):
            vec_x.extend(vectors[i][0])
            vec_y.extend(vectors[i][1])

        return vec_x, vec_y

    def rearrange_vector(self, vec_x, vec_y):
        """
        ### xとy成分ベクトルを結合
        ### 後のPCA実行時は１次元の座標数列ベクトルで行うため
        Combine and Rearrange the x and y vectors
            ex. x = [0, 1, 2, 3, 4]
                y = [5, 6, 7, 8, 9]
                --> vec_1dim = [0, 5, 1, 6, 2, 7, 3, 8, 4, 9]
        
        Args:
            vec_x(NumPy array): x-component coordinate vector
            vec_y(NumPy array): y-component coordinate vector
        Returns:
            vec_1dim(NumPy array): 1-dimensional coordinate vector
        """
        vec_1dim = []
        for i in range(len(vec_x)):
            vec_1dim.append(vec_x[i])
            vec_1dim.append(vec_y[i])

        return np.array(vec_1dim)



if __name__=="__main__":
    """
    データを生成
    """
    # 内挿補間数
    num_point = 64

    list_1dim_vector = []
    for _df in list_df:
        onedim_vector = GenerateCoordinateVector(num_point, _df)
        _1dim_vector = onedim_vector()
        list_1dim_vector.append(_1dim_vector)
    print(np.array(list_1dim_vector).shape)


    """
    生成1次元ベクトルを確認
    """
    fig, axes = plt.subplots(3, 4, figsize=(15, 10), tight_layout=True)
    for i in range(3):
        for j in range(4):
            # 生成補間データ
            axes[i][j].scatter(list_1dim_vector[i+j][0::2], list_1dim_vector[i+j][1::2], color='b')
            axes[i][j].plot(list_1dim_vector[i+j][0::2], list_1dim_vector[i+j][1::2], color='b')
            # 元データ
            axes[i][j].scatter(list_df[i+j]['x'], list_df[i+j]['y'], color='r')
            # axes[i][j].set_xlim(-1, 1)
            axes[i][j].set_ylim(0, 1)
    plt.show()
    plt.close()


    """
    生成1次元ベクトルの長さが一致しているかを確認
    """
    _lengths = []
    for i in range(num_loop):
        _lengths.append(len(list_1dim_vector[i]))
    _lengths = set(_lengths)
    print(len(_lengths))
    print(_lengths)
    assert len(_lengths) == 1
















