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
from sklearn.decomposition import PCA

np.random.seed(seed=2022)



class ShapeVector(object):
    """
    形状データ用のクラス
    
    Args:
        .name: 形状名
        .data: １次元の形状データ
        .datalength: 形状データの長さ
                     --> PCAはデータ長さが同一を仮定する
    """
    def __init__(self, name, data):
        # Noneの変数は定義後に追加される
        self.name = name
        self.data = np.array(data)
        self.datalength = data.shape[0]
        self.normalized_data = None  # normalized data
        self.ave = None
        self.residual = None



from sklearn.decomposition import PCA

class ModulePCA(object):
    """
    形状の座標数列ベクトルに前処理を加え、PCAを実行
    """
    def __init__(self, shapes, 
                n_components=2, 
                random_state=0):
        """
        Args:
            shapes(dict): 形状情報の辞書
        """
        self.shapes = shapes
        self.n_components = n_components
        self.random_state = random_state

        # 実行中に定義
        self.x_max = None
        self.y_max = None
        self.pca = None
        self.d = None
        self.D = None

    def __call__(self):
        """
        Return: 
            ***(***): ***
        """
        # x軸の原点補正
        self.shapes = self.calibrate_x_coord_origin(self.shapes)
        # 規格化操作に必要なx, y成分の最大値を取得
        self.x_max, self.y_max = self.get_x_and_y_value(self.shapes)
        # 規格化(範囲:0 ~ 1)
        self.shapes = self.normalize_coordinate_values(self.shapes)
        """
        Note: 実行不要

        # 形状を規格化
        self.shapes = self.normalize_coordinate_values(self.shapes, self.transformer[self.transformer_type])
        # 形状の平均値を算出
        self.shapes = self.average_shapes(self.shapes)
        # 各形状の平均値からの残差を算出
        self.shapes = self.residual_from_average(self.shapes)
        """
        # 各形状データを集約してデータセットを作成
        self.X = self.prepare_dataset(self.shapes)
        # PCA実行
        self.pca, self.X_pca, self.X_pca_inv = self.perform_pca(self.X)
        # 寄与率
        self.d = self.contribution_rate(self.pca)
        # 累積寄与率
        self.D = self.cumulative_contribution_rate(self.d)

        return None

    def calibrate_x_coord_origin(self, shapes):
        """
        x軸の原点補正
        
        Args:
            shapes(dict): dictionary of each shape instance
                --> key: filename of CSV(shape)
                --> value: instance from class "ShapeVector"
        Returns:
            shapes(dict): dictionary of each shape instance
        """
        # x座標成分をゼロ点補正
        for key in shapes.keys():
            shape_x = shapes[key].data[0::2]  # 偶数成分
            x_min = np.min(shape_x)
            shapes[key].data[0::2] = shape_x - x_min

        return shapes

    def get_x_and_y_value(self, shapes):
        """
        規格化操作に必要なx, y成分の最大値を取得
        
        Args:
            shapes(dict): dictionary of each shape instance
                --> key: filename of CSV(shape)
                --> value: instance from class "ShapeVector"
        Returns:
            x_max, y_max(float): max values of x- and y-axis components
        """
        # x, y成分の最大値を取得, x座標成分をゼロ点補正
        x_max, y_max = 0, 0
        for key in shapes.keys():
            shape_x = shapes[key].data[0::2]  # 偶数成分
            shape_y = shapes[key].data[1::2]  # 奇数成分
            # 最大値の判定と保存
            _x_max = np.max(shape_x)
            _y_max = np.max(shape_y)
            if _x_max > x_max:
                x_max = _x_max
            if _y_max > y_max:
                y_max = _y_max
        
        return x_max, y_max

    def normalize_coordinate_values(self, shapes):
        """
        規格化
        Normalize the coordinate values of each shape,
        where the values are stored into ".normalized_data" method
        in a instance "shapes" of class "ShapeVector".
        
        Args:
            shapes(dict): dictionary of each shape instance
                --> key: filename of CSV(shape)
                --> value: instance from class "ShapeVector"
        Returns:
            shapes(dict): dictionary of each shape instance
        """
        for key in shapes.keys():
            # 初期化
            shapes[key].normalized_data = shapes[key].data.copy()
            # 規格化を実行
            # 偶数成分
            shapes[key].normalized_data[0::2] = shapes[key].data[0::2] / self.x_max
            # 奇数成分
            shapes[key].normalized_data[1::2] = shapes[key].data[1::2] / self.y_max

        return shapes

    def inverse_normalize_coordinate_values(self, shapes):
        """
        規格化の逆変換
        Inverse-normalization for the coordinate values of each shape,
        where the values are stored into ".inv_normalized_data" method
        
        Args:
            shapes(dict): dictionary of each shape instance
                --> key: filename of CSV(shape)
                --> value: instance from class "ShapeVector"
        Returns:
            shapes(dict): dictionary of each shape instance
        """
        for key in shapes.keys():
            # 初期化
            shapes[key].inv_normalized_data = shapes[key].data.copy()
            # 偶数成分
            shapes[key].inv_normalized_data[0::2] = self.x_max * shapes[key].normalized_data[0::2]
            # 奇数成分
            shapes[key].inv_normalized_data[1::2] = self.y_max * shapes[key].normalized_data[1::2]

        return shapes

    def average_shapes(self, shapes):
        """
        Add average quantity of reference shapes
        into a instance "shapes" of class "ShapeVector".
        
        Args:
            shapes(dict): dictionary of each shape instance
                --> key: filename of CSV(shape)
                --> value: instance from class "ShapeVector"
        Returns:
            shapes(dict): dictionary of each shape instance
        """
        data = []
        # 辞書型変数の各keyごとに平均量を計算
        for key in shapes.keys():
            data.append(shapes[key].normalized_data)
        ave = np.average(data, axis=0)
        # 各形状インスタンスに平均量を追加
        for key in shapes.keys():
            shapes[key].average = ave

        return shapes

    def residual_from_average(self, shapes):
        """
        Add residual from average quantity of reference shape
        into a instance "shapes" of class "ShapeVector".
        
        Args:
            shapes(dict): dictionary of each shape instance
                --> key: filename of CSV(shape)
                --> value: instance from class "ShapeVector"
        Returns:
            shapes(dict): dictionary of each shape instance
        """
        # residual of each shape
        for key in shapes.keys():
            shapes[key].residual = shapes[key].normalized_data - shapes[key].average
        return shapes

    def prepare_dataset(self, shapes):
        """
        各形状データを集約してデータセットを作成
        """
        X = []
        for key in shapes.keys():
            X.append(shapes[key].normalized_data)
            
        return np.array(X)

    def perform_pca(self, X):
        """
        PCA実行
        """
        pca = PCA(n_components=self.n_components, random_state=self.random_state)
        pca.fit(X)
        X_pca = pca.transform(X)
        X_pca_inv = pca.inverse_transform(X_pca)

        return pca, X_pca, X_pca_inv

    def contribution_rate(self, pca):
        """
        寄与率
        Calculate contribution rate
        
        Args:
            pca(instance): created from sklearn.decomposition
        Return:
            d(ndarray): contribution rate
        """
        return np.array(pca.explained_variance_ratio_)

    def cumulative_contribution_rate(self, d):
        """
        累積寄与率
        Calculate cumulative contribution rate
        
        Args:
            d(NumPy array): contribution rate
        Return:
            D(NumPy array): cumulative contribution rate
        """
        def _cumulative_summation(_x):
            """
            Calculate cumulative summation
            
            _x(NumPy array): sequence-value list
            """
            _sum = 0.
            for _i in range(len(_x)):
                _sum += _x[_i]

            return _sum

        # Calculate D(cumulative contribution rate)
        D = []
        for i in range(len(d)):
            D.append(_cumulative_summation(d[:i+1]))

        return np.array(D)

    def plot_normalized_shapes(self):
        """
        規格後の形状を確認
        """
        fig, axes = plt.subplots(3, 4, figsize=(15, 10), tight_layout=True)
        keys = list(self.shapes.keys())
        for i in range(3):
            for j in range(4):
                # 規格化後
                shape_x = self.shapes[keys[i+j]].data[0::2]  # 偶数成分
                shape_y = self.shapes[keys[i+j]].data[1::2]  # 奇数成分
                axes[i][j].scatter(shape_x, shape_y, color='b')
                axes[i][j].plot(shape_x, shape_y, color='b')
                # 規格化後
                shape_x = self.shapes[keys[i+j]].normalized_data[0::2]  # 偶数成分
                shape_y = self.shapes[keys[i+j]].normalized_data[1::2]  # 奇数成分
                axes[i][j].scatter(shape_x, shape_y, color='r')
                axes[i][j].plot(shape_x, shape_y, color='r')
                #
                # axes[i][j].set_xlim(-1, 1)
                # axes[i][j].set_ylim(0, 1)
        plt.show()
        plt.close()

    def plot_reconstructed_shapes(self):
        """
        PCA復元後の形状を確認
        """
        fig, axes = plt.subplots(3, 4, figsize=(15, 10), tight_layout=True)
        keys = list(self.shapes.keys())
        for i in range(3):
            for j in range(4):
                # 復元前
                shape_x = self.X[i+j][0::2]  # 偶数成分
                shape_y = self.X[i+j][1::2]  # 奇数成分
                axes[i][j].scatter(shape_x, shape_y, color='b')
                axes[i][j].plot(shape_x, shape_y, color='b')
                # 復元後
                shape_x = self.X_pca_inv[i+j][0::2]  # 偶数成分
                shape_y = self.X_pca_inv[i+j][1::2]  # 奇数成分
                axes[i][j].scatter(shape_x, shape_y, color='r')
                axes[i][j].plot(shape_x, shape_y, color='r')
        plt.show()
        plt.close()

    def plot_cumulative_and_contribution_rate(self):
        """
        累積and寄与率を描画
        """
        plt.figure(figsize=(6, 6))
        plt.scatter(np.arange(self.n_components), self.d, label='contribution_rate', color='b')
        plt.plot(np.arange(self.n_components), self.d, label='contribution_rate', color='b')
        plt.scatter(np.arange(self.n_components), self.D, label='contribution_rate', color='r')
        plt.plot(np.arange(self.n_components), self.D, label='contribution_rate', color='r')
        plt.xlabel('Number of PCA components')
        plt.ylabel('Rate')
        plt.show()
        plt.close()



if __name__=="__main__":
    # 形状を構成する離散点列の集合データ
    # 辞書型 --> key: ファイル(形状)名, value: CSVデータ
    shapes = {}

    k = -1
    for _1dim_vector in list_1dim_vector:
        k += 1
        key = f"id_{k}"
        shapes[key] = ShapeVector(key, _1dim_vector)

    print(shapes["id_0"].datalength)


    # 重複要素を削除後の要素数で判断
    datalengths = []
    for key in shapes.keys():
        datalengths.append(shapes[key].datalength)

    def check_datalengh(data):
        if len(set(data)) == 1:
            print('Data length are ALL SAME.')
            print('--> OK!')
        else:
            print('Data length are NOT ALL SAME.')
            print('--> NO!')

    check_datalengh(datalengths)


    # PCA実行
    n_components = 20
    pca_shapes = ModulePCA(shapes, 
                            n_components=n_components,
                            random_state=2022)
    pca_shapes()
    pca_shapes.plot_normalized_shapes()
    pca_shapes.plot_reconstructed_shapes()
    pca_shapes.plot_cumulative_and_contribution_rate()





