# -*- coding: utf-8 -*-
# @Time : 2024/5/8 22:26
# @Author : nanji
# @Site : 
# @File : testOptimalK02.py
# @Software: PyCharm 
# @Comment : 
import warnings

warnings.filterwarnings('ignore')
from gap_statistic import OptimalK
import numpy as np
import matplotlib.pyplot as plt


def gap_statistic_K(data, range_K, pro_num):
	K = np.arange(1, range_K)
	optimalK = OptimalK(n_jobs=1, parallel_backend='joblib')
	n_clusters = optimalK(data, cluster_array=K)
	# Gap values plot
	plt.plot(optimalK.gap_df.n_clusters, optimalK.gap_df.gap_value, linewidth=3)
	plt.scatter(optimalK.gap_df[optimalK.gap_df.n_clusters == n_clusters].n_clusters,
				optimalK.gap_df[optimalK.gap_df.n_clusters == n_clusters].gap_value, s=250, c='r')
	plt.grid(True)
	plt.xlabel('Cluster Count')
	plt.ylabel('Gap Value')
	plt.title('Gap Values by Cluster Count')
	plt.savefig(f'/Users/cecilia/Desktop/K_图片/{pro_num}_gap_values_K.jpg')
	plt.cla()
	plt.show()

