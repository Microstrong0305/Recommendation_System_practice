# Recommendation_System_practice
记录学习推荐系统的demo
1. [基于TF-IDF算法实现商品标题的关键词提取](./基于标签的推荐/6-1.py)
	- 使用商品和对应的短标题数据作为数据集，案例来源于《推荐系统开发实战》
1. [利用标签推荐算法实现艺术家的推荐](./基于标签的推荐/6-2.py)
	- 使用Last.fm数据集，案例来源于《推荐系统开发实战》
1. [简单的实现推荐系统的召回模型和排序模型，其中召回模型使用协同过滤算法，排序模型使用gbdt+lr算法](./cf_gbdt_lr/)
	- 使用的数据为ml-100k的数据，data_process.py为数据处理脚本, gbdt_lr.py为gbdt和lr的排序模型的训练脚本, cf_gbdt_lr_prdict.py为融合ALS和gbdt_lr整体的预测，其中ALS为召回模型。
	- 案例来源： [wuxinping1992/cf_gbdt_lr](https://github.com/wuxinping1992/cf_gbdt_lr)
1. [ALS_Lightfm-关注隐式反馈-xingwudao](./learning-to-rank-with-implicit-matrix-master/)
	- 关注隐式反馈，探索了一些隐式反馈做推荐的方法。代码对应博客：[为什么一定要重视隐式反馈？](https://mp.weixin.qq.com/s/lidie27y4obx4St3uHb8CA)
	- 案例来源：[xingwudao/learning-to-rank-with-implicit-matrix](https://github.com/xingwudao/learning-to-rank-with-implicit-matrix)
1. [Multi-Armed Bandit: epsilon-greedy](./cold_start_EE)
1. [FM（Factorization Machines）实践](./FM)
	- [FM（Factorization Machines）的理论与实践 - 小孩不笨的文章 - 知乎](https://zhuanlan.zhihu.com/p/50426292)
		- 本文使用的数据是movielens-100k，数据包括u.item，u.user，ua.base及ua.test。 | [GitHub](https://github.com/LLSean/data-mining)
	- [《推荐系统算法实践》，黄美灵著，第7章因子分解机](./rs_huangmeiling)
	- [MovieLen20M-FM-U2I-I2i-Faiss](https://github.com/yrwanziqi/MovieLen20M-FM-U2I-I2i-Faiss)
1. [LightFM练习]()
	- [How to build a Movie Recommender System in Python using LightFm](https://towardsdatascience.com/how-to-build-a-movie-recommender-system-in-python-using-lightfm-8fa49d7cbe3b) | [GitHub](https://github.com/amkurian/movie-recommendation-system)
	- [How to Build a Scalable Recommender System for an e-commerce with LightFM in python](https://towardsdatascience.com/if-you-cant-measure-it-you-can-t-improve-it-5c059014faad) | [AI前线翻译](https://mp.weixin.qq.com/s/ss9UlDU9lYokS-EDrPnOxQ) | [GitHub-code](https://github.com/nxs5899/Recommender-System-LightFM)
1. [FNN(FM+DNN)](./FNN)
	- [基于深度学习的推荐(二)：基于FM初始化的FNN](https://mp.weixin.qq.com/s/rWK1NVO87Zt37AFvgpm4yw) | [GitHub](https://github.com/wyl6/Recommender-Systems-Samples/tree/master/RecSys%20And%20Deep%20Learning/DNN/fnn)
1. [Wide & Deep](./Wide%20&%20Deep)
	- 案例来源：《推荐系统算法实践》，黄美玲，第12章 Wide & Deep模型
1. [Deep-Cross-Net](./Deep-Cross-Net)
	- 案例来源：[FitzFan/Deep-Cross-Net](https://github.com/FitzFan/Deep-Cross-Net)

