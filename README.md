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