# Recommendation_System_practice
记录学习推荐系统的demo
1. [基于TF-IDF算法实现商品标题的关键词提取](./基于标签的推荐/6-1.py)
	- 使用商品和对应的短标题数据作为数据集，案例来源于《推荐系统开发实战》
1. [利用标签推荐算法实现艺术家的推荐](./基于标签的推荐/6-2.py)
	- 使用Last.fm数据集，案例来源于《推荐系统开发实战》
1. [简单的实现推荐系统的召回模型和排序模型，其中召回模型使用协同过滤算法，排序模型使用gbdt+lr算法](./cf_gbdt_lr/)
	- 使用的数据为ml-100k的数据，data_process.py为数据处理脚本, gbdt_lr.py为gbdt和lr的排序模型的训练脚本, cf_gbdt_lr_prdict.py为融合ALS和gbdt_lr整体的预测，其中ALS为召回模型。
	- 案例来源： [wuxinping1992/cf_gbdt_lr](https://github.com/wuxinping1992/cf_gbdt_lr)