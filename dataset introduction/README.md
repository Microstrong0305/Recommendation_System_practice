# 1. MovieLens数据集介绍

MovieLens数据集保存了用户对电影的评分。基于这个数据集，我们可以测试一些推荐算法、评分预测算法。

## MovieLens 100k

该数据集记录了943个用户对1682部电影的共100,000个评分，每个用户至少对20部电影进行了评分。

- 文件u.info保存了该数据集的概要：

> 943 users
> 1682 items
> 100000 ratings

- 文件u.item保存了item的信息，也就是电影的信息，共1682部电影，其id依次是1、2、……、1682。文件中每一行保存了一部电影的信息，格式如下：

> movie id | movie title | release date | video release date | IMDb URL | unknown | Action | Adventure | Animation | Children's | Comedy | Crime | Documentary | Drama | Fantasy | Film-Noir | Horror | Musical | Mystery | Romance | Sci-Fi | Thriller | War | Western |

注意，最后19个字段保存的是该电影的类型，一个字段对应一个类型，值为0代表不属于该类型，值为1代表属于该类型，类型信息保存在文件u.genre中。

随便浏览了一下u.item文件，发现基本是20世纪90年代的电影。

- 文件u.genre保存了电影的类型信息。

- 文件u.user保存了用户的信息，共有943个用户，其id依次是1、2、……、943。文件中每一行保存了一个用户的信息，格式如下：

> user id | age | gender | occupation | zip code

- 文件u.occupation保存了用户职业的集合。

下面介绍数据集的主要文件。

- 文件u.data保存了所有的评分记录，每一行是一个用户对一部电影的评分，共有100000条记录。当然，如果某用户没有对某电影评分，则不会包含在该文件中。评分的分值在1到5之间，就是1、2、3、5这5个评分。每一行格式如下：

> user id | item id | rating | timestamp

其中，item id就是电影的id，时间戳timestamp是评分时间。我转换了下时间戳，也是在20世纪90年代。

- 文件u1.base和文件u1.test放在一起就是文件u.data。将u.data按照80%/20%的比例分成u1.base和u1.test，可以将u1.base作为训练集，u1.test作为测试集。u2、u3、u4、u5系列文件和u1类似。u1、u2、u3、u4、u5的测试集是不相交的，它们可以用来做（5折交叉验证）5 fold cross validation。

- 文件ua.base和文件ua.test也是由u.data拆分而来，在ua.test中包含了每个用户对10部电影的评分，从u.data去掉ua.test得到ua.base。ub.base和ub.test也使用了同样的生成方法。另外，ua.test和ub.test是不相交的。

## MovieLens 1M

该数据集保存的是6040个用户对3952部电影的1000209个评分记录。具体可以参考其README文件。

## MovieLens 10M

71567个用户，10681部电影，10000054条评分记录，同时多了个用户为电影设置的标签。具体可以阅读其中的README.html。

> Reference：
> [1. MovieLens数据集介绍](https://www.letiantian.me/2014-11-20-introduce-movielens-dataset/)

