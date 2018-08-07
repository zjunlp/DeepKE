# deepke

## 数据准备

文件（来源） | 样本 
---- | ---- 
com2abbr.txt（原始）|沙河实业股份有限公司 沙河股份
stock.sql（原始）|[stock_code,ChiName,ChiNameAbbr]<br />('000001', '平安银行股份有限公司', '平安银行');
rel_per_com.txt（原始）|非独立董事 刘旭 湖南博云新材料股份有限公司
kg_company_management.sql（原始）|[stock_code, manager_name,manager_position…]（主要部分）
per_com.txt（原始）| 董事   深圳中国农大科技股份有限公司 刘多宏
rel_per_com.txt（由前面两个生成）| 非独立董事 刘旭 湖南博云新材料股份有限公司

运行`preprocess.py`的`get_initial_sample()`，主要包括

* **初步整理原始文本**</br>
包括去掉不必要的符号、将数字用NUM替换等
* **整理远程监督的数据**</br>
得到职位相关的数据，包括所有的人和公司，放在*per_pool*中和*com_pool*中以及具有职位关系的三元组*rel_per_com*
* **初步采样**</br>
通过远程监督进行采样，目前的设定是遍历所有句子，如果两实体出现在句子中且二者在*rel_per_com*中有关系，则标记为正样本；不在*rel_per_com* 中的两实体标记为负
* **规则过滤数据**</br>
 **噪音来源：**
  * 远程监督数据源自身的噪音，如人名为‘智慧’
  * 一人多职位，
    * 如句子“B为A的实际控股人” ，在*rel_per_com* 中有「A,B,董事长」 ，句子会被标记为董事长的正样本
  * 静态的远程监督数据源和随时间动态变化的职位关系之间的冲突，
    * 对句子“A曾任B的董事长”，在 *rel_per_com* 中有「A,B,董事长」，句子会被标记为董事长的正样本
    * 对句子“任命A为B的总裁”，在 *rel_per_com* 中关于「A,B」 的关系只有「A,B,副总裁」，句子也会被标记为副总裁的正样本
    * 对句子“任命A为B的总裁”，句子中有于「A,B」，但是在 *rel_per_com* 中没有任何于「A,B」的职位信息，会被标记为负样本
    *  ...

  **正样本过滤:** </br>
   * 关系关键词必须在句子中，考虑一人多职位
   
  **负样本过滤:**</br>
   * 正则表达式识别‘A的董事长B‘这种类型的句子，回标为正样本
   * 远程监督本身的噪音，如在*per_pool* 中有「周建灿」和「周建」，有句子“金盾董事长周建灿”，直接的实体链接会标出「金盾董事，周建，董事长」


## 训练
* 运行`preprocess.py`的`train_preprocess()`，生成训练数据
* 运行`python train.py`，模型存在`../model`下
* 参数在`config.py`中进行配置，包括*GPU_ID*, *learning_rate*等

## 测试
* 运行`preprocess.py`的`predict_preprocess()`，生成可以输入模型的数据
* 运行`python test.py`，结果保存在`../result`下

## 模型
考虑到实验效果，目前使用多个二分类模型</br>
参考：[Lin et al. (2017)](http://www.aclweb.org/anthology/D15-1203).</br>
输入：句子，两实体及相应的位置信息，用于判断并列语句的辅助信息序列</br>
输出：是否具有相应关系</br>

## 结果
![pcnn result](https://github.com/zjunlp/deepke/blob/dev/result/result.png)
