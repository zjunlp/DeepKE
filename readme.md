1. 运行环境：python3，字符编码为unicode<br>
   目前python2即使设置了'utf-8'，也会出现（__label_re中positon）一些问题，
   可能会影响后期的过滤，
2. 从database中读取kg_data没有写
3. character_level的几个问题
4. 多线程终止自动`cat sample* sample` 
5. __find_entity() 两个循环可能有问题
6. 在sample.py中定义了两种过滤规则，一种是关键字，还有一种是长度，目前将这两个规则写在sample.py中
7. 考虑到character_level，e1和e2的sent_id比较麻烦，目前不计划输入到模型中，（如果需要，可以在模型中单独为随机初始化embedding，但是没有意义，或者将embedding表示为所有word emb的average）
8. position 输入到模型中可以是负数吗？
9. 现在是所有的都要ner的，只是如果模型不需要ner，在生成数据时候就直接将ner的输入置为0
10. sample.py 的gen_train_data中最好生成一个基本版本，不需要每次都重复操作segment,ner,
11. 多分类是不是多个类别最好不要强相关？

- 一些样本pattern：<br>
  - e1的董事长e2<br>
  - e1董事长e2<br>
  - e1的董事长候选人xx,e2,xx<br>
  - e1的副总裁xx、e2、xx和xx<br>
  - e1董事长xx,xx法定代表人e2  *（如果标记这种是负样本，算是正确的负样本了：
并列句子，且在kg中没有相应triple）*<br>
  - e1董事长，法定代表人e2<br>
  - e1董事长、法定代表人e2<br>
  - e1董事长兼法定代表人e2<br>
  - e1......法定代表人e2<br>
  - e1前总经理e2 *（如果kg中存储的未更新呢）*<br> 
  - e1副总经理e2 *（如果kg已经存储的'总经理'或者'总监'呢）*<br>
  - e1担任e2总经理
  - e1曾担任e2总经理
  - e2...e1现任董事长，总经理，总裁


- 从**未知关系**中找到相对质量高的负例
    1. 如果句子被标记为'未知关系'(`rel == '未知关系'`) and `pos1-pos2 > 50` ，可能介绍性质的出问题
    2. 并列句子( `sent.count(',') > 6`) and `'，/、/；' in sent[pos1:pos2]`   
    3. 互斥关系作为负样本，（副总经理作为总经理的负样本）
    
- 对正例稍作过滤
    1. rel in sent
    2. time_related_words = ['曾经','前','曾','前任','候选人','',] 
    3. sensitive_word

- **vacab,embedding**
    1. 如果用test.txt分词之后的进行处理，之后给模型的embeddings应该是加上'E1''E2'的，（
    在训练的build_embeddings阶段，赋值为0，训练过程中得到的E1和E2保存下来，测试阶段，build_embeddings时候从保存的结果中加载二者的embeddings）
    vocab.txt在训练emb的时候生成，在vocab.py的load_from_file中加上'E1'和'E2'
    2. 但是1训练的embeddings是不是要重新保存？只是保存特殊4个字符的，或者是所有的都重新保存？如果是都重新保存的话，预训练好像就不用做了；但是如果只是保存4个的，和其他word存在不同步问题？
    3. 
    

- 更改dataset.py
   1. 将动态填充放在加载数据时
   2. for index, row in content.iterrows():
   3. rel = row.relations # 不会因为存储的顺序发生变化改变而发生错误
   4. load_data ner2id = v.get_ner_id(ner) 写的很好
   5. 
  
  
- 文件名：
    1. clean后的文本名 original_text.txt
    2. 对 orginal_text.txt进行分词或者分字之后的 seg_original_text.txt
    3. 

- 执行顺序
    1. preprocess 原始的文本 clean训练集合
    2. sample
    3. train
    4. preprocess predict.txt
    5. 