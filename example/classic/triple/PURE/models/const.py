task_ner_labels = {
    'ace04': ['FAC', 'WEA', 'LOC', 'VEH', 'GPE', 'ORG', 'PER'],
    'ace05': ['FAC', 'WEA', 'LOC', 'VEH', 'GPE', 'ORG', 'PER'],
    'scierc': ['Method', 'OtherScientificTerm', 'Task', 'Generic', 'Material', 'Metric'],
    'cmeie': ['药物', '社会学', '其他', '部位', '检查', '预后', '其他治疗', '流行病学', '症状', '疾病', '手术治疗'],
}

task_rel_labels = {
    'ace04': ['PER-SOC', 'OTHER-AFF', 'ART', 'GPE-AFF', 'EMP-ORG', 'PHYS'],
    'ace05': ['ART', 'ORG-AFF', 'GEN-AFF', 'PHYS', 'PER-SOC', 'PART-WHOLE'],
    'scierc': ['PART-OF', 'USED-FOR', 'FEATURE-OF', 'CONJUNCTION', 'EVALUATE-FOR', 'HYPONYM-OF', 'COMPARE'],
    'cmeie': ['预后生存率', '多发季节', '药物治疗', '发病机制', '外侵部位', '相关（导致）', '并发症', '发病部位', '辅助检查', '病理生理', '病理分型', '手术治疗', '筛查', '发病率', '传播途径', '阶段', '影像学检查', '化疗', '相关（转化）', '风险评估因素', '转移部位', '内窥镜检查', '相关（症状）', '病因', '遗传因素', '发病性别倾向', '临床表现', '病史', '死亡率', '辅助治疗', '发病年龄', '鉴别诊断', '实验室检查', '治疗后症状', '高危因素', '多发地区', '放射治疗', '侵及周围组织转移的症状', '预防', '就诊科室', '组织学检查', '预后状况', '同义词', '多发群体'],
}

def get_labelmap(label_list):
    label2id = {}
    id2label = {}
    for i, label in enumerate(label_list):
        label2id[label] = i + 1
        id2label[i + 1] = label
    return label2id, id2label
