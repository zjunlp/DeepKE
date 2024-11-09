import json
import hashlib

def get_string_list(l):
    return "[" + ', '.join(l) + "]"


def get_string_dict(d):
    s_d = []
    for k, value in d.items():
        s_value =  k + ": " + "[" + ', '.join(value) + "]"
        s_d.append(s_value)
    return '{' + ', '.join(s_d) + '}'


def read_from_json(path):
    datas = []
    with open(path, 'r', encoding='utf-8') as reader:
        for line in reader:
            data = json.loads(line)
            datas.append(data)
    return datas


def write_to_json(path, datas):
    with open(path, 'w', encoding='utf-8') as writer:
        for data in datas:
            writer.write(json.dumps(data, ensure_ascii=False)+"\n")


def stable_hash(input_str):
    sha256 = hashlib.sha256()
    sha256.update(input_str.encode('utf-8'))
    return sha256.hexdigest()



wiki_cate_schema_zh_train = {
    '人物': ['别名', '国籍', '死亡地点', '出生地点', '职业', '父母', '作品', '死亡日期', '职务', '成就', '所属组织', '出生日期', '籍贯', '兄弟姊妹', '配偶'], 
    '地理地区': ['别名', '行政中心', '长度', '人口', '海拔', '面积', '宽度', '位于'], 
    '建筑': ['创建者', '别名', '长度', '名称由来', '高度', '事件', '成就', '创建时间', '面积', '宽度', '位于'], 
    '人造物件': ['别名', '制造商', '用途', '发现者或发明者', '材料', '产地', '成就', '品牌'], 
    '生物': ['主要食物来源', '别名', '用途', '长度', '高度', '学名', '分布', '父级分类单元', '宽度', '质量'], 
    '天文对象': ['别名', '绝对星等', '属于', '直径', '发现者或发明者', '名称由来', '发现时间', '质量'], 
    '组织': ['别名', '成立时间', '创办者', '成员', '子组织', '解散时间', '成就', '事件', '位于'], 
    '自然科学': ['别名', '用途', '发现者或发明者', '产地', '生成物', '组成'], 
    '医学': ['别名', '症状', '病因', '可能后果'], 
    '运输': ['别名', '车站等级', '开通时间', '长度', '线路', '车站编号', '面积', '成立或创建时间', '途径', '宽度', '位于'], 
    '事件': ['别名', '发生地点', '发生时间', '主办方', '赞助者', '获胜者', '获奖者', '参与者', '所获奖项'], 
    '作品': ['出版商', '别名', '作曲者', '作者', '编剧', '票房', '导演', '改编自', '出版日期', '产地', '成就', '角色', '表演者', '作词者']
}

wiki_cate_schema_zh = {
    '人物': ['出生地点', '出生日期', '国籍', '职业', '作品', '成就', '籍贯', '职务', '配偶', '父母', '别名', '所属组织', '死亡日期', '兄弟姊妹', '墓地'], 
    '地理地区': ['位于', '别名', '人口', '行政中心', '面积', '成就', '长度', '宽度', '海拔'], 
    '建筑': ['位于', '别名', '成就', '事件', '创建时间', '宽度', '长度', '创建者', '高度', '面积', '名称由来'], 
    '作品': ['作者', '出版时间', '别名', '产地', '改编自', '演员', '出版商', '成就', '表演者', '导演', '制片人', '编剧', '曲目', '作曲者', '作词者', '制作商', '票房', '出版平台'], 
    '生物': ['分布', '父级分类单元', '长度', '主要食物来源', '别名', '学名', '重量', '宽度', '高度'], 
    '人造物件': ['别名', '品牌', '生产时间', '材料', '产地', '用途', '制造商', '发现者或发明者'], 
    '自然科学': ['别名', '性质', '组成', '生成物', '用途', '产地', '发现者或发明者'], 
    '组织': ['位于', '别名', '子组织', '成立时间', '产品', '成就', '成员', '创始人', '解散时间', '事件'], 
    '运输': ['位于', '创建时间', '线路', '开通时间', '途经', '面积', '别名', '长度', '宽度', '成就', '车站等级'], 
    '事件': ['参与者', '发生地点', '发生时间', '别名', '赞助者', '伤亡人数', '起因', '导致', '主办方', '所获奖项', '获胜者'], 
    '天文对象': ['别名', '属于', '发现或发明时间', '发现者或发明者', '名称由来', '绝对星等', '直径', '质量'], 
    '医学': ['症状', '别名', '发病部位', '可能后果', '病因']
}

wiki_cate_schema_en_train = {
    'Person': ['place of death', 'position held', 'parent', 'sibling', 'achievement', 'date of death', 'ancestral home', 'spouse', 'country of citizenship', 'work', 'date of birth', 'member of', 'place of birth', 'occupation', 'alternative name'], 
    'Geographic_Location': ['located in', 'height', 'area', 'elevation above sea level', 'length', 'width', 'population', 'capital', 'alternative name'], 
    'Building': ['event', 'located in', 'height', 'area', 'named after', 'creator', 'length', 'width', 'achievement', 'creation time', 'alternative name'], 
    'Artificial_Object': ['brand', 'height', 'has use', 'manufacturer', 'country of origin', 'length', 'width', 'mass', 'price', 'achievement', 'discoverer or inventor', 'made from material', 'alternative name'], 
    'Creature': ['taxon name', 'diameter', 'height', 'length', 'distribution', 'parent taxon', 'width', 'mass', 'main food source', 'has use', 'alternative name'], 
    'Astronomy': ['diameter', 'discoverer', 'of', 'height', 'named after', 'length', 'width', 'mass', 'time of discovery', 'absolute magnitude', 'alternative name'], 
    'Organization': ['event', 'located in', 'member', 'dissolution time', 'location of formation', 'has subsidiary', 'founded by', 'achievement', 'date of incorporation', 'alternative name'], 'Natural_Science': ['country of origin', 'discoverer or inventor', 'has use', 'composition', 'alternative name', 'product'], 
    'Medicine': ['etiology', 'possible consequences', 'alternative name', 'symptoms and signs'], 
    'Transport': ['class of station', 'located in', 'date of official opening', 'height', 'connecting line', 'area', 'length', 'width', 'station code', 'inception', 'alternative name', 'pass'], 
    'Event': ['winner', 'scene', 'nominated by', 'organizer', 'prize-winner', 'sponsor', 'award received', 'occurrence time', 'participant', 'successful candidate', 'alternative name'], 
    'Works': ['box office', 'author', 'based on', 'screenwriter', 'characters', 'director', 'publication date', 'country of origin', 'lyrics by', 'performer', 'publisher', 'achievement', 'intended public', 'composer', 'alternative name']
}

wiki_cate_schema_en =  {
    'Person': ['place of birth', 'date of birth', 'country of citizenship', 'occupation', 'work', 'achievement', 'ancestral home', 'position', 'spouse', 'parent', 'alternative name', 'affiliated organization', 'date of death', 'sibling', 'place of death'], 
    'Geographic_Location': ['located in', 'alternative name', 'population', 'capital', 'area', 'achievement', 'length', 'width', 'elevation'], 
    'Building': ['located in', 'alternative name', 'achievement', 'event', 'creation time', 'width', 'length', 'creator', 'height', 'area', 'named after'], 
    'Works': ['author', 'publication date', 'alternative name', 'country of origin', 'based on', 'cast member', 'publisher', 'achievement', 'performer', 'director', 'producer', 'screenwriter', 'tracklist', 'composer', 'lyricist', 'production company', 'box office', 'publishing platform'], 
    'Creature': ['distribution', 'parent taxon', 'length', 'main food source', 'alternative name', 'taxon name', 'weight', 'width', 'height'], 
    'Artificial_Object': ['alternative name', 'brand', 'production date', 'made from material', 'country of origin', 'has use', 'manufacturer', 'discoverer or inventor'], 
    'Natural_Science': ['alternative name', 'properties', 'composition', 'product', 'has use', 'country of origin', 'discoverer or inventor', 'causes'], 
    'Organization': ['located in', 'alternative name', 'has subsidiary', 'date of incorporation', 'product', 'achievement', 'member', 'founded by', 'dissolution time', 'event'], 
    'Transport': ['located in', 'inception', 'connecting line', 'date of official opening', 'pass', 'area', 'alternative name', 'length', 'width', 'achievement', 'class of station'], 
    'Event': ['participant', 'scene', 'occurrence time', 'alternative name', 'sponsor', 'casualties', 'has cause', 'has effect', 'organizer', 'award received', 'winner'], 
    'Astronomy': ['alternative name', 'of', 'time of discovery or invention', 'discoverer or inventor', 'name after', 'absolute magnitude', 'diameter', 'mass'], 
    'Medicine': ['symptoms', 'alternative name', 'affected body part', 'possible consequences', 'etiology']
}


