import pytest
from serializer import Serializer


def test_serializer_for_no_chinese_split():
    text1 = "\nI\'m  his pupp\'peer, and i have a ball\t"
    text2 = '\t叫Stam一起到nba打篮球\n'
    text3 = '\n\n现在时刻2014-04-08\t\t'

    serializer = Serializer(do_chinese_split=False)
    serial_text1 = serializer.serialize(text1)
    serial_text2 = serializer.serialize(text2)
    serial_text3 = serializer.serialize(text3)

    assert serial_text1 == ['i', "'", 'm', 'his', 'pupp', "'", 'peer', ',', 'and', 'i', 'have', 'a', 'ball']
    assert serial_text2 == ['叫', 'stam', '一', '起', '到', 'nba', '打', '篮', '球']
    assert serial_text3 == ['现', '在', '时', '刻', '2014', '-', '04', '-', '08']


def test_serializer_for_chinese_split():
    text1 = "\nI\'m  his pupp\'peer, and i have a basketball\t"
    text2 = '\t叫Stam一起到nba打篮球\n'
    text3 = '\n\n现在时刻2014-04-08\t\t'

    serializer = Serializer(do_chinese_split=True)
    serial_text1 = serializer.serialize(text1)
    serial_text2 = serializer.serialize(text2)
    serial_text3 = serializer.serialize(text3)

    assert serial_text1 == ['i', "'", 'm', 'his', 'pupp', "'", 'peer', ',', 'and', 'i', 'have', 'a', 'basketball']
    assert serial_text2 == ['叫', 'stam', '一起', '到', 'nba', '打篮球']
    assert serial_text3 == ['现在', '时刻', '2014', '-', '04', '-', '08']


if __name__ == '__main__':
    pytest.main()
