from typing import Dict


def list_dictionary(d, n_tab=-1):
    if isinstance(d, list):
        for i in d:
            list_dictionary(i, n_tab)
    elif isinstance(d, dict):
        n_tab += 1
        for key, value in d.items():
            if key == '<end>':
                print("{}{}".format(" " * n_tab, key))
            else:
                print("{}{}".format(" " * n_tab, key))
                list_dictionary(value, n_tab)
    else:
        print("{}{}".format("\t" * n_tab, d))


def print_tree(tree):
    list_dictionary(tree)


def get_label_name_tree(label_name_list, tokenizer, end_symbol='<end>'):
    sub_token_tree = dict()

    label_tree = dict()
    for typename in label_name_list:
        after_tokenized = tokenizer.encode(typename, add_special_tokens=False)
        # label_tree[typename] = tokenizer.convert_ids_to_tokens(after_tokenized)
        label_tree[typename] = after_tokenized

    for _, sub_label_seq in label_tree.items():
        parent = sub_token_tree
        for value in sub_label_seq:
            if value not in parent:
                parent[value] = dict()
            parent = parent[value]

        parent[end_symbol] = None

    return sub_token_tree


class PrefixTree:
    def __init__(self, label_name_list, tokenizer, end_symbol='<end>'):
        self.label_name_list = label_name_list
        self._tokenizer = tokenizer
        self.label_name_tree = get_label_name_tree(label_name_list, tokenizer, end_symbol)
        self._end_symbol = end_symbol

    def is_end_of_tree(self, tree: Dict):
        return len(tree) == 1 and self._end_symbol in tree
