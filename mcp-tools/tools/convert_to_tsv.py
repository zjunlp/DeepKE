import json
import hashlib

def generate_id(text):
    return hashlib.md5(text.encode("utf-8")).hexdigest()

def text_to_0x02_sequence(text, fill='O'):
    return '\x02'.join(text), '\x02'.join([fill] * len(text)), '\x02'.join([fill] * len(text)), '\x02'.join([str(i) for i in range(len(text))])

def write_raw_file(text, raw_path):
    data = {"text": text, "id": generate_id(text)}
    with open(raw_path, 'w', encoding='utf-8') as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")

def write_single_sentence_trigger_tsv(text, tsv_path):
    with open(tsv_path, 'w', encoding='utf-8') as f:
        f.write("text_a\tlabel\tindex\n")
        text_a = '\x02'.join(text)
        label = '\x02'.join(['O'] * len(text))
        f.write(f"{text_a}\t{label}\t0\n")

def write_single_sentence_role_tsv(text, tsv_path):
    with open(tsv_path, 'w', encoding='utf-8') as f:
        f.write("text_a\tlabel\ttrigger_tag\tindex\n")
        text_a = '\x02'.join(text)
        label = '\x02'.join(['O'] * len(text))
        trigger_tag = '\x02'.join(['O'] * len(text))
        f.write(f"{text_a}\t{label}\t{trigger_tag}\t0\n")

def input_to_raw_and_tsv(text, raw_path, tsv_role_path, tsv_trigger_path):
    write_raw_file(text, raw_path)
    write_single_sentence_role_tsv(text, tsv_role_path)
    write_single_sentence_trigger_tsv(text, tsv_trigger_path)
    print(f"已生成：\n- raw: {raw_path}\n")

# 示例
# if __name__ == "__main__":
#     input_text = "振华三部曲的《暗恋橘生淮南》终于定档了。"
#     input_to_raw_and_tsv(input_text, "user_raw.json", "user_role_tsv.tsv", "user_trigger_tsv.tsv")
