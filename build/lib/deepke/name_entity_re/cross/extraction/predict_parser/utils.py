#!/usr/bin/env python
# -*- coding: utf-8 -*-
import re


def fix_unk_from_text(span, text, unk='<unk>'):
    """
    Find span from the text to fix unk in the generated span
    从 text 中找到 span，修复span

    Example:
    span = "<unk> colo e Bengo"
    text = "At 159 meters above sea level , Angola International Airport is located at Ícolo e Bengo , part of Luanda Province , in Angola ."

    span = "<unk> colo e Bengo"
    text = "Ícolo e Bengo , part of Luanda Province , in Angola ."

    span = "Arr<unk> s negre"
    text = "The main ingredients of Arròs negre , which is from Spain , are white rice , cuttlefish or squid , cephalopod ink , cubanelle and cubanelle peppers . Arròs negre is from the Catalonia region ."

    span = "colo <unk>"
    text = "At 159 meters above sea level , Angola International Airport is located at e Bengo , part of Luanda Province , in Angola . coloÍ"

    span = "Tarō As<unk>"
    text = "The leader of Japan is Tarō Asō ."

    span = "Tar<unk> As<unk>"
    text = "The leader of Japan is Tarō Asō ."

    span = "<unk>Tar As<unk>"
    text = "The leader of Japan is ōTar Asō ."
    """
    if unk not in span:
        return span

    def clean_wildcard(x):
        sp = ".*?()[]+"
        return re.sub("("+"|".join([f"\\{s}" for s in sp])+")", "\\\\\g<1>", x)

    match = r'\s*\S+\s*'.join([clean_wildcard(item.strip()) for item in span.split(unk)])

    result = re.search(match, text)

    if not result:
        return span
    return result.group().strip()


def test_fix_unk_from_text():

    span_text_list = [
        ("<unk> colo e Bengo",
         "At 159 meters above sea level , Angola International Airport is located at Ícolo e Bengo , part of Luanda Province , in Angola .",
         "Ícolo e Bengo"),
        ("<unk> colo e Bengo",
         "Ícolo e Bengo , part of Luanda Province , in Angola .",
         "Ícolo e Bengo"),
        ("Arr<unk> s negre",
         "The main ingredients of Arròs negre , which is from Spain , are white rice , cuttlefish or squid , cephalopod ink , cubanelle and cubanelle peppers . Arròs negre is from the Catalonia region .",
         "Arròs negre"),
        ("colo <unk>",
         "At 159 meters above sea level , Angola International Airport is located at e Bengo , part of Luanda Province , in Angola . coloÍ",
         "coloÍ"),
        ("Tarō As<unk>", "The leader of Japan is Tarō Asō .", "Tarō Asō"),
        ("Tar<unk> As<unk>", "The leader of Japan is Tarō Asō .", "Tarō Asō"),
        ("<unk>Tar As<unk>", "The leader of Japan is ōTar Asō .", "ōTar Asō"),
        ("Atatürk Monument ( <unk> zmir )",
         "The Atatürk Monument ( İzmir ) can be found in Turkey .",
         "Atatürk Monument ( İzmir )"),
        ("The Atatürk Monument [ <unk> zmir ]",
         "The Atatürk Monument [ İzmir ] can be found in Turkey .",
         "The Atatürk Monument [ İzmir ]")
    ]

    for span, text, gold in span_text_list:
        print(span, '|', fix_unk_from_text(span, text))
        assert fix_unk_from_text(span, text) == gold


if __name__ == "__main__":
    test_fix_unk_from_text()
