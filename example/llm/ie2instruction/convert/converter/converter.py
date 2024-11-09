class Converter:
    def __init__(self, language="zh", NAN="NAN", prefix=""):
        self.NAN = NAN
        self.prefix = prefix
        self.language = language

    def get_label_dict(self, records):
        raise NotImplementedError

    def convert(self, text, label_dict, s_schema1=""):
        output_text = self.convert_target(text, s_schema1, label_dict)
        return output_text

