

class Extracter:
    def __init__(self, language="zh", NAN="NAN", prefix="输入中包含的实体是：\n", Reject="No event found."):
        self.language = language
        self.NAN = NAN
        self.prefix = prefix
        self.Reject = Reject

    def extract(self, output):
        return self.post_process(output)