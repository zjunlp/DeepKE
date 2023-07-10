import torch
from collections import OrderedDict
from argparse import ArgumentParser
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import dispatch_model
from accelerate.utils import get_balanced_memory, infer_auto_device_map

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--multi-gpu", default=False, action="store_true", help="Whether to use multi gpu.")
    parser.add_argument("--device-map", default=None, help="the device map for accelerate.")
    parser.add_argument("--delta", default=None, type=str, help="The delta path.")
    args = parser.parse_args()
    return args

def load_delta(delta_path):
    delta_dict = torch.load(delta_path)
    delta_with_prefix = OrderedDict()
    for k, v in delta_dict.items():
        # CpmBeeModel -> CpmBeeForCasualLM
        if k.startswith("encoder.") or k.startswith("input_embedding.") or k.startswith("position_bias."):
            delta_with_prefix["cpmbee."+k] = v
        else:
            delta_with_prefix[k] = v
    del delta_dict
    return delta_with_prefix

def main():
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained("openbmb/cpm-bee-10b", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained("openbmb/cpm-bee-10b", trust_remote_code=True).cuda()

    if args.delta is not None:
        from opendelta import LoraModel
        delta_model = LoraModel(backbone_model=model, modified_modules=["project_q", "project_v"], backend="hf")
        model.load_state_dict(load_delta(args.delta), strict=False)

    if args.multi_gpu:
        if args.device_map is None:
            max_memory = get_balanced_memory(
                model,
                no_split_module_classes=["CpmBeeTransformerBlock"]
            )
            device_map = infer_auto_device_map(model, max_memory=max_memory, no_split_module_classes=["CpmBeeTransformerBlock"])
            # make sure the data on the same device when projecting hidden states to logits.
            device_map["cpmbee.encoder.output_layernorm"] = device_map["cpmbee.input_embedding"] = 0
        else:
            device_map = args.device_map

        model = dispatch_model(model, device_map=device_map)

    res = model.generate(
        [
            {"input": "今天天气是真的", "<ans>": ""},
            {"input": "NGC 6231是一个位于天蝎座的疏散星团，天球座标为赤经16时54分，赤纬-41度48分，视觉观测大小约45角分，亮度约2.6视星等，距地球5900光年。NGC 6231年龄约为三百二十万年，是一个非常年轻的星团，星团内的最亮星是5等的天蝎座 ζ1星。用双筒望远镜或小型望远镜就能看到个别的行星。NGC 6231在1654年被意大利天文学家乔瓦尼·巴蒂斯特·霍迪尔纳（Giovanni Battista Hodierna）以Luminosae的名字首次纪录在星表中，但是未见记载于夏尔·梅西耶的天体列表和威廉·赫歇尔的深空天体目录。这个天体在1678年被爱德蒙·哈雷（I.7）、1745年被夏西亚科斯（Jean-Phillippe Loys de Cheseaux）（9）、1751年被尼可拉·路易·拉卡伊（II.13）分别再次独立发现。", "question": "NGC 6231的经纬度是多少？", "<ans>": ""}
        ],
        tokenizer,
        max_new_tokens=100,
        repetition_penalty=1.1
    )
    print(res)

if __name__ == "__main__":
    main()
