import yaml
import os

def config_from_keyboard(cfg):
    """
    让用户根据 cfg.txt 的内容逐行输入配置。如果 cfg 中已有默认值，则显示它，并询问用户是否要更改。
    """
    config = {}

    # 读取 cfg.txt 文件中的配置项
    cfg_file_path = "conf/cfg.txt"
    if not os.path.exists(cfg_file_path):
        raise FileNotFoundError(f"{cfg_file_path} 不存在，请确认文件路径是否正确。")

    with open(cfg_file_path, "r") as file:
        # 逐行读取每个键
        for line in file:
            key = line.strip().replace(":", "")  # 去掉冒号和换行符
            if key:
                # 获取现有的配置值，或设置为 None
                default_value = cfg.get(key, None)
                # 如果 cfg 中已有值，提示用户选择是否使用现有值
                if default_value is not None:
                    user_input = input(f"当前 {key} 的值为 '{default_value}'，按 Enter 保持不变或输入新值： ")
                else:
                    user_input = input(f"请输入 {key} 的值 (留空则为 null)： ")

                # 如果用户未输入内容，使用默认值；如果没有默认值，设置为 None
                config[key] = user_input if user_input else default_value

    # 将最终配置写入 config.yaml
    with open("config.yaml", "w") as yaml_file:
        yaml.dump(config, yaml_file, allow_unicode=True, default_flow_style=False)
    print("config.yaml 文件已成功生成。")
