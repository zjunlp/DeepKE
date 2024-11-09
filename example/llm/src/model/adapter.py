import torch
from typing import TYPE_CHECKING, Optional, Set, Tuple, List
from peft import PeftModel, TaskType, LoraConfig, get_peft_model

from utils.logging import get_logger
from utils.constants import LAYERNORM_NAMES
from args import FinetuningArguments, ModelArguments


if TYPE_CHECKING:
    from transformers.modeling_utils import PreTrainedModel

logger = get_logger(__name__)



def find_all_linear_modules(
    model: "PreTrainedModel",
    quantization_bit: Optional[int] = None
) -> List[str]:
    r"""
    Finds all available modules to apply lora.
    """
    if quantization_bit is not None:
        import bitsandbytes as bnb
        linear_cls = bnb.nn.Linear4bit if quantization_bit == 4 else bnb.nn.Linear8bitLt
    else:
        linear_cls = torch.nn.Linear

    output_layer_names = ["lm_head"]
    if model.config.model_type == "chatglm":
        output_layer_names.append("output_layer")

    module_names = set()
    for name, module in model.named_modules():
        if (
            isinstance(module, linear_cls)
            and not any([output_layer in name for output_layer in output_layer_names])
        ):
            module_names.add(name.split(".")[-1])

    logger.info("Found linear modules: {}".format(",".join(module_names)))
    return list(module_names)




def prepare_model_for_training(
    model: "PreTrainedModel",
    finetuning_args: "FinetuningArguments",
    output_layer_name: Optional[str] = "lm_head",
    use_gradient_checkpointing: Optional[bool] = True,
    layernorm_names: Optional[Set[str]] = LAYERNORM_NAMES
) -> "PreTrainedModel":
    r"""
    Includes:
        (1) cast the layernorm in fp32
        (2) make output embedding layer require grads
        (3) upcast the lm_head to fp32
    Inspired by: https://github.com/huggingface/peft/blob/v0.2.0/src/peft/utils/other.py#L33
    """
    if finetuning_args.upcast_layernorm:
        for name, param in model.named_parameters():
            if param.ndim == 1 and any(ln_name in name for ln_name in layernorm_names):
                param.data = param.data.to(torch.float32)
        logger.info("Upcasting weights in layernorm in float32.")

    if finetuning_args.neft_alpha > 1e-6:
        def neftune_forward_hook(module: torch.nn.Module, args: Tuple[torch.Tensor], output: torch.Tensor):
            if module.training:
                dims = torch.tensor(output.size(1) * output.size(2))
                mag_norm = finetuning_args.neft_alpha / torch.sqrt(dims)
                output = output + torch.zeros_like(output).uniform_(-mag_norm, mag_norm)
            return output

        model.get_input_embeddings().register_forward_hook(neftune_forward_hook)
        logger.info("Using noisy embedding with alpha={:.2f}".format(finetuning_args.neft_alpha))


    if use_gradient_checkpointing and getattr(model, "supports_gradient_checkpointing", False):
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module: torch.nn.Module, args: Tuple[torch.Tensor], output: torch.Tensor):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        model.gradient_checkpointing_enable()
        model.config.use_cache = False # turn off when gradient checkpointing is enabled
        logger.info("Gradient checkpointing enabled.")


    if finetuning_args.finetuning_type != "full" and hasattr(model, output_layer_name):
        output_layer = getattr(model, output_layer_name)
        if isinstance(output_layer, torch.nn.Linear):
            def fp32_forward_pre_hook(module: torch.nn.Module, args: Tuple[torch.Tensor]):
                return args[0].to(output_layer.weight.dtype)
            def fp32_forward_post_hook(module: torch.nn.Module, args: Tuple[torch.Tensor], output: torch.Tensor):
                return output.to(torch.float32)
            output_layer.register_forward_pre_hook(fp32_forward_pre_hook)
            output_layer.register_forward_hook(fp32_forward_post_hook)

    return model



def init_adapter(
    model: "PreTrainedModel",
    model_args: "ModelArguments",
    finetuning_args: "FinetuningArguments",
    is_trainable: bool
) -> "PreTrainedModel":
    r"""
    Initializes the adapters.

    Support full-parameter, freeze and LoRA training.

    Note that the trainable parameters must be cast to float32.
    """
    if (not is_trainable) and model_args.checkpoint_dir is None:
        logger.info("Checkpoint is not found at evaluation, load the original model.")
        return model

    if finetuning_args.finetuning_type == "full" and is_trainable:
        logger.info("Fine-tuning method: Full")
        model = model.float()

    if finetuning_args.finetuning_type == "freeze" and is_trainable:
        logger.info("Fine-tuning method: Freeze")
        num_layers = (
            getattr(model.config, "num_hidden_layers", None)
            or getattr(model.config, "num_layers", None)
            or getattr(model.config, "n_layer", None)
        )
        if not num_layers:
            raise ValueError("Current model does not support freeze tuning.")
        if finetuning_args.num_layer_trainable > 0: # fine-tuning the last n layers if num_layer_trainable > 0
            trainable_layer_ids = [num_layers - k - 1 for k in range(finetuning_args.num_layer_trainable)]
        else: # fine-tuning the first n layers if num_layer_trainable < 0
            trainable_layer_ids = [k for k in range(-finetuning_args.num_layer_trainable)]

        trainable_layers = []
        for module_name in finetuning_args.name_module_trainable:
            for idx in trainable_layer_ids:
                trainable_layers.append("{:d}.{}".format(idx, module_name))

        for name, param in model.named_parameters():
            if not any(trainable_layer in name for trainable_layer in trainable_layers):
                param.requires_grad_(False)
            else:
                param.data = param.data.to(torch.float32)

    if finetuning_args.finetuning_type == "lora":
        logger.info("Fine-tuning method: LoRA")
        checkpoint_to_resume = None

        if model_args.checkpoint_dir is not None:   
            if is_trainable and finetuning_args.resume_lora_training: 
                checkpoints_to_merge, checkpoint_to_resume = model_args.checkpoint_dir[:-1], model_args.checkpoint_dir[-1]
            else:
                checkpoints_to_merge = model_args.checkpoint_dir

            for checkpoint in checkpoints_to_merge:
                model = PeftModel.from_pretrained(model, checkpoint)
                if model_args.bits >= 16:
                    model = model.merge_and_unload()

            if len(checkpoints_to_merge) > 0:
                logger.info("Merged model checkpoint(s): {}.".format(checkpoints_to_merge))

            if checkpoint_to_resume is not None: # resume lora training
                logger.info("Resume model checkpoint(s): {} .".format(checkpoint_to_resume))
                model = PeftModel.from_pretrained(model, checkpoint_to_resume, is_trainable=is_trainable)

        if is_trainable and checkpoint_to_resume is None:    # create new lora weights while training
            if len(finetuning_args.lora_target_modules) == 1 and finetuning_args.lora_target_modules[0] == "all":
                target_modules = find_all_linear_modules(model, model_args.bits)
            else:
                target_modules = finetuning_args.lora_target_modules

            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=finetuning_args.lora_r,
                lora_alpha=finetuning_args.lora_alpha,
                lora_dropout=finetuning_args.lora_dropout,
                target_modules=target_modules,
                modules_to_save=finetuning_args.additional_target
            )
            model = get_peft_model(model, lora_config)

    if model_args.checkpoint_dir is not None:
        logger.info("Loaded fine-tuned model from checkpoint(s): {}".format(",".join(model_args.checkpoint_dir)))

    return model