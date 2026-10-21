import torch

from exp1.common import seed_all
from exp1.modeling import add_lora, select_torch_backend


def tiny_qwen(device="cpu", checkpointing=False, adapters=True):
    """Random tiny genuine Qwen3.5 modules, explicitly a correctness fixture."""
    from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration
    seed_all(17)
    config = Qwen3_5Config(
        text_config={"vocab_size": 128, "hidden_size": 32, "intermediate_size": 64,
                     "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2,
                     "head_dim": 8, "linear_num_key_heads": 2, "linear_num_value_heads": 2,
                     "linear_key_head_dim": 8, "linear_value_head_dim": 8,
                     "linear_conv_kernel_dim": 4, "layer_types": ["linear_attention", "full_attention"],
                     "rope_parameters": {"rope_type": "default", "rope_theta": 10000,
                                         "partial_rotary_factor": 0.5, "mrope_section": [1, 1, 0]},
                     "attention_dropout": 0.0, "eos_token_id": 2, "pad_token_id": 2, "use_cache": False},
        vision_config={"depth": 1, "hidden_size": 16, "intermediate_size": 32, "num_heads": 2,
                       "out_hidden_size": 32, "patch_size": 2, "num_position_embeddings": 16,
                       "deepstack_visual_indexes": []},
        image_token_id=120, video_token_id=121, vision_start_token_id=122, vision_end_token_id=123)
    model = select_torch_backend(Qwen3_5ForConditionalGeneration(config))
    model.requires_grad_(False)
    model = model.to(device=device, dtype=torch.bfloat16 if device == "cuda" else torch.float32)
    if adapters:
        model = add_lora(model)
    if checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return model


def examples():
    return [{"id": f"fixture/{i}", "input_ids": [3 + ((j * 7 + i) % 100) for j in range(8 + i % 3)],
             "labels": [-100] * 4 + [3 + ((j * 7 + i) % 100) for j in range(4, 8 + i % 3)]}
            for i in range(8)]
