import torch
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer
# ==================== 1. 导入 DynamicCache ====================
from transformers.cache_utils import DynamicCache

def setup_environment():
    """
    步骤1: 负责设置项目环境，包括加载模型和分词器。
    """
    model_name = "Qwen/Qwen3-4B-Instruct-2507"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16,
        device_map="auto"
    ).eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"模型 '{model_name}' 加载成功并运行在 {device} 上。")
    return tokenizer, model, device


def prepare_debate_prompt():
    """
    步骤3: 准备M-MLU问题和辩论各阶段所需的提示。
    """
    # ... (这部分函数保持我们之前优化好的版本不变) ...
    mmlu_question = "Which of the following is a direct product of the light-dependent reactions of photosynthesis?"
    options = { "A": "Glucose", "B": "ATP and NADPH", "C": "Oxygen gas (O₂)", "D": "Carbon dioxide (CO₂)" }
    formatted_options = "\n".join([f"{key}:{value}" for key, value in options.items()])
    initial_prompt = f"Question: {mmlu_question}\n\nOptions:\n{formatted_options}"
    prompts = {
        "agent_initial": ("..."), # 请使用你之前优化好的简洁Prompt
        "agent_initial_b": ("..."),
        "agent_reflect": ("...")
    }
    return initial_prompt, prompts


def slice_kv_cache(past_key_values, start_index, end_index):
    # 这个函数处理元组，保持不变
    new_kv_cache = []
    for layer_kv in past_key_values:
        key_tensor, value_tensor = layer_kv
        sliced_key = key_tensor[:, :, start_index:end_index, :]
        sliced_value = value_tensor[:, :, start_index:end_index, :]
        new_kv_cache.append((sliced_key, sliced_value))
    return tuple(new_kv_cache)

def concatenate_kv_caches(cache1, cache2):
    # 这个函数处理元组，保持不变
    concatenated_cache = []
    if not cache1: return cache2
    if not cache2: return cache1
    assert len(cache1) == len(cache2)
    for layer_cache1, layer_cache2 in zip(cache1, cache2):
        key1, value1 = layer_cache1
        key2, value2 = layer_cache2
        concatenated_key = torch.cat([key1, key2], dim=2)
        concatenated_value = torch.cat([value1, value2], dim=2)
        concatenated_cache.append((concatenated_key, concatenated_value))
    return tuple(concatenated_cache)

# ==================== 2. 修改 extend_kv_cache 以处理格式转换 ====================
def extend_kv_cache(model, tokenizer, prompt_text, past_key_values):
    """
    只计算新prompt_text的KV Cache，并将其追加到现有的past_key_values上。
    增加了对不同Cache格式的处理。
    """
    new_input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(model.device)

    # 检查传入的 past_key_values 是否为元组，如果是，则转换为 DynamicCache 对象
    if isinstance(past_key_values, tuple):
        cache_to_pass = DynamicCache.from_legacy_cache(past_key_values)
    else:
        cache_to_pass = past_key_values # 如果已经是对象格式，则直接使用

    with torch.no_grad():
        outputs = model(
            input_ids=new_input_ids,
            past_key_values=cache_to_pass, # 传入转换后的 Cache 对象
            use_cache=True,
        )
    
    # 检查返回的 Cache 是否为新格式，如果是，则转回元组以保持后续代码兼容性
    if hasattr(outputs.past_key_values, "to_legacy_cache"):
        return outputs.past_key_values.to_legacy_cache(), new_input_ids
    else:
        return outputs.past_key_values, new_input_ids

if __name__ == "__main__":
    # --- 准备阶段 ---
    tokenizer, model, device = setup_environment()
    question_prompt, role_prompts = prepare_debate_prompt()
    print("\n✅ 准备工作完成。")

    # --- 第1轮辩论 ---
    print("\n--- 辩论开始：第1轮 ---")
    round1_prompt = f"{question_prompt}\n\n{role_prompts['agent_initial']}"
    round1_inputs = tokenizer(round1_prompt, return_tensors="pt").to(device)

    round1_prompt_b = f"{question_prompt}\n\n{role_prompts['agent_initial_b']}"
    round1_inputs_b = tokenizer(round1_prompt_b, return_tensors="pt").to(device)
    with torch.no_grad():
        # 第一次generate，我们让它返回它默认的cache格式
        outputs_a_r1 = model.generate(**round1_inputs, max_new_tokens=200, use_cache=True, return_dict_in_generate=True, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.eos_token_id)
        outputs_b_r1 = model.generate(**round1_inputs_b, max_new_tokens=200, use_cache=True, return_dict_in_generate=True, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.eos_token_id)

    # 为了让我们自己的函数能处理，如果返回的是对象，就转成元组
    agent_a_kv_cache_r1 = outputs_a_r1.past_key_values
    if hasattr(agent_a_kv_cache_r1, "to_legacy_cache"):
        agent_a_kv_cache_r1 = agent_a_kv_cache_r1.to_legacy_cache()
    new_token_ids_a = outputs_a_r1.sequences[0, round1_inputs.input_ids.shape[1]:]

    agent_b_kv_cache_r1 = outputs_b_r1.past_key_values
    if hasattr(agent_b_kv_cache_r1, "to_legacy_cache"):
        agent_b_kv_cache_r1 = agent_b_kv_cache_r1.to_legacy_cache()
    new_token_ids_b = outputs_b_r1.sequences[0, round1_inputs_b.input_ids.shape[1]:]
    
    del outputs_a_r1, outputs_b_r1
    
    print("Agent A (R1):", tokenizer.decode(new_token_ids_a, skip_special_tokens=True))
    print("\nAgent B (R1):", tokenizer.decode(new_token_ids_b, skip_special_tokens=True))
    print("\n✅ 第1轮完成。")
    
    # --- 准备第2轮 ---
    start_idx_b = round1_inputs_b.input_ids.shape[1]
    end_idx_b = agent_b_kv_cache_r1[0][0].shape[2]
    agent_b_kv_cache_r1_sliced = slice_kv_cache(agent_b_kv_cache_r1, start_idx_b, end_idx_b)

    cache_with_both_arguments = concatenate_kv_caches(agent_a_kv_cache_r1, agent_b_kv_cache_r1_sliced)
    del agent_a_kv_cache_r1, agent_b_kv_cache_r1_sliced, agent_b_kv_cache_r1
    
    reflection_prompt_text_a = f"\n\n{role_prompts['agent_reflect']}"
    # 调用我们修改后的 extend_kv_cache，它现在能处理格式转换了
    agent_a_final_kv_cache_r2_input, _ = extend_kv_cache(model, tokenizer, reflection_prompt_text_a, cache_with_both_arguments)
    del cache_with_both_arguments
    
    gc.collect()
    torch.cuda.empty_cache()

    print("\n✅ Agent A 第二轮的“思想融合”输入已准备就绪。")

    # --- 第2轮辩论 ---
    print("\n--- 辩论进行中：第2轮 (Agent A发言) ---")
    final_input_cache_len = agent_a_final_kv_cache_r2_input[0][0].shape[2]
    tokenizer.pad_token_id = tokenizer.eos_token_id
    start_token_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
    input_ids_r2 = torch.tensor([[start_token_id]], dtype=torch.long, device=device)
    attention_mask_r2 = torch.ones(1, final_input_cache_len + 1, dtype=torch.long, device=device)
    cache_position = torch.tensor([[final_input_cache_len]], device=device)

    with torch.no_grad():
        # ==================== 3. 修改第二次 generate 调用以处理格式转换 ====================
        # 在传入 generate 前，也将我们的元组 Cache 转换为 DynamicCache 对象
        final_cache_to_pass = DynamicCache.from_legacy_cache(agent_a_final_kv_cache_r2_input)
        
        outputs_a_r2 = model.generate(
            input_ids=input_ids_r2,
            past_key_values=final_cache_to_pass, # 传入转换后的对象
            attention_mask=attention_mask_r2,
            cache_position=cache_position,
            max_new_tokens=200,
            use_cache=True,
            return_dict_in_generate=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
        
    new_token_ids_a_r2 = outputs_a_r2.sequences[0, input_ids_r2.shape[1]:]
    agent_a_response_text_r2 = tokenizer.decode(new_token_ids_a_r2, skip_special_tokens=True)

    print("\n--- Agent A 的第二轮回应 ---")
    print(f"\nAgent A (R2 - 基于对B思想的融合):\n{agent_a_response_text_r2}")
    print("\n✅ 辩论结束。")