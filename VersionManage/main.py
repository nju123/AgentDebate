import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MAX_NEW_TOKENS = 512

def setup_environment():
    """
    负责设置项目环境，包括加载模型和分词器。
    """
    model_name = "Qwen/Qwen3-4B-Instruct-2507"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16,
        device_map="auto"
    ).eval() #设置为评估模式

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"模型 '{model_name}' 加载成功并运行在 {device} 上。")
    return tokenizer, model, device


def prepare_debate_prompt():
    """
    准备M-MLU问题和辩论各阶段所需的提示。
    """
    # 生物学问题：光合作用
    mmlu_question = "Which of the following is a direct product of the light-dependent reactions of photosynthesis?"
    options = {
        "A": "Glucose",
        "B": "ATP and NADPH",
        "C": "Oxygen gas (O₂)",
        "D": "Carbon dioxide (CO₂)"
    }

    # 将问题和选项格式化为字符串
    formatted_options = "\n".join([f"{key}:{value}" for key,value in options.items()])
    initial_prompt = f"Question: {mmlu_question}\n\nOptions:\n{formatted_options}"
    
    prompts = {
        "agent_initial": "You are biologist A. Your task is to select the best answer and provide a concise reason. "
                         "Structure your response *exactly* as follows:\n"
                         "I believe the correct answer is [option].\n"
                         "Reason: [Your 1-2 sentence reasoning here]\n"
                         "DO NOT ADD OTHER CONTENT! YOUR ANSWER SHOULD STOP IN 200 WORDS!",
                         
        "agent_initial_b": "You are assigned to argue for option C. Your sole mission is to build the strongest case for C. "
                           "Structure your response *exactly* as follows:\n"
                           "I believe the correct answer is C.\n"
                           "Reason: [Your 1-2 sentence reasoning here, arguing for C]\n"
                           "DO NOT ADD OTHER CONTENT! YOUR ANSWER SHOULD STOP IN 200 WORDS!",
                          
        "agent_reflect": '''The next part is your opponent's argument.\n 
                        YOU NEED TO DO: First summarize its key points in one sentence.\n
                        Then, state whether you will stick to your original answer or change your mind, and explain why in 1-2 sentences.\n
                        Structure your response as:\n
                            Summary: [...]
                            Stance: [Stick/Change]  
                            Reason: [...]
                        DO NOT ADD OTHER CONTENT!    
                        '''
    }
    # =======================================================
    return initial_prompt, prompts

def slice_kv_cache(past_key_values, start_index, end_index):
    """
    裁剪KV Cache以仅保留指定范围的序列。
    """
    new_kv_cache = []
    for layer_kv in past_key_values:
        key_tensor, value_tensor = layer_kv
        sliced_key = key_tensor[:, :, start_index:end_index, :]
        sliced_value = value_tensor[:, :, start_index:end_index, :]
        new_kv_cache.append((sliced_key, sliced_value))
    return tuple(new_kv_cache)

def concatenate_kv_caches(cache1, cache2):
    """
    将两个KV Cache沿着序列长度维度拼接起来。
    """
    concatenated_cache = []
    assert len(cache1) == len(cache2), "KV Caches的层数必须相同才能拼接"
    for layer_cache1, layer_cache2 in zip(cache1, cache2):
        key1, value1 = layer_cache1
        key2, value2 = layer_cache2
        concatenated_key = torch.cat([key1, key2], dim=2)
        concatenated_value = torch.cat([value1, value2], dim=2)
        concatenated_cache.append((concatenated_key, concatenated_value))
    return tuple(concatenated_cache)

def extend_kv_cache(model, tokenizer, prompt_text, past_key_values):
    """
    只计算新prompt_text的KV Cache，并将其追加到现有的past_key_values上。
    """
    new_input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(model.device)
    with torch.no_grad():
        outputs = model(
            input_ids=new_input_ids,
            past_key_values=past_key_values,
            use_cache=True
        )
    return outputs.past_key_values, new_input_ids




if __name__ == "__main__":
    # 准备tokenizer,model,device
    tokenizer, model, device = setup_environment()
    question_prompt, role_prompts = prepare_debate_prompt()
    print("\n✅ 准备工作完成。")

    # 并行生成观点并捕获KV Cache
    print("\n--- 辩论开始：第1轮 ---")
    round1_prompt = f"{question_prompt}\n\n{role_prompts['agent_initial']}"
    round1_inputs = tokenizer(round1_prompt, return_tensors="pt").to(device)

    round1_prompt_b = f"{question_prompt}\n\n{role_prompts['agent_initial_b']}"
    round1_inputs_b = tokenizer(round1_prompt_b, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs_a_r1 = model.generate(
            **round1_inputs, 
            max_new_tokens=150, 
            use_cache=True, 
            return_dict_in_generate=True,
            eos_token_id = tokenizer.eos_token_id,
            pad_token_id = tokenizer.eos_token_id,
            repetition_penalty=1.15
            )
        outputs_b_r1 = model.generate(
            **round1_inputs_b, 
            max_new_tokens=150, 
            use_cache=True, 
            return_dict_in_generate=True,
            eos_token_id = tokenizer.eos_token_id,
            pad_token_id = tokenizer.eos_token_id,
            repetition_penalty=1.15
            )

    # 提取第1轮结果 (full tokens, new tokens, kv_cache)
    full_token_ids_a = outputs_a_r1.sequences
    agent_a_kv_cache_r1 = outputs_a_r1.past_key_values
    new_token_ids_a = full_token_ids_a[0, round1_inputs.input_ids.shape[1]:]

    full_token_ids_b = outputs_b_r1.sequences
    agent_b_kv_cache_r1 = outputs_b_r1.past_key_values
    new_token_ids_b = full_token_ids_b[0, round1_inputs_b.input_ids.shape[1]:]

    # 打印第1轮结果
    print("Agent A (R1):", tokenizer.decode(new_token_ids_a, skip_special_tokens=True))
    # 直接解码整个 new_token_ids_b 张量
    print("\nAgent B (R1):", tokenizer.decode(new_token_ids_b, skip_special_tokens=True))
    print("\n✅ 第1轮完成。")

    # 准备第2轮：截取和拼接KV Cache
    # 截取 B 的回答部分的KV Cache
    start_idx_b = round1_inputs_b.input_ids.shape[1]
    end_idx_b = agent_b_kv_cache_r1[0][0].shape[2]
    agent_b_kv_cache_r1_sliced = slice_kv_cache(agent_b_kv_cache_r1, start_idx_b, end_idx_b)
    
    # 构建 Agent A 第二轮的输入
    reflection_prompt_text_a = f"\n\n{role_prompts['agent_reflect']}"
    # 注意：这里 extend_kv_cache 应该基于 A 的原始 cache 进行扩展
    reflection_kv_cache_a, _ = extend_kv_cache(model, tokenizer, reflection_prompt_text_a, agent_a_kv_cache_r1)
    agent_a_final_kv_cache_r2_input = concatenate_kv_caches(reflection_kv_cache_a, agent_b_kv_cache_r1_sliced)


    print("\n✅ Agent A 第二轮的“思想融合”输入已准备就绪。")

    # 第2轮辩论: Agent A 发言
    print("\n--- 辩论进行中：第2轮 (Agent A发言) ---")
    
    # 准备启动token和正确的attention_mask
    final_input_cache_len = agent_a_final_kv_cache_r2_input[0][0].shape[2]

    tokenizer.pad_token_id = tokenizer.eos_token_id
    start_token_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id

    input_ids_r2 = torch.tensor([[start_token_id]], dtype=torch.long, device=device)
    attention_mask_r2 = torch.ones(1, final_input_cache_len + 1, dtype=torch.long, device=device)


    cache_position = torch.tensor([[final_input_cache_len]], device=device)

    with torch.no_grad():
        outputs_a_r2 = model.generate(
            input_ids=input_ids_r2,
            past_key_values=agent_a_final_kv_cache_r2_input,
            attention_mask=attention_mask_r2,
            cache_position=cache_position, # 在generate调用中传入cache_position
            max_new_tokens=150,
            use_cache=True,
            return_dict_in_generate=True,
            eos_token_id = tokenizer.eos_token_id,
            pad_token_id = tokenizer.eos_token_id,
            
        )

    # 解码和打印新生成的部分
    new_token_ids_a_r2 = outputs_a_r2.sequences[0, input_ids_r2.shape[1]:]
    agent_a_response_text_r2 = tokenizer.decode(new_token_ids_a_r2, skip_special_tokens=True)

    print("\n--- Agent A 的第二轮回应 ---")
    print(f"\nAgent A (R2 - 基于对B思想的融合): {agent_a_response_text_r2}")