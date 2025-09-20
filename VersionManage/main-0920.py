# 9.20，使用apply_template模版
# 解决了
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
    准备MMLU问题和辩论各阶段所需的结构化提示。
    返回问题、选项和各角色的系统指令。
    """

    mmlu_question = "Alice and Bob share an apartment. Late at night Alice orders food. When the courier arrives, Alice is in the shower, so she texts Bob: “Can you go down to the gate and pick it up? I’ll pay.” Bob, busy gaming, tosses the keys to their roommate Carol and says, “Grab it for me.” While walking downstairs Carol trips over the threshold while looking at her phone, the food spills, and she fractures her wrist. Carol now sues for medical costs and the price of the ruined meal."
    options = {
        "A": "Alice – she created a gratuitous delegation by asking Bob.",
        "B": "Bob – he accepted the task but improperly re-delegated it.", 
        "C": "The delivery platform – it failed to complete safe delivery to the door.",
        "D": "Carol herself – she assumed the risk by not watching her step"
    }

    # 将问题和选项格式化为用户提问内容
    formatted_options = "\n".join([f"{key}: {value}" for key, value in options.items()])
    user_question_content = f"Question: {mmlu_question}\n\nOptions:\n{formatted_options}"
    
    # 定义结构化的系统指令和反思指令
    system_prompts = {
        "agent_a": "You are lawyer A. Your task is to select the best answer and provide a concise reason. "
                   "Structure your response *exactly* as follows:\n"
                   "I believe the correct answer is [option].\n"
                   "Reason: [Your 1-2 sentence reasoning here]\n"
                   "You MUST NOT add any other content.",
                         
        "agent_b": "You are assigned to argue for option C. Your sole mission is to build the strongest case for C. "
                   "Structure your response *exactly* as follows:\n"
                   "I believe the correct answer is C.\n"
                   "Reason: [Your 1-2 sentence reasoning here, arguing for C]\n"
                   "You MUST NOT add any other content.",
    }
    
    reflection_instruction = (
        "The next part is your opponent's argument.\n\n"
        "First summarize your opponent's key points in one sentence.\n"
        "Then, state whether you will stick to your initial answer or change your mind, and explain why in 1-2 sentences.\n"
        "Structure your response as:\n"
        "I believe the correct answer is [option].\n"
        "Summary: [...]\n"
        "Stance: [Stick/Change]\n"
        "Reason: [...]\n"
    )

    return user_question_content, system_prompts, reflection_instruction

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
    # 1. 准备tokenizer,model,device
    tokenizer, model, device = setup_environment()
    question,system_prompts,reflection_instruction = prepare_debate_prompt()
    print("✅ 准备工作完成。\n\n")

    # 2. 并行生成观点并捕获KV Cache
    print("辩论开始：第1轮 ---")
    messages_a_r1 = [
        {"role":"system","content":system_prompts['agent_a']},
        {"role":"user","content":question}
    ]

    # 使用apply_chat_template将消息列表转换为token_ids
    # add_generation_prompt = True会在末位加上<|im_start|>assistant\n，提示模型开始生成
    round1_inputs_a = tokenizer.apply_chat_template(
        messages_a_r1,
        add_generation_prompt = True,
        return_tensors = 'pt'
    ).to(device)

    # 对Agent B进行同样的操作
    messages_b_r1 = [
        {"role":"system","content":system_prompts['agent_b']},
        {"role":"user","content":question}
    ]
    round1_inputs_b = tokenizer.apply_chat_template(
        messages_b_r1,
        add_generation_prompt = True,
        return_tensors = 'pt'
    ).to(device)

    with torch.no_grad():
        outputs_a_r1 = model.generate(
            input_ids = round1_inputs_a, 
            max_new_tokens=150, 
            use_cache=True, 
            return_dict_in_generate=True,
            eos_token_id = tokenizer.eos_token_id,
            pad_token_id = tokenizer.eos_token_id,
            repetition_penalty=1.15
            )
        outputs_b_r1 = model.generate(
            input_ids = round1_inputs_b, 
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
    new_token_ids_a = full_token_ids_a[0, round1_inputs_a.shape[1]:]
    response_text_a_r1 = tokenizer.decode(new_token_ids_a,skip_special_tokens=True)

    full_token_ids_b = outputs_b_r1.sequences
    agent_b_kv_cache_r1 = outputs_b_r1.past_key_values
    new_token_ids_b = full_token_ids_b[0, round1_inputs_b.shape[1]:]
    response_text_b_r1 = tokenizer.decode(new_token_ids_b,skip_special_tokens=True)

    # 打印第1轮结果
    print("Agent A (R1):", response_text_a_r1)
    # 直接解码整个 new_token_ids_b 张量
    print("\nAgent B (R1):", response_text_b_r1)
    print("\n✅ 第1轮完成。")

    # 3. 准备第2轮：截取和拼接KV Cache
    # 截取 B 的回答部分的KV Cache
    start_idx_b = round1_inputs_b.shape[1]
    end_idx_b = agent_b_kv_cache_r1[0][0].shape[2]
    agent_b_kv_cache_r1_sliced = slice_kv_cache(agent_b_kv_cache_r1, start_idx_b, end_idx_b)
    
    # 构建 Agent A 第二轮的输入--现在的设计需要符合chat_template的格式
    reflection_prompt_text_a = reflection_instruction
    messages_a_r2_extension = [
        {"role":"user","content":reflection_prompt_text_a}
    ]
    reflection_prompt_text = tokenizer.apply_chat_template(
        messages_a_r2_extension,
        add_generation_prompt = False,
        tokenize = False
    )
    # 注意：这里 extend_kv_cache 应该基于 A 的原始 cache 进行扩展
    reflection_kv_cache_a, _ = extend_kv_cache(model, tokenizer, reflection_prompt_text, agent_a_kv_cache_r1)
    agent_a_final_kv_cache_r2_input = concatenate_kv_caches(reflection_kv_cache_a, agent_b_kv_cache_r1_sliced)


    print("\n✅ Agent A 第二轮的“思想融合”输入已准备就绪。")

    # 第2轮辩论: Agent A 发言
    print("\n--- 辩论进行中：第2轮 (Agent A发言) ---")

    # generation_prompt_ids = tokenizer.apply_chat_template(
    #     [{"role":"assistant","content":""}],
    #     add_generation_prompt = True,
    #     return_tensors='pt'
    # )

    # decoded_prompt = tokenizer.decode(generation_prompt_ids[0])
    # print(f"启动生成的Prompt Tokens解码后为: '{decoded_prompt}'")

    start_assistant_text = "\n<|im_start|>assistant\n"
    input_ids_r2 = tokenizer(start_assistant_text,return_tensors='pt').input_ids.to(device)
    
    # 准备启动token和正确的attention_mask
    final_input_cache_len = agent_a_final_kv_cache_r2_input[0][0].shape[2]

    attention_mask_r2 = torch.ones(1, final_input_cache_len + input_ids_r2.shape[1], dtype=torch.long, device=device)

    cache_position = torch.tensor([[final_input_cache_len]], device=device)

    with torch.no_grad():
        outputs_a_r2 = model.generate(
            input_ids=input_ids_r2,
            past_key_values=agent_a_final_kv_cache_r2_input,
            attention_mask=attention_mask_r2,
            cache_position=cache_position, # 在generate调用中传入cache_position
            max_new_tokens=500,
            use_cache=True,
            return_dict_in_generate=True,
            eos_token_id = tokenizer.eos_token_id,
            pad_token_id = tokenizer.eos_token_id,
            
        )

    # 解码和打印新生成的部分
    new_token_ids_a_r2 = outputs_a_r2.sequences[0, input_ids_r2.shape[1]:]
    agent_a_response_text_r2 = tokenizer.decode(new_token_ids_a_r2, skip_special_tokens=True)

    print("\n--- Agent A 的第二轮回应 ---")
    print(f"\nAgent A (R2 - 基于对B思想的融合):\n {agent_a_response_text_r2}")