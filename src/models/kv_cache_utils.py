import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def slice_kv_cache(past_key_values, start_index, end_index):
    """
    裁剪KV Cache以仅保留指定范围的序列。
    """
    new_kv_cache = []
    for layer_kv in past_key_values:
        key_tensor, value_tensor = layer_kv
        # 索引说明: [batch_size, num_heads, sequence_length, head_dim]
        # 我们在 sequence_length 维度 (dim=2) 上进行切片
        sliced_key = key_tensor[:, :, start_index:end_index, :]
        sliced_value = value_tensor[:, :, start_index:end_index, :]
        new_kv_cache.append((sliced_key, sliced_value))
    return tuple(new_kv_cache)

def concatenate_kv_caches(cache1, cache2):
    """
    将两个KV Cache沿着序列长度维度拼接起来。
    """
    if not cache1:
        return cache2
    if not cache2:
        return cache1
        
    concatenated_cache = []
    assert len(cache1) == len(cache2), "KV Caches的层数必须相同才能拼接"
    
    for layer_cache1, layer_cache2 in zip(cache1, cache2):
        key1, value1 = layer_cache1
        key2, value2 = layer_cache2
        
        # 在 sequence_length 维度 (dim=2) 上拼接
        concatenated_key = torch.cat([key1, key2], dim=2)
        concatenated_value = torch.cat([value1, value2], dim=2)
        concatenated_cache.append((concatenated_key, concatenated_value))
        
    return tuple(concatenated_cache)

def extend_kv_cache(model, tokenizer, prompt_text, past_key_values):
    """
    只计算新prompt_text的KV Cache，并将其追加到现有的past_key_values上。
    """
    new_input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(model.device)
    
    # 如果没有历史 cache，这就是一个普通的前向传播
    if past_key_values is None:
        with torch.no_grad():
            outputs = model(input_ids=new_input_ids, use_cache=True)
        return outputs.past_key_values, new_input_ids

    # 有历史 cache，只计算新 token 的
    cache_len = past_key_values[0][0].shape[2]
    attention_mask = torch.ones(1, cache_len + new_input_ids.shape[1], dtype=torch.long, device=model.device)
    cache_position = torch.tensor([[cache_len]], device=model.device)

    with torch.no_grad():
        outputs = model(
            input_ids=new_input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            cache_position=cache_position,
            use_cache=True
        )
    
    # 这里的 outputs.past_key_values 已经是拼接好的了
    return outputs.past_key_values, new_input_ids