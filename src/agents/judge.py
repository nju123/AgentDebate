import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
# 导入我们需要的 KV Cache 工具函数
from src.models.kv_cache_utils import extend_kv_cache

class Judge:
    """
    代表一个裁决者智能体。
    它基于拼接好的、代表完整辩论历史的 KV Cache 做出最终决定。
    """
    def __init__(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, system_prompt: str):
        """
        初始化裁判。
        
        Args:
            model: 预加载的 transformers 模型。
            tokenizer: 预加载的 tokenizer。
            system_prompt: 裁判的系统指令。
        """
        self.model = model
        self.tokenizer = tokenizer
        self.system_prompt = system_prompt
        self.device = model.device

    def make_decision(self, question_content: str, final_debate_kv_cache: tuple):
        """
        根据拼接好的、代表完整辩论历史的 KV Cache 做出裁决。

        Args:
            question_content (str): M-MLU 问题的文本。
            final_debate_kv_cache (tuple): 经拼接后，代表完整辩论历史的 KV Cache。

        Returns:
            str: 最终裁定的选项字母 (e.g., "A") 或 "PARSE_ERROR"。
        """
        print("[INFO] Judge 开始基于融合的 KV-Cache 进行裁决...")

        # 1. 准备 Judge 的 system prompt 和 user prompt
        #    注意：我们不仅需要 user prompt，也需要 system prompt 来设定 Judge 的角色
        judge_user_prompt = f"""Based on the preceding debate about the question below, please choose the single best answer.
--- QUESTION ---
{question_content}
---
Fisrt conclude two agents key points in one or two sentence.
Then give me your analyse and your final choice concisely. 
Your output should follow this format:
"Agents key points:"
"Reasoning: [Your step-by-step analysis here, less than 3 step]\n"
"Judge Answer: [The single letter of your choice]"
"""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": judge_user_prompt}
        ]

        # 2. 将 Judge 的指令（system + user）转换为文本
        #    我们用 apply_chat_template 来确保格式正确，但这次我们不从头开始，
        #    而是要把它“嫁接”到已有的 KV Cache 上。
        #    因此，我们不能直接用 extend_kv_cache，因为它假设了连续的对话历史。
        #    一个更干净的方式是，将 Judge 视为一个全新的对话，
        #    但是它的 "past_key_values" 是辩论历史。
        #    所以，我们需要计算 Judge 指令的 KV Cache，然后拼接到辩论历史后面。
        
        #    为了简化，我们先用 extend_kv_cache 实现，这在语义上也是合理的：
        #    即在辩论历史这个 "context" 之后，引入了 Judge 的指令。

        # 2.1 准备 Judge 的指令文本，并计算其 KV Cache
        judge_instruction_text = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=False, tokenize=False
        )

        # 2.2 在完整的辩论历史 cache 上，扩展出 Judge 的指令 cache
        #     这里 extend_kv_cache 内部会处理好 position_ids
        judge_cache_with_instruction, _ = extend_kv_cache(
            self.model, self.tokenizer, judge_instruction_text, final_debate_kv_cache
        )

        # 3. 准备启动生成的 input_ids 和相关参数
        start_assistant_text = "\n<|im_start|>assistant\n"
        input_ids = self.tokenizer(start_assistant_text, return_tensors="pt").input_ids.to(self.device)
        
        past_key_values = judge_cache_with_instruction
        cache_len = past_key_values[0][0].shape[2]
        attention_mask = torch.ones(1, cache_len + input_ids.shape[1], dtype=torch.long, device=self.device)
        cache_position = torch.tensor([[cache_len]], device=self.device)

        # 4. 执行生成，得到裁决结果
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                cache_position=cache_position,
                max_new_tokens=800,
                use_cache=True,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        # 5. 解析输出
        response = self.tokenizer.decode(outputs[0, input_ids.shape[1]:], skip_special_tokens=True)
        
        print(f"[INFO] Judge 的原始输出: '{response}'")
        
        # # 提取第一个有效的字母
        # if response and response[0] in "ABCD":
        #     return response[0]
        # else:
        #     return "PARSE_ERROR"
        
        # 主要解析逻辑：寻找 "Judge Answer: X"
        match = re.search(r"Judge Answer:\s*([A-D])", response, re.IGNORECASE)
        if match:
            return match.group(1).upper()
        
        # 备用逻辑1：如果模型在最后只输出了一个字母
        # 检查最后20个字符，避免匹配到文本中间的 "A"
        last_part = response.strip()[-20:]
        # 查找被非字母数字字符包围的单个选项字母
        final_letters = re.findall(r'[^A-Z0-9]([A-D])[^A-Z0-9]?$', last_part.strip())
        if final_letters:
            return final_letters[-1].upper()

        # 备用逻辑2：如果上述都失败，直接在最后部分找最后一个出现的选项字母
        final_letters_in_last_part = re.findall(r'([A-D])', last_part)
        if final_letters_in_last_part:
            return final_letters_in_last_part[-1].upper()

        # 如果所有方法都失败了
        print(f"[Warning] {self.name}: 无法从以下回应中解析出最终答案:\n'{latest_response}'")
        return None