import torch
import re

# 我们需要从 src.models 模块中导入我们之前创建的工具函数
from src.models.kv_cache_utils import extend_kv_cache, concatenate_kv_caches, slice_kv_cache

class DebateAgent:
    """
    代表一个参与辩论的智能体。
    它管理自己的对话历史、KV Cache 状态，并能生成回应。
    """
    def __init__(self, model, tokenizer, system_prompt, name="Agent"):
        """
        初始化一个智能体。
        
        Args:
            model: 预加载的 transformers 模型。
            tokenizer: 预加载的 tokenizer。
            system_prompt: 该智能体的系统指令。
            name: 智能体的名字，用于打印日志。
        """
        self.model = model
        self.tokenizer = tokenizer
        self.system_prompt = system_prompt
        self.name = name
        self.device = model.device

        # --- 状态变量 ---
        # 对话历史，以 ChatML 格式存储
        self.messages = [{"role": "system", "content": self.system_prompt}]
        # 完整的对话历史对应的 KV Cache
        self.kv_cache = None
        # 完整的对话历史对应的 token IDs
        self.full_token_ids = None

    def generate_response(self, new_user_prompt_content=None, opponent_cache_slice=None):
        """
        生成一次回应。可以是第一轮独立思考，也可以是基于对手观点的反思。
        
        Args:
            new_user_prompt_content (str, optional): 新的用户输入内容, 比如问题或反思指令。
            opponent_cache_slice (tuple, optional): 对手上一轮回答的 KV Cache 切片。
        """
        # 如果有新的用户输入，先更新消息历史
        if new_user_prompt_content:
            self.messages.append({"role": "user", "content": new_user_prompt_content})

        # --- 准备 generate 函数的输入 ---
        
        # 判断是否是第一轮，如果是，则引入随机性
        is_first_round = (self.kv_cache is None)

        if is_first_round:
            input_ids = self.tokenizer.apply_chat_template(
                self.messages, add_generation_prompt=True, return_tensors="pt"
            ).to(self.device)
            past_key_values = None
            attention_mask = None
            cache_position = None
        else: # 后续轮次
            user_turn_text = self.tokenizer.apply_chat_template(
                [self.messages[-1]], add_generation_prompt=False, tokenize=False
            )
            self_cache_extended, _ = extend_kv_cache(
                self.model, self.tokenizer, user_turn_text, self.kv_cache
            )
            combined_cache = concatenate_kv_caches(self_cache_extended, opponent_cache_slice)
            past_key_values = combined_cache
            start_assistant_text = "\n<|im_start|>assistant\n"
            input_ids = self.tokenizer(start_assistant_text, return_tensors="pt").input_ids.to(self.device)
            cache_len = past_key_values[0][0].shape[2]
            attention_mask = torch.ones(1, cache_len + input_ids.shape[1], dtype=torch.long, device=self.device)
            cache_position = torch.tensor([[cache_len]], device=self.device)

        # --- 执行生成 ---
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                cache_position=cache_position,
                max_new_tokens=800, # 增加 token 限制以容纳 CoT
                use_cache=True,
                return_dict_in_generate=True,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
                
                # !! 核心改动：在第一轮引入随机性 !!
                do_sample=is_first_round,
                temperature=0.7 if is_first_round else 1.0, # temperature=1.0 接近于无随机性
                top_p=0.9 if is_first_round else 1.0,
            )

        # --- 更新自身状态 (这部分逻辑也需要微调) ---
        # 旧的 self.full_token_ids 拼接逻辑有点问题，这里修正一下
        if is_first_round:
            self.full_token_ids = outputs.sequences
            # 第一轮的 new_token_ids 是从原始输入的末尾开始的
            new_token_ids = self.full_token_ids[0, input_ids.shape[1]:]
        else:
            new_generated_ids = outputs.sequences[0, input_ids.shape[1]:]
            self.full_token_ids = torch.cat([self.full_token_ids, new_generated_ids.unsqueeze(0)], dim=1)
            new_token_ids = new_generated_ids

        response_text = self.tokenizer.decode(new_token_ids, skip_special_tokens=True).strip()
        self.messages.append({"role": "assistant", "content": response_text})
        
        # 打印日志
        print(f"\n--- {self.name} 生成回应 (Round {len(self.messages)//2}) ---")
        print(response_text)
        print("--------------------")

    def get_latest_response(self):
        """
        获取最新一次的回答文本。
        """
        if self.messages and self.messages[-1]["role"] == "assistant":
            return self.messages[-1]["content"]
        return "尚未生成任何回应。"

    def parse_answer_option(self):
        """
        从 "Final Answer: [X]" 格式的回答中解析出选择的选项（A, B, C, D）。
        增加了备用逻辑以提高鲁棒性。
        """
        latest_response = self.get_latest_response()
        
        # 主要解析逻辑：寻找 "Final Answer: X"
        match = re.search(r"Final Answer:\s*([A-D])", latest_response, re.IGNORECASE)
        if match:
            return match.group(1).upper()
        
        # 备用逻辑1：如果模型在最后只输出了一个字母
        # 检查最后20个字符，避免匹配到文本中间的 "A"
        last_part = latest_response.strip()[-20:]
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

    def get_answer_cache_slice(self):
        """
        获取自己最新一次回答（即最后一个 assistant turn）的 KV Cache。
        """
        if self.kv_cache is None or len(self.messages) < 3: # 至少要有 sys, user, assistant
            return None

        # 1. 先生成除了最后一个 assistant turn 之外的所有历史的文本
        history_text_before_last_answer = self.tokenizer.apply_chat_template(
            self.messages[:-1], add_generation_prompt=False, tokenize=False
        )
        # 再加上 assistant 的起始提示符
        start_assistant_text = "\n<|im_start|>assistant\n"
        full_history_text = history_text_before_last_answer + start_assistant_text

        # 2. 计算这个历史的 token 长度
        #    注意：这里我们不能用 apply_chat_template 因为它会处理模板逻辑，
        #    而 full_token_ids 是一个扁平的 token 序列。
        #    我们需要找到 assistant 回答的真正起始点。
        #
        #    一个更鲁棒的方法是：
        #    总长度 = len(full_token_ids)
        #    回答长度 = len(response_tokens)
        #    起始点 = 总长度 - 回答长度
        
        # 计算除了最后一个 assistant turn 外的 message 列表的 token 长度
        # 这就是上一次 generate 时的 input_ids 的长度
        
        # 我们需要一个更简单的方式来找到上一个回答的起始点。
        # 一个简单有效的方法是计算历史消息的 token 长度。
        
        # 重新生成历史 input_ids (不含最后一个 assistant 回答)
        history_ids_before_last_answer = self.tokenizer.apply_chat_template(
            self.messages[:-1], add_generation_prompt=True, return_tensors="pt"
        ).to(self.device)

        start_index = history_ids_before_last_answer.shape[1]
        
        # 结束位置是当前 kv_cache 的总长度
        end_index = self.kv_cache[0][0].shape[2]
        
        if start_index >= end_index:
            print(f"[Warning] {self.name}: 计算 cache slice 索引时出错, start >= end。")
            return None

        return slice_kv_cache(self.kv_cache, start_index, end_index)