import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys, os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.agents.debate_agent import DebateAgent
from src.agents.judge import Judge # <-- 从新的位置导入
from src.models.kv_cache_utils import concatenate_kv_caches # <-- 我们需要这个来拼接最终cache

# ... setup_environment 和 prepare_mmlu_sample_and_prompts 函数保持不变 ...
def setup_environment():
    model_name = "Qwen/Qwen3-4B-Instruct-2507"
    print(f"正在加载模型: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto"
    ).eval()
    print(f"模型 '{model_name}' 加载成功。")
    return tokenizer, model

def prepare_mmlu_sample_and_prompts():
    mmlu_question = "Which of the following is a direct product of the light-dependent reactions of photosynthesis?"
    options = { "A": "Glucose", "B": "ATP and NADPH", "C": "Oxygen gas (O₂)", "D": "Carbon dioxide (CO₂)" }
    correct_answer = "B"
    formatted_options = "\n".join([f"{key}: {value}" for key, value in options.items()])
    user_question_content = f"Question: {mmlu_question}\n\nOptions:\n{formatted_options}"
    general_system_prompt = (
        "You are an expert biologist. Your task is to select the best answer from the given options and provide a concise reason. "
        "Structure your response *exactly* as follows:\n"
        "I believe the correct answer is [option].\n"
        "Reason: [Your 1-2 sentence reasoning here]\n"
        "You MUST NOT add any other content."
    )
    reflection_instruction = (
        "You have received your opponent's argument. "
        "First, summarize its key points in one sentence.\n"
        "Then, state whether you will stick to your initial answer or change your mind, and explain why in 1-2 sentences.\n"
        "Structure your response as:\n"
        "Summary: [...]\n"
        "Stance: [Stick/Change]\n"
        "Reason: [...]\n"
    )
    judge_system_prompt = (
        "You are an impartial judge. You will be shown a debate between two experts on a multiple-choice question. "
        "Your task is to analyze their arguments and determine the single best answer. "
        "Respond ONLY with the letter of the correct option (e.g., A)."
    )
    prompts = {
        "system": general_system_prompt,
        "reflection": reflection_instruction,
        "judge": judge_system_prompt
    }
    return user_question_content, options, correct_answer, prompts

def interleave_and_combine_caches(agent_a, agent_b, total_rounds):
    """
    交错拼接两个 agent 的 KV Cache，形成一个完整的辩论历史 Cache。
    这是一个简化的实现，假设轮次是 A->B, A->B ...
    """
    # 这是一个复杂的任务，我们先用一个简化逻辑：
    # 将 agent A 的最终 cache 和 agent B 最终 cache 的最后一部分拼接起来
    # 更准确的方法需要按 token 级别交错，非常复杂
    # 对于 Judge 来说，一个合理的简化是让他基于其中一个 agent 的最终视角来判断
    # 这里我们选择 A 的视角，并将 B 的最后一个回答拼接到 A 的最终 cache 后面
    print("[INFO] 正在为 Judge 准备融合的 KV Cache...")
    final_a_cache = agent_a.kv_cache
    last_b_answer_cache = agent_b.get_answer_cache_slice()
    
    if last_b_answer_cache:
        # TODO: 未来 RoPE 修正需要在这里应用
        return concatenate_kv_caches(final_a_cache, last_b_answer_cache)
    return final_a_cache


def main():
    MAX_DEBATE_ROUNDS = 3
    tokenizer, model = setup_environment()
    question_content, _, correct_answer, prompts = prepare_mmlu_sample_and_prompts()

    print("\n✅ 准备工作完成。")
    # ... 打印问题 ...

    print("\n[INFO] 初始化智能体和裁判...")
    agent_a = DebateAgent(model, tokenizer, prompts['system'], name="Agent A")
    agent_b = DebateAgent(model, tokenizer, prompts['system'], name="Agent B")
    judge = Judge(model, tokenizer, prompts['judge']) # Judge 现在是 Agent
    final_answer = None

    for r in range(1, MAX_DEBATE_ROUNDS + 1):
        # ... (辩论循环逻辑完全不变) ...
        print(f"\n--- 第{r}轮 ---")

        if r == 1:
            # 第一轮：独立发言
            agent_a.generate_response(new_user_prompt_content=question_content)
            agent_b.generate_response(new_user_prompt_content=question_content)
        else:
            # 后续轮次：相互反思
            # 注意: agent_b 反思的是 agent_a 上一轮的观点，反之亦然
            # 为了避免状态污染，我们先获取所有上一轮的cache
            cache_a_prev = agent_a.get_answer_cache_slice()
            cache_b_prev = agent_b.get_answer_cache_slice()

            print(f"\n[INFO] Agent A 开始反思 Agent B (第{r-1}轮)的观点...")
            agent_a.generate_response(prompts['reflection'], opponent_cache_slice=cache_b_prev)
            
            print(f"\n[INFO] Agent B 开始反思 Agent A (第{r-1}轮)的观点...")
            agent_b.generate_response(prompts['reflection'], opponent_cache_slice=cache_a_prev)

        # 解析本轮答案
        answer_a = agent_a.parse_answer_option()
        answer_b = agent_b.parse_answer_option()
        print(f"\n[Round {r} Results] Agent A 选择: {answer_a}, Agent B 选择: {answer_b}")

        # 检查是否达成共识
        if answer_a is not None and answer_a == answer_b:
            print(f"\n✅ 在第{r}轮达成共识: {answer_a}。辩论结束。")
            final_answer = answer_a
            break

    if final_answer is None:
        print(f"\n--- {MAX_DEBATE_ROUNDS}轮后未达成共识，进入基于 KV-Cache 的裁决阶段 ---")
        
        # !! 核心改动：使用 KV-Cache 进行裁决 !!
        final_debate_cache = interleave_and_combine_caches(agent_a, agent_b, MAX_DEBATE_ROUNDS)
        
        # 暂时注释掉Judge，因为这个拼接逻辑比较复杂
        # final_answer = judge.make_decision(question_content, final_debate_cache)
        # print(f"\n[INFO] 裁判最终决定: {final_answer}")
        
        # 作为一个临时的、更简单的裁决策略：
        print("[INFO] 裁决阶段使用临时策略：以Agent A的最终答案为准。")
        final_answer = agent_a.parse_answer_option()

    print(f"\n--- 最终结果 ---")
    # ... (打印最终结果的逻辑不变) ...
    print(f"辩论/裁决后的最终答案: {final_answer}")
    print(f"标准答案: {correct_answer}")
    if final_answer == correct_answer:
        print("结果正确！🎉")
    else:
        print("结果错误。")
    print("\n✅ 脚本执行完毕。")


if __name__ == "__main__":
    main()