import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys, os
from datasets import load_dataset
from tqdm import tqdm
import random

# --- 1. 设置与导入 ---
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.agents.debate_agent import DebateAgent
from src.agents.judge import Judge
from src.models.kv_cache_utils import concatenate_kv_caches

# --- 2. 核心函数 (大部分与之前相同) ---

def setup_environment():
    model_name = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    print(f"正在加载模型: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.float16, device_map="auto"
    ).eval()
    print(f"模型 '{model_name}' 加载成功。")
    return tokenizer, model

def prepare_prompts():
    """只准备通用的 prompt 模板"""
    general_system_prompt = (
        "You are a meticulous expert. Your goal is to find the correct answer to a multiple-choice question through careful reasoning. "
        "Please think step by step to analyze the problem  ."
    )

    user_instruction_format = (
        "Analyze the following question and options. "
        "Provide a step-by-step analysis in no more than 3-4 key steps. " 
        "Your entire reasoning should be concise and under 400 words. " 
        "Conclude your response in the following format:\n\n"
        "Reasoning: [Your step-by-step analysis here]\n\n"
        "Final Answer: [The single letter of your choice]"
    )

    reflection_instruction = (
        "You have received an argument from another expert. "
        "First, briefly summarize their key point. Then, critically evaluate their argument in relation to your own in 1-2 sentences."
        "Provide your updated step-by-step reasoning and conclude with your final answer (max 3 steps, under 150 words).\n\n"
        "Structure your response as:\n\n"
        "Summary: [...]\n\n"
        "Critique: [Your evaluation of the opponent's argument]\n\n"
        "Reasoning: [Your updated step-by-step analysis]\n\n"
        "Final Answer: [The single letter of your choice]"
    )

    judge_system_prompt = (
        "You are an impartial judge. You will be shown a debate. "
        "Your task is to analyze the arguments and determine the single best answer. "
        "Respond ONLY with the letter of the correct option (e.g., A)."
    )

    prompts = {
        "system": general_system_prompt,
        "user_format": user_instruction_format, # 新增
        "reflection": reflection_instruction,
        "judge": judge_system_prompt
    }

    return prompts

def format_mmlu_question(example):
    """将 MMLU 数据集的一个样本格式化为我们需要的内容"""
    question = example['question']
    # MMLU a,b,c,d 对应 choices 列表的 0,1,2,3
    options_list = example['choices']
    options_dict = {chr(65 + i): opt for i, opt in enumerate(options_list)}
    # MMLU answer 是数字索引
    correct_answer_letter = chr(65 + example['answer'])
    
    formatted_options = "\n".join([f"{key}: {value}" for key, value in options_dict.items()])
    question_content = f"Question: {question}\n\nOptions:\n{formatted_options}"
    
    return question_content, options_dict, correct_answer_letter

def interleave_and_combine_caches(agent_a, agent_b):
    """为 Judge 准备融合的 KV Cache 的简化实现"""
    final_a_cache = agent_a.kv_cache
    last_b_answer_cache = agent_b.get_answer_cache_slice()
    if last_b_answer_cache:
        return concatenate_kv_caches(final_a_cache, last_b_answer_cache)
    return final_a_cache

# --- 3. 将单次辩论封装成一个函数 ---

def run_single_debate_trial(model, tokenizer, prompts, mmlu_example, max_rounds=3):
    """
    对单个 MMLU 样本运行一次完整的多轮辩论+裁决流程。
    """
    question_content, _, correct_answer = format_mmlu_question(mmlu_example)

    print("\n" + "="*80)
    print(f"正在处理新问题...")
    print(f"问题内容:\n{question_content}")
    print(f"正确答案: {correct_answer}")
    print("="*80 + "\n")

    # 初始化
    agent_a = DebateAgent(model, tokenizer, prompts['system'], name="Agent A")
    agent_b = DebateAgent(model, tokenizer, prompts['system'], name="Agent B")
    judge = Judge(model, tokenizer, prompts['judge'])
    final_answer = None

    # 辩论循环
    for r in range(1, max_rounds + 1):
        if r == 1:
            user_prompt_for_r1 = f"{question_content}\n\n{prompts['user_format']}"
            
            agent_a.generate_response(new_user_prompt_content=user_prompt_for_r1)
            agent_b.generate_response(new_user_prompt_content=user_prompt_for_r1)
        else:
            cache_a_prev = agent_a.get_answer_cache_slice()
            cache_b_prev = agent_b.get_answer_cache_slice()
            agent_a.generate_response(prompts['reflection'], opponent_cache_slice=cache_b_prev)
            agent_b.generate_response(prompts['reflection'], opponent_cache_slice=cache_a_prev)

        answer_a = agent_a.parse_answer_option()
        answer_b = agent_b.parse_answer_option()

        if answer_a is not None and answer_a == answer_b:
            final_answer = answer_a
            break
    
    # 裁决
    if final_answer is None:
        final_debate_cache = interleave_and_combine_caches(agent_a, agent_b)
        final_answer = judge.make_decision(question_content, final_debate_cache)
        # final_answer = agent_a.parse_answer_option() if agent_a.parse_answer_option() else "NO_ANSWER"

    return final_answer, correct_answer

# --- 4. 主评测循环 ---

def main():
    # --- 设置 ---
    MAX_SAMPLES = 10 # !! 重要：先在一个小子集上测试 !!
    MAX_DEBATE_ROUNDS = 3

    # --- 加载模型和 prompts ---
    tokenizer, model = setup_environment()
    prompts = prepare_prompts()

    # --- 加载 MMLU 数据集 ---
    print("\n[INFO] 正在加载 MMLU 数据集...")
    # 我们选择一个特定的子集，来快速测试
    dataset = load_dataset("cais/mmlu", 'professional_law', split='test')
    # 为了防止每次都测同样的问题，可以打乱一下
    dataset = dataset.shuffle(seed=42).select(range(MAX_SAMPLES))
    print(f"[INFO] 数据集加载完成。将评测 {len(dataset)} 个样本。")

    # --- 评测循环 ---
    correct_predictions = 0
    errors = 0
    
    for example in tqdm(dataset, desc="MMLU Evaluation"):
        try:
            final_answer, correct_answer = run_single_debate_trial(
                model, tokenizer, prompts, example, MAX_DEBATE_ROUNDS
            )
            print(f"最终答案: {final_answer}, 正确答案: {correct_answer}")
            if final_answer == correct_answer:
                correct_predictions += 1
        except Exception as e:
            print(f"\n[ERROR] 处理样本时发生错误: {e}")
            errors += 1
    
    # --- 打印最终结果 ---
    total_processed = len(dataset) - errors
    accuracy = (correct_predictions / total_processed) * 100 if total_processed > 0 else 0
    
    print("\n\n--- 评测结果 ---")
    print(f"总样本数: {len(dataset)}")
    print(f"成功处理: {total_processed}")
    print(f"正确预测: {correct_predictions}")
    print(f"错误数: {errors}")
    print(f"准确率: {accuracy:.2f}%")
    print("------------------")

if __name__ == "__main__":
    main()