import torch
import json
import os
import re
import threading
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel  # 添加PEFT库导入
from rag import simple_rag_query
from typing import Optional

# 全局变量
MODEL_PATH = ""  # 基础模型路径
LORA_PATH = ""  # LoRA适配器路径
PROMPT = "你是一个医学专家，你需要根据用户的问题，给出带有思考的回答。"
SUMMARY_PROMPT = """你是一个资料提取助手。你的任务是根据提供的问题（或查询），从给定的文本资料中提取所有直接相关的信息。
请确保提取的信息是客观的，并且不包含你自己的解释、推断或对主问题的直接回答。
如果资料中存在与查询直接相关的信息，请以 "YES#" 开头，后跟提取的相关文本。
如果资料中没有与查询直接相关的信息，请回答 "NO#无相关内容"。

提供的查询是："{question}"
提供的资料是："{rag_context}"

你的提取结果是："""
def initialize_model_pretrain(model_path, device_id=0):
    """
    初始化预训练模型（不加载LoRA），用于总结功能
    
    Args:
        model_path: 基础模型路径
        device_id: 设备ID，默认为0
    
    Returns:
        tuple: (tokenizer, model) 或在失败时 (None, None)
    """
    if not os.path.exists(model_path):
        print(f"警告: 基础模型路径 '{model_path}' 未找到。")
        return None, None
    
    try:
        print(f"正在GPU {device_id}上加载预训练模型用于总结: {model_path}")
        
        # 设置设备
        torch.cuda.set_device(device_id)
        
        # 加载tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, 
            use_fast=False, 
            trust_remote_code=True,
            padding_side="left"
        )
        
        # 加载模型
        model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            torch_dtype=torch.bfloat16,
            device_map=f"cuda:{device_id}",  # 指定具体设备
            trust_remote_code=True
        )
        
        # 设置pad_token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            print(f"Tokenizer pad_token_id set to eos_token_id: {tokenizer.eos_token_id}")
        
        model.eval()  # 设置为评估模式
        
        print(f"预训练模型加载成功，设备: GPU {device_id}")
        return tokenizer, model
        
    except Exception as e:
        print(f"加载预训练模型时出错: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def initialize_model(model_path=None, lora_path=None, thread_id=None):
    """
    初始化带LoRA适配器的模型
    
    Args:
        model_path: 基础模型路径
        lora_path: LoRA适配器路径
        thread_id: 线程ID，用于日志
        
    Returns:
        tuple: (tokenizer, model) 或在失败时 (None, None)
    """
    base_path = model_path if model_path else MODEL_PATH
    adapter_path = lora_path if lora_path else LORA_PATH
    thread_info = f"线程 {thread_id}: " if thread_id is not None else ""
    
    if not os.path.exists(base_path):
        print(f"{thread_info}警告: 基础模型路径 '{base_path}' 未找到。")
        return None, None
        
    if not os.path.exists(adapter_path):
        print(f"{thread_info}警告: LoRA适配器路径 '{adapter_path}' 未找到。")
        return None, None
    
    try:
        print(f"{thread_info}正在加载基础模型: {base_path}")
        tokenizer = AutoTokenizer.from_pretrained(base_path, use_fast=False, trust_remote_code=True)
        base_model = AutoModelForCausalLM.from_pretrained(
            base_path, 
            device_map="auto", 
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
            print(f"{thread_info}Tokenizer pad_token_id set to eos_token_id: {tokenizer.eos_token_id}")
        
        print(f"{thread_info}正在加载LoRA适配器: {adapter_path}")
        # 使用PeftModel加载LoRA适配器
        model = PeftModel.from_pretrained(
            base_model,
            adapter_path,
            torch_dtype=torch.bfloat16
        )
        model.eval()  # 设置为评估模式
        
        print(f"{thread_info}LoRA模型加载成功")
        return tokenizer, model
    except Exception as e:
        print(f"{thread_info}加载模型时出错: {e}")
        return None, None

# 其余函数保持不变
def predict(messages_list, tokenizer, model, thread_id=None):
    """
    使用提供的模型实例进行预测，而不是使用全局变量
    
    Args:
        messages_list: 消息列表，每个元素为一个完整的提示
        tokenizer: 分词器实例
        model: 模型实例
        thread_id: 线程ID，用于日志
        
    Returns:
        list: 模型响应列表
    """
    thread_info = f"线程 {thread_id}: " if thread_id is not None else ""
    
    if not tokenizer or not model:
        print(f"{thread_info}错误: 模型或分词器为None")
        return []
    
    device = next(model.parameters()).device  # 自动检测模型所在设备
    responses = []
    
    for messages in messages_list:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([text], return_tensors="pt").to(device)
        
        try:
            generated_ids_full = model.generate(
                model_inputs.input_ids, 
                max_new_tokens=2048
            )
            
            generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids_full)]
            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            responses.append(response)
        except Exception as e:
            print(f"{thread_info}生成预测时出错: {e}")
            responses.append(f"错误: 生成失败 - {str(e)}")
    
    return responses

def summary(rag_context, question, tokenizer, model, thread_id=None):
    """
    使用模型对RAG上下文进行信息提取
    
    Args:
        rag_context: RAG查询得到的上下文信息
        question: 原始问题
        tokenizer: 分词器实例
        model: 模型实例
        thread_id: 线程ID，用于日志
        
    Returns:
        str: 提取后的相关信息
    """
    thread_info = f"线程 {thread_id}: " if thread_id is not None else ""
    
    if not rag_context or not rag_context.strip():
        print(f"{thread_info}RAG上下文为空，跳过信息提取")
        return ""
    
    if not tokenizer or not model:
        print(f"{thread_info}模型未初始化，跳过信息提取")
        return ""
    
    try:
        # 构建信息提取请求的消息
        summary_input = SUMMARY_PROMPT.format(question=question, rag_context=rag_context)
        
        summary_messages = [
            {"role": "system", "content": "你是一个专业的医学资料提取助手。"},
            {"role": "user", "content": summary_input}
        ]
        
        print(f"{thread_info}开始提取RAG相关信息，原始长度: {len(rag_context)} 字符")
        
        # 调用模型进行信息提取
        summary_responses = predict([summary_messages], tokenizer, model, thread_id)
        
        if summary_responses and len(summary_responses) > 0:
            raw_extracted_text = summary_responses[0].strip()
            print(raw_extracted_text)    
            # 去除 <think>...</think> 部分
            clean_extracted_text = re.sub(r"<think>.*?</think>", "", raw_extracted_text, flags=re.DOTALL | re.IGNORECASE).strip()
            
            # 如果去除思考部分后文本为空，使用原始文本
            if not clean_extracted_text:
                print(f"{thread_info}警告: 去除思考部分后提取结果为空，使用原始响应")
                clean_extracted_text = raw_extracted_text
            
            print(f"{thread_info}RAG信息提取完成，原始长度: {len(raw_extracted_text)} → 清理后长度: {len(clean_extracted_text)} 字符")
            
            # 处理提取结果
            if clean_extracted_text.startswith("YES#"):
                # 提取YES#后的内容
                extracted_content = clean_extracted_text[4:].strip()  # 去除"YES#"
                
                if len(extracted_content) < 10:
                    print(f"{thread_info}警告: 提取的相关信息过短，可能存在问题")
                elif len(extracted_content) > 2000:
                    print(f"{thread_info}警告: 提取的相关信息过长，进行截断")
                    extracted_content = extracted_content[:1800] + "..."
                
                print(f"{thread_info}✓ 成功提取到相关信息")
                return extracted_content
                
            elif clean_extracted_text.startswith("NO#"):
                print(f"{thread_info}✓ 模型判断无相关内容")
                return ""
            else:
                # 如果没有按预期格式返回，尝试提取有用信息
                print(f"{thread_info}警告: 提取结果格式不符合预期，尝试直接使用")
                if len(clean_extracted_text) > 2000:
                    clean_extracted_text = clean_extracted_text[:1800] + "..."
                return clean_extracted_text
        else:
            print(f"{thread_info}信息提取失败，返回空字符串")
            return ""
            
    except Exception as e:
        print(f"{thread_info}信息提取过程出错: {e}")
        import traceback
        traceback.print_exc()
        return ""

def prepare_message_for_item(item, tokenizer, model, summary_tokenizer, summary_model, thread_id=None):
    """
    为单个输入项准备模型消息，使用两阶段处理：先总结RAG信息，再回答问题
    
    Args:
        item: 输入项字典
        tokenizer: 问答模型的分词器实例（LoRA模型）
        model: 问答模型实例（LoRA模型）
        summary_tokenizer: 总结模型的分词器实例（预训练模型）
        summary_model: 总结模型实例（预训练模型）
        thread_id: 线程ID，用于日志
        
    Returns:
        tuple: (消息列表, 原始RAG上下文, 总结后的RAG上下文) 
    """
    thread_info = f"线程 {thread_id}: " if thread_id is not None else ""
    
    instruction = PROMPT
    question_text = item.get('question', '')
    question_type_text = item.get('question_type', '题目')
    options_dict = item.get('option', {})
    options_str = ""
    
    if isinstance(options_dict, dict):
        for key, value in sorted(options_dict.items()): 
            if value and value.strip(): 
                options_str += f"{key}. {value}\n" 
    
    options_str = options_str.strip()
    
    # 构建基本的输入文本
    input_text = f"这是一道{question_type_text}。\n题目：{question_text}\n\n选项：\n{options_str}"
    
    # RAG查询部分
    rag_contexts = []
    combined_rag_context = ""
    summarized_rag_context = ""
    
    try:
        print(f"{thread_info}========== 阶段1: RAG信息收集 ==========")
        
        # 1. 对问题本身进行RAG查询
        if question_text.strip():
            print(f"{thread_info}正在为问题查询RAG系统: {question_text[:50]}...")
            question_rag_context = simple_rag_query(question_text, top_k=3)
            
            if question_rag_context:
                print(f"{thread_info}✓ 问题RAG查询成功，获得 {len(question_rag_context)} 字符的相关信息")
                rag_contexts.append(f"【问题相关资料】\n{question_rag_context}")
            else:
                print(f"{thread_info}✗ 问题RAG查询未返回相关结果")
        
        # 2. 对每个选项进行RAG查询
        if isinstance(options_dict, dict):
            for option_key, option_value in sorted(options_dict.items()):
                if option_value and option_value.strip():
                    print(f"{thread_info}正在为选项{option_key}查询RAG系统: {option_value[:30]}...")
                    option_rag_context = simple_rag_query(option_value, top_k=3)
                    
                    if option_rag_context:
                        print(f"{thread_info}✓ 选项{option_key}RAG查询成功，获得 {len(option_rag_context)} 字符的相关信息")
                        rag_contexts.append(f"【选项{option_key}相关资料】\n{option_rag_context}")
                    else:
                        print(f"{thread_info}✗ 选项{option_key}RAG查询未返回相关结果")
        
        # 3. 整合所有RAG结果
        if rag_contexts:
            combined_rag_context = "\n\n".join(rag_contexts)
            print(f"{thread_info}✓ 总共获得 {len(rag_contexts)} 个RAG查询结果，总长度 {len(combined_rag_context)} 字符")
        else:
            print(f"{thread_info}✗ 所有RAG查询都未返回相关结果")
            
        # 4. 阶段2: 使用预训练模型总结RAG信息
        if combined_rag_context and summary_tokenizer and summary_model:
            print(f"{thread_info}========== 阶段2: RAG信息总结（使用预训练模型） ==========")
            summarized_rag_context = summary(
                combined_rag_context, 
                question_text, 
                summary_tokenizer,   # 使用预训练模型的tokenizer
                summary_model,       # 使用预训练模型
                thread_id
            )
            
            if summarized_rag_context:
                print(f"{thread_info}✓ RAG信息总结成功")
                print(f"{thread_info}原始长度: {len(combined_rag_context)} → 总结长度: {len(summarized_rag_context)}")
                # 使用总结后的信息
                input_text = f"参考以下医学知识：\n{summarized_rag_context}\n\n---\n\n{input_text}"
            else:
                print(f"{thread_info}✗ RAG信息总结失败，使用原始信息")
                # 如果总结失败，仍使用原始信息但进行截断
                if len(combined_rag_context) > 10000:
                    truncated_context = combined_rag_context[:10000] + "\n...(内容过长已截断)"
                    input_text = f"参考以下医学知识：\n{truncated_context}\n\n---\n\n{input_text}"
                else:
                    input_text = f"参考以下医学知识：\n{combined_rag_context}\n\n---\n\n{input_text}"
        elif combined_rag_context:
            # 如果没有总结模型，直接使用原始RAG信息
            print(f"{thread_info}========== 跳过总结阶段（预训练模型未加载） ==========")
            if len(combined_rag_context) > 3000:
                truncated_context = combined_rag_context[:3000] + "\n...(内容过长已截断)"
                input_text = f"参考以下医学知识：\n{truncated_context}\n\n---\n\n{input_text}"
            else:
                input_text = f"参考以下医学知识：\n{combined_rag_context}\n\n---\n\n{input_text}"
        
        print(f"{thread_info}========== 阶段3: 准备最终问答（使用LoRA模型） ==========")
        
    except Exception as e:
        print(f"{thread_info}✗ RAG查询或总结过程出错: {e}")
    
    # 准备最终的消息
    messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": input_text}
    ]
    
    print(f"{thread_info}最终输入文本长度: {len(input_text)} 字符")
    
    return messages, combined_rag_context, summarized_rag_context

def prepare_message_for_item1(item, thread_id=None):
    """
    为单个输入项准备模型消息的简化版本，不进行RAG信息收集和总结
    
    Args:
        item: 输入项字典
        thread_id: 线程ID，用于日志
        
    Returns:
        tuple: (消息列表, 空字符串, 空字符串) - 保持与原函数返回格式一致
    """
    thread_info = f"线程 {thread_id}: " if thread_id is not None else ""
    
    instruction = PROMPT
    question_text = item.get('question', '')
    question_type_text = item.get('question_type', '题目')
    options_dict = item.get('option', {})
    options_str = ""
    
    # 构建选项字符串
    if isinstance(options_dict, dict):
        for key, value in sorted(options_dict.items()): 
            if value and value.strip(): 
                options_str += f"{key}. {value}\n" 
    
    options_str = options_str.strip()
    
    # 构建基本的输入文本（不包含RAG信息）
    input_text = f"这是一道{question_type_text}。\n题目：{question_text}\n\n选项：\n{options_str}"
    
    print(f"{thread_info}========== 准备问答（无RAG） ==========")
    print(f"{thread_info}问题: {question_text[:50]}...")
    print(f"{thread_info}选项数量: {len(options_dict)}")
    
    # 准备最终的消息
    messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": input_text}
    ]
    
    print(f"{thread_info}最终输入文本长度: {len(input_text)} 字符")
    
    # 返回格式与原函数保持一致：(消息列表, 原始RAG上下文, 总结后的RAG上下文)
    # 这里RAG相关的都返回空字符串
    return messages, "", ""

def parse_model_output(raw_output_text):
    """
    解析模型输出，提取思考、答案和解释部分
    
    Args:
        raw_output_text: 原始模型输出文本
        
    Returns:
        tuple: (思考内容, 答案, 解释)
    """
    parsed_thinking = ""
    parsed_answer = ""
    parsed_explanation = ""

    # 1. 提取 <think>...</think> 内容
    think_match = re.search(r"<think>(.*?)</think>", raw_output_text, re.DOTALL | re.IGNORECASE)
    
    text_for_answer_explanation = raw_output_text # 默认为原始文本

    if think_match:
        parsed_thinking = think_match.group(1).strip()
        # 从原始文本中移除 <think>...</think> 块
        text_for_answer_explanation = re.sub(r"<think>.*?</think>", "", raw_output_text, count=1, flags=re.DOTALL | re.IGNORECASE).strip()
    else:
        text_for_answer_explanation = raw_output_text.strip()

    # 2. 解析 "答案：" 和 "解释："
    answer_marker = "答案："
    explanation_marker = "解释："
    
    # 使用正则表达式提取答案
    answer_regex = rf"{answer_marker}\s*(.*?)\s*(?:{explanation_marker}|$)"
    answer_match_in_remaining = re.search(answer_regex, text_for_answer_explanation, re.DOTALL | re.IGNORECASE)
    if answer_match_in_remaining:
        parsed_answer = answer_match_in_remaining.group(1).strip()

    # 提取解释
    explanation_regex = rf"{explanation_marker}\s*(.*)"
    explanation_match_in_remaining = re.search(explanation_regex, text_for_answer_explanation, re.DOTALL | re.IGNORECASE)
    if explanation_match_in_remaining:
        parsed_explanation = explanation_match_in_remaining.group(1).strip()

    # Fallback 逻辑
    if not parsed_answer and not parsed_explanation and text_for_answer_explanation:
        answer_start_idx_fb = text_for_answer_explanation.lower().find(answer_marker.lower())
        explanation_start_idx_fb = text_for_answer_explanation.lower().find(explanation_marker.lower())

        if answer_start_idx_fb != -1:
            if explanation_start_idx_fb != -1 and explanation_start_idx_fb > answer_start_idx_fb:
                parsed_answer = text_for_answer_explanation[answer_start_idx_fb + len(answer_marker):explanation_start_idx_fb].strip()
                parsed_explanation = text_for_answer_explanation[explanation_start_idx_fb + len(explanation_marker):].strip()
            else:
                parsed_answer = text_for_answer_explanation[answer_start_idx_fb + len(answer_marker):].strip()
                if not parsed_explanation:
                     parsed_explanation = "（解释部分未按预期格式提供或未找到）"
        elif explanation_start_idx_fb != -1:
             parsed_explanation = text_for_answer_explanation[explanation_start_idx_fb + len(explanation_marker):].strip()
             if not parsed_answer:
                 parsed_answer = "（答案部分未按预期格式提供或未找到）"
        else:
            if text_for_answer_explanation:
                print(f"警告: 模型输出不包含明确的 '答案：' 或 '解释：' 标记。处理后文本: '{text_for_answer_explanation[:200]}...'")
                parsed_explanation = text_for_answer_explanation
                if not parsed_answer: parsed_answer = "（未知）"

    # 清理 parsed_answer (选项字母)
    if parsed_answer:
        cleaned_options = re.findall(r"[A-E]", parsed_answer.upper())
        if cleaned_options:
            parsed_answer = "".join(sorted(list(set(cleaned_options))))

    return parsed_thinking, parsed_answer, parsed_explanation

def parse_model_output1(raw_output_text, options_dict=None):
    """
    解析模型输出，使用与evaluate_on_choice_formula相同的答案提取逻辑，增强多选题处理
    
    Args:
        raw_output_text: 原始模型输出文本
        options_dict: 选项字典，用于验证提取的答案
        
    Returns:
        tuple: (思考内容, 答案, 解释)
    """
    parsed_thinking = ""
    parsed_answer = ""
    parsed_explanation = ""

    # 1. 提取 <think>...</think> 内容
    think_match = re.search(r"<think>(.*?)</think>", raw_output_text, re.DOTALL | re.IGNORECASE)
    
    text_for_answer_explanation = raw_output_text # 默认为原始文本

    if think_match:
        parsed_thinking = think_match.group(1).strip()
        # 从原始文本中移除 <think>...</think> 块
        text_for_answer_explanation = re.sub(r"<think>.*?</think>", "", raw_output_text, count=1, flags=re.DOTALL | re.IGNORECASE).strip()
    else:
        text_for_answer_explanation = raw_output_text.strip()

    # 2. 解析答案 - 使用与evaluate_on_choice_formula完全相同的逻辑，但增强多选题处理
    predicted_answer_letter = ""
    
    # 优先从<answer>标签中提取答案
    answer_tag_match = re.search(r'<answer>(.*?)</answer>', text_for_answer_explanation, re.DOTALL | re.IGNORECASE)
    if answer_tag_match:
        answer_content = answer_tag_match.group(1).strip()
        
        # 方法2（先处理）：处理多行答案格式，针对每行都有选项字母的情况
        # 更宽松的多行选项匹配
        option_lines = re.findall(r'(?:^|\n|\r|\s+)([A-E])(?:[\.\s\n:：,，、]|$)', answer_content.upper())
        if option_lines and options_dict:
            # 确保提取的字母都是有效选项
            valid_options = [letter for letter in option_lines if letter in options_dict]
            if valid_options:
                # 组合成一个字符串，如"ABC"
                predicted_answer_letter = ''.join(sorted(set(valid_options)))
        
        # 方法1（后处理）：提取标准格式的答案（针对"答案：ABC"这种格式）
        if not predicted_answer_letter:
            letter_match = re.search(r'[单选多选]?题?答案[是为：:]*\s*([A-Z]+)', answer_content, re.IGNORECASE)
            if letter_match:
                extracted_letter = letter_match.group(1).upper()
                if options_dict and all(letter in options_dict for letter in extracted_letter):
                    predicted_answer_letter = extracted_letter
        
        # 方法3（最后处理）：如果上面方法都失败，查找所有大写字母作为候选
        if not predicted_answer_letter and options_dict:
            # 多选题可能直接列出选项字母，如"正确答案为：B C"
            # 先查找是否有明确的"正确答案"后跟选项字母
            conclusion_match = re.search(r'(正确[的答案]为[是：:]*|答案[是为：:]*|选择[：:]*)\s*([A-Z\s,，和及与\+]+)', answer_content, re.IGNORECASE)
            if conclusion_match:
                # 从结论中提取所有有效选项字母
                candidate_letters = re.findall(r'[A-E]', conclusion_match.group(2).upper())
                valid_letters = [letter for letter in candidate_letters if letter in options_dict]
                if valid_letters:
                    predicted_answer_letter = ''.join(sorted(set(valid_letters)))
            
            # 如果上述方法都失败，提取所有有效的大写字母
            if not predicted_answer_letter:
                valid_letters = [letter for letter in re.findall(r'([A-E])', answer_content.upper()) 
                                if letter in options_dict]
                if valid_letters:
                    # 去重并排序，比如从"ABCA"变成"ABC"
                    predicted_answer_letter = ''.join(sorted(set(valid_letters)))
    
    parsed_answer = predicted_answer_letter
    
    # 3. 解析 "解释："部分
    explanation_marker = "解释："
    explanation_regex = rf"{explanation_marker}\s*(.*)"
    explanation_match_in_remaining = re.search(explanation_regex, text_for_answer_explanation, re.DOTALL | re.IGNORECASE)
    if explanation_match_in_remaining:
        parsed_explanation = explanation_match_in_remaining.group(1).strip()
    else:
        # Fallback 提取解释
        explanation_start_idx_fb = text_for_answer_explanation.lower().find(explanation_marker.lower())
        if explanation_start_idx_fb != -1:
            parsed_explanation = text_for_answer_explanation[explanation_start_idx_fb + len(explanation_marker):].strip()
        else:
            # 如果没有明确的"解释："标记，移除<answer>部分后剩余的文本作为解释
            if answer_tag_match:
                parsed_explanation = re.sub(r'<answer>.*?</answer>', '', text_for_answer_explanation, flags=re.DOTALL | re.IGNORECASE).strip()
            else:
                parsed_explanation = text_for_answer_explanation
    
    return parsed_thinking, parsed_answer, parsed_explanation


def process_items_batch(items_batch, tokenizer, model, summary_tokenizer, summary_model, thread_id=None):
    """
    处理一批输入项，使用传入的模型和分词器
    
    Args:
        items_batch: 输入项列表
        tokenizer: 问答模型的分词器实例
        model: 问答模型实例（LoRA模型）
        summary_tokenizer: 总结模型的分词器实例（预训练模型）
        summary_model: 总结模型实例（预训练模型）
        thread_id: 线程ID，用于日志
        
    Returns:
        list: 处理后的结果列表
    """
    thread_info = f"线程 {thread_id}: " if thread_id is not None else ""
    
    if not items_batch:
        print(f"{thread_info}没有可处理的项目")
        return []
    
    print(f"{thread_info}准备处理 {len(items_batch)} 个项目...")
    print(f"{thread_info}LoRA模型设备: {next(model.parameters()).device if model else 'N/A'}")
    print(f"{thread_info}预训练模型设备: {next(summary_model.parameters()).device if summary_model else 'N/A'}")
    
    # 检查模型是否正确加载
    if not tokenizer or not model:
        print(f"{thread_info}错误: LoRA模型或分词器未正确加载")
        return []
    
    if not summary_tokenizer or not summary_model:
        print(f"{thread_info}警告: 预训练模型未正确加载，将跳过总结功能")
    
    # 为每个项目准备消息，同时收集RAG上下文
    all_messages = []
    all_original_rag_contexts = []
    all_summarized_rag_contexts = []
    
    for i, item in enumerate(items_batch):
        print(f"\n{thread_info}========== 准备项目 {i+1}/{len(items_batch)} ==========")
        
        try:
            messages, original_rag, summarized_rag = prepare_message_for_item(
                item, tokenizer, model, summary_tokenizer, summary_model, thread_id
            )
            # messages, original_rag, summarized_rag = prepare_message_for_item1(
            #     item, thread_id
            # )
            all_messages.append(messages)
            all_original_rag_contexts.append(original_rag)
            all_summarized_rag_contexts.append(summarized_rag)
        except Exception as e:
            print(f"{thread_info}准备项目 {i+1} 时出错: {e}")
            # 创建一个默认的消息
            default_messages = [
                {"role": "system", "content": PROMPT},
                {"role": "user", "content": f"请回答问题: {item.get('question', '无问题内容')}"}
            ]
            all_messages.append(default_messages)
            all_original_rag_contexts.append("")
            all_summarized_rag_contexts.append("")
    
    # 发送到LoRA模型进行最终预测
    print(f"\n{thread_info}========== 开始最终问答阶段（使用LoRA模型） ==========")
    print(f"{thread_info}开始向LoRA模型发送 {len(all_messages)} 条请求进行最终预测...")
    
    try:
        all_raw_responses = predict(all_messages, tokenizer, model, thread_id)
        print(f"{thread_info}LoRA模型预测完成")
    except Exception as e:
        print(f"{thread_info}LoRA模型预测失败: {e}")
        # 创建默认响应
        all_raw_responses = [f"预测失败: {str(e)}"] * len(all_messages)
    
    # 处理结果
    results = []
    for i, (item, raw_response, original_rag, summarized_rag) in enumerate(
        zip(items_batch, all_raw_responses, all_original_rag_contexts, all_summarized_rag_contexts)
    ):
        item_id = item.get('id', 'N/A')
        print(f"\n{thread_info}========== 处理结果 {i+1}/{len(items_batch)} (ID: {item_id}) ==========")
        
        # 打印问题内容
        print(f"\n{thread_info}问题: {item.get('question', '无问题内容')}")
        
        # 打印选项
        options_dict = item.get('option', {})
        if isinstance(options_dict, dict):
            print(f"\n{thread_info}选项:")
            for key, value in sorted(options_dict.items()):
                if value and value.strip():
                    print(f"{thread_info}- {key}. {value}")
        
        # 打印RAG上下文信息
        if original_rag:
            print(f"\n{thread_info}原始RAG上下文长度: {len(original_rag)} 字符")
        if summarized_rag:
            print(f"{thread_info}总结RAG上下文长度: {len(summarized_rag)} 字符")
            if original_rag:
                compression_ratio = len(summarized_rag) / len(original_rag) * 100
                print(f"{thread_info}压缩比: {compression_ratio:.1f}%")
        
        # 打印模型原始输出
        print(f"\n{thread_info}LoRA模型原始输出:\n{'-'*40}\n{raw_response}\n{'-'*40}")

        # 解析模型输出
        try:
            parsed_thinking, parsed_answer, parsed_explanation = parse_model_output1(raw_response, options_dict)
        except Exception as e:
            print(f"{thread_info}解析模型输出时出错: {e}")
            parsed_thinking = ""
            parsed_answer = "解析失败"
            parsed_explanation = raw_response
        
        # 打印解析结果
        print(f"\n{thread_info}解析结果:")
        print(f"{thread_info}- 答案: {parsed_answer}")
        if parsed_explanation:
            display_explanation = parsed_explanation[:100] + "..." if len(parsed_explanation) > 100 else parsed_explanation
            print(f"{thread_info}- 解释: {display_explanation}")
        if parsed_thinking:
            display_thinking = parsed_thinking[:100] + "..." if len(parsed_thinking) > 100 else parsed_thinking
            print(f"{thread_info}- 思考过程: {display_thinking}")
        
        # 清理换行符和多余空格的函数
        def clean_text(text):
            if not text:
                return ""
            # 替换多个连续换行符为单个空格
            cleaned = re.sub(r'\n+', ' ', str(text))
            # 替换多个连续空格为单个空格
            cleaned = re.sub(r'\s+', ' ', cleaned)
            return cleaned.strip()
        
        # 构建输出项，添加rag相关字段
        output_item = {
            "id": item_id,
            "question": item.get("question"),
            "question_type": item.get("question_type"),
            "option": item.get("option", {}),
            "answer": parsed_answer,
            "llm_answer": clean_text(parsed_explanation),
            "think": clean_text(parsed_thinking),
            "rag_summarized": clean_text(summarized_rag),
            # "rag": clean_text(summarized_rag if summarized_rag else original_rag)
        }
        
        results.append(output_item)
        print(f"\n{thread_info}项目 {item_id} 处理完毕")
        
    return results
def save_results_to_file(results, output_path, thread_lock=None):
    """
    将结果保存到文件
    
    Args:
        results: 结果列表
        output_path: 输出文件路径
        thread_lock: 线程锁，用于多线程环境
    """
    if not results:
        print("没有结果需要保存")
        return
    
    # 在多线程环境中使用锁
    if thread_lock:
        thread_lock.acquire()
    
    try:
        # 读取现有文件
        existing_data = []
        if os.path.exists(output_path):
            with open(output_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    existing_data = json.loads(content)
        
        # 创建ID到项目的映射
        item_map = {item.get('id'): item for item in existing_data if isinstance(item, dict) and 'id' in item}
        
        # 更新或添加新处理的项目
        for item in results:
            if 'id' in item:
                item_map[item['id']] = item
        
        # 转换回列表并排序
        final_data = list(item_map.values())
        final_data.sort(key=lambda x: x.get('id', float('inf')))
        
        # 保存到文件
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(final_data, f, ensure_ascii=False, indent=4)
            
        print(f"已保存 {len(final_data)} 个项目到 {output_path}")
        
    except Exception as e:
        print(f"保存结果时出错: {e}")
    finally:
        # 释放锁
        if thread_lock:
            thread_lock.release()

def process_input_file(input_path, output_path, tokenizer=None, model=None, start_idx=0, step_size=1, thread_id=None):
    """
    处理整个输入文件中的项目
    
    Args:
        input_path: 输入文件路径
        output_path: 输出文件路径
        tokenizer: 分词器实例
        model: 模型实例
        start_idx: 起始索引
        step_size: 步长
        thread_id: 线程ID
    """
    thread_info = f"线程 {thread_id}: " if thread_id is not None else ""
    
    # 读取输入文件
    try:
        with open(input_path, "r", encoding="utf-8") as f:
            all_input_items = json.load(f)
    except Exception as e:
        print(f"{thread_info}读取输入文件失败: {e}")
        return []
    
    # 根据起始索引和步长选择要处理的项目
    items_to_process = all_input_items[start_idx::step_size]
    print(f"{thread_info}将处理 {len(items_to_process)} 个项目")
    
    # 处理选定的项目
    results = process_items_batch(items_to_process, tokenizer, model, thread_id)
    
    return results

def main():
    """
    主函数，用于直接运行脚本时的执行流程
    """
    # 默认的输入和输出文件路径
    input_json_file = "" 
    output_json_file = ""  # 修改输出文件名以区分
    
    # 可以接受命令行参数来指定模型和适配器路径
    # 这里仅使用默认值进行示例
    base_model_path = MODEL_PATH
    lora_adapter_path = LORA_PATH
    
    print(f"开始处理文件: {input_json_file}")
    print(f"使用基础模型: {base_model_path}")
    print(f"使用LoRA适配器: {lora_adapter_path}")
    
    # 初始化LoRA模型
    tokenizer, model = initialize_model(base_model_path, lora_adapter_path)
    if tokenizer is None or model is None:
        print("模型初始化失败，退出")
        return
    
    # 处理文件
    results = process_input_file(input_json_file, output_json_file, tokenizer=tokenizer, model=model)
    
    # 保存结果
    save_results_to_file(results, output_json_file)
    
    print("处理完成")

if __name__ == "__main__":
    main()

def initialize_rag(rag_name: Optional[str] = None, thread_id=None):
    """
    初始化RAG系统
    
    Args:
        rag_name: 指定的RAG系统名称，如果为None则使用默认或让用户选择
        thread_id: 线程ID，用于日志
        
    Returns:
        bool: 初始化是否成功
    """
    thread_info = f"线程 {thread_id}: " if thread_id is not None else ""
    
    try:
        from rag import switch_rag_system, list_available_rags, get_rag_system_non_interactive
        
        print(f"{thread_info}正在初始化RAG系统...")
        
        # 如果没有指定RAG名称，根据文件名自动判断或使用默认
        if rag_name is None:
            # 可以根据处理的文件自动选择RAG系统
            # 这里先使用默认的非交互式方式
            rag_system = get_rag_system_non_interactive(use_default=True)
            if rag_system is None:
                print(f"{thread_info}❌ 无法加载默认RAG系统")
                return False
            print(f"{thread_info}✅ 使用默认RAG系统")
            return True
        else:
            # 切换到指定的RAG系统
            success = switch_rag_system(rag_name)
            if success:
                print(f"{thread_info}✅ RAG系统初始化成功: {rag_name}")
                return True
            else:
                print(f"{thread_info}❌ RAG系统初始化失败: {rag_name}")
                # 显示可用的RAG系统
                print(f"{thread_info}可用的RAG系统:")
                available_rags = list_available_rags()
                for rag_id in available_rags:
                    print(f"{thread_info}  - {rag_id}")
                return False
                
    except Exception as e:
        print(f"{thread_info}❌ 初始化RAG系统时出错: {e}")
        import traceback
        traceback.print_exc()
        return False

def auto_select_rag_by_filename(filename: str) -> Optional[str]:
    """
    根据文件名自动选择合适的RAG系统
    
    Args:
        filename: 输入文件名
        
    Returns:
        str: 推荐的RAG系统名称，如果无法判断则返回None
    """
    filename_lower = filename.lower()
    
    # 根据文件名关键词判断应该使用哪个RAG系统
    if "金匮" in filename_lower or "jingui" in filename_lower:
        return "jingui_rag"
    elif "温病" in filename_lower or "wenbing" in filename_lower:
        return "wenbingxue_rag"
    elif "针灸" in filename_lower or "zhenjiu" in filename_lower:
        return "zhenjiu_rag"  # 如果有针灸相关的RAG
    elif "伤寒" in filename_lower or "shanghan" in filename_lower:
        return "shanghan_rag"  # 如果有伤寒论相关的RAG
    else:
        # 如果无法判断，返回None使用默认
        return None