import json
import time
import threading
import os
import gc
import torch
from inference_LoRA import initialize_model, process_items_batch, save_results_to_file, initialize_model_pretrain

# --- 配置 ---
INPUT_JSON_FILE = "" 
OUTPUT_JSON_FILE = ""
MODEL_PATH = ""  # 基础模型路径
LORA_PATH = ""  # LoRA适配器路径

# 新增：RAG系统配置
RAG_SYSTEM_NAME = "jingui_rag"  # 指定使用的RAG系统，None表示自动选择或使用默认
AUTO_SELECT_RAG = False  # 是否根据文件名自动选择RAG系统

# --- 全局变量 ---
MAX_THREADS = 1  # 线程数
output_file_lock = threading.Lock()  # 输出文件锁
processed_results = []  # 处理结果
results_lock = threading.Lock()  # 结果列表锁
BATCH_SIZE = 1  # 每个线程一次性处理的批次大小

def process_items_thread(items, thread_id):
    """
    线程工作函数：每个线程加载自己的模型实例和RAG系统
    
    Args:
        items: 要处理的项目列表
        thread_id: 线程ID
    """
    print(f"线程 {thread_id}: 开始处理 {len(items)} 个项目")
    
    # 1. 初始化RAG系统
    print(f"线程 {thread_id}: 初始化RAG系统...")
    rag_name = RAG_SYSTEM_NAME
    
    # 如果启用自动选择且没有指定RAG名称，根据文件名自动选择
    if AUTO_SELECT_RAG and rag_name is None:
        from inference_LoRA import auto_select_rag_by_filename
        rag_name = auto_select_rag_by_filename(INPUT_JSON_FILE)
        if rag_name:
            print(f"线程 {thread_id}: 根据文件名自动选择RAG系统: {rag_name}")
        else:
            print(f"线程 {thread_id}: 无法根据文件名判断RAG系统，将使用默认")
    
    # 初始化RAG系统
    from inference_LoRA import initialize_rag
    rag_success = initialize_rag(rag_name, thread_id)
    
    if not rag_success:
        print(f"线程 {thread_id}: RAG系统初始化失败，但继续执行（将不使用RAG功能）")
        # 可以选择继续执行或退出
        # return  # 如果RAG是必需的，可以取消注释这行来退出线程
    
    # 2. 加载LoRA模型
    print(f"线程 {thread_id}: 加载LoRA模型...")
    from inference_LoRA import initialize_model
    tokenizer, model = initialize_model(MODEL_PATH, LORA_PATH, thread_id)
    
    # 3. 加载预训练模型（用于总结）
    print(f"线程 {thread_id}: 加载预训练模型...")
    from inference_LoRA import initialize_model_pretrain
    tokenizer_pre, model_pre = initialize_model_pretrain(MODEL_PATH, thread_id)
    
    if tokenizer is None or model is None:
        print(f"线程 {thread_id}: LoRA模型初始化失败，退出线程")
        return
    
    if tokenizer_pre is None or model_pre is None:
        print(f"线程 {thread_id}: 预训练模型初始化失败，将跳过总结功能")
        tokenizer_pre, model_pre = None, None
    
    try:
        # 4. 分批处理项目
        for i in range(0, len(items), BATCH_SIZE):
            batch = items[i:i + BATCH_SIZE]
            print(f"线程 {thread_id}: 处理批次 {i//BATCH_SIZE + 1}/{(len(items)-1)//BATCH_SIZE + 1}")
            
            from inference_LoRA import process_items_batch
            results = process_items_batch(
                batch, 
                tokenizer, model,           # LoRA模型
                tokenizer_pre, model_pre,   # 预训练模型
                thread_id
            )
            
            # 保存批次结果
            with results_lock:
                processed_results.extend(results)
            
            # 定期保存到文件
            print(f"线程 {thread_id}: 批次 {i//BATCH_SIZE + 1}/{(len(items)-1)//BATCH_SIZE + 1} 处理完成，保存结果...")
            from inference_LoRA import save_results_to_file
            save_results_to_file(results, OUTPUT_JSON_FILE, output_file_lock)
    
    except Exception as e:
        print(f"线程 {thread_id}: 处理过程中出错: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 5. 清理模型资源
        print(f"线程 {thread_id}: 清理模型资源...")
        try:
            del model
            del tokenizer
            if model_pre is not None:
                del model_pre
            if tokenizer_pre is not None:
                del tokenizer_pre
        except:
            pass
        torch.cuda.empty_cache()
        gc.collect()
        print(f"线程 {thread_id}: 资源已释放")

def main():
    print("LoRA模型多线程处理开始...")
    print(f"基础模型路径: {MODEL_PATH}")
    print(f"LoRA适配器路径: {LORA_PATH}")
    
    # 显示RAG配置信息
    if RAG_SYSTEM_NAME:
        print(f"指定RAG系统: {RAG_SYSTEM_NAME}")
    elif AUTO_SELECT_RAG:
        print("启用RAG系统自动选择")
    else:
        print("使用默认RAG系统")
    
    # 读取输入文件
    try:
        with open(INPUT_JSON_FILE, "r", encoding="utf-8") as f:
            all_input_items = json.load(f)
    except Exception as e:
        print(f"读取输入文件失败: {e}")
        return
    
    # 读取输出文件以确定已处理项目
    existing_ids = set()
    if os.path.exists(OUTPUT_JSON_FILE):
        try:
            with open(OUTPUT_JSON_FILE, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    existing_data = json.loads(content)
                    for item in existing_data:
                        if isinstance(item, dict) and 'id' in item:
                            existing_ids.add(item['id'])
        except Exception as e:
            print(f"读取现有输出文件失败: {e}")
    
    # 筛选未处理项目
    items_to_process = [item for item in all_input_items 
                        if isinstance(item, dict) and 'id' in item and item['id'] not in existing_ids]
    print(f"找到 {len(items_to_process)} 个未处理项目")
    
    if not items_to_process:
        print("没有需要处理的项目")
        return
    
    # 为每个线程分配项目
    thread_items = [[] for _ in range(MAX_THREADS)]
    for i, item in enumerate(items_to_process):
        thread_items[i % MAX_THREADS].append(item)
    
    # 创建并启动线程
    threads = []
    for i in range(MAX_THREADS):
        if thread_items[i]:  # 只为有任务的线程创建线程
            thread = threading.Thread(
                target=process_items_thread,
                args=(thread_items[i], i)
            )
            threads.append(thread)
            thread.start()
            print(f"线程 {i} 已启动，分配了 {len(thread_items[i])} 个项目")
            time.sleep(10)  # 增加线程启动间隔，避免模型加载冲突
    
    # 等待所有线程完成
    for i, thread in enumerate(threads):
        thread.join()
        print(f"线程 {i} 已完成")
    
    # 最终保存所有结果
    print("所有线程已完成，保存最终结果...")
    from inference_LoRA import save_results_to_file
    save_results_to_file(processed_results, OUTPUT_JSON_FILE)
    
    print("处理完成")

if __name__ == "__main__":
    main()