import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
import os
from config import get_config

def load_rag_system(rag_name: Optional[str] = None):
    """根据配置加载RAG系统"""
    try:
        from rag_system import UniversalRAG
        
        config = get_config()
        
        # 如果没有指定RAG名称，使用配置中的默认值
        if rag_name is None:
            rag_name = config.get_default_rag_name()
        
        # 获取RAG配置
        rag_config = config.get_rag_config(rag_name)
        rag_path = Path(rag_config.path)
        
        if not rag_path.exists():
            print(f"⚠️  RAG系统目录不存在: {rag_config.path}")
            
            # 尝试使用备用RAG
            fallback_name = config.get_fallback_rag_name()
            if rag_name != fallback_name:
                print(f"尝试加载备用RAG: {fallback_name}")
                return load_rag_system(fallback_name)
            else:
                raise FileNotFoundError(f"RAG系统目录不存在: {rag_config.path}")
        
        print(f"🔍 正在加载RAG系统: {rag_config.name} ({rag_config.path})")
        rag_system = UniversalRAG.load(rag_config.path, rag_config.model_path)
        print(f"✅ RAG系统加载成功")
        
        return rag_system
        
    except Exception as e:
        print(f"❌ RAG系统加载失败: {e}")
        return None

def query_rag_for_question(question: str, rag_system=None, rag_name: Optional[str] = None, top_k: Optional[int] = None):
    """
    根据问题查询RAG系统，返回相关信息
    
    Args:
        question: 用户问题
        rag_system: RAG系统实例，如果为None则根据配置加载
        rag_name: 指定使用的RAG系统名称
        top_k: 返回的最相关结果数量
        
    Returns:
        str: 格式化的RAG查询结果文本
    """
    config = get_config()
    search_config = config.get_search_config()
    
    # 使用配置中的默认值
    if top_k is None:
        top_k = search_config.default_top_k
    
    if rag_system is None:
        rag_system = load_rag_system(rag_name)
        if rag_system is None:
            return ""
    
    try:
        # 执行RAG查询
        results = rag_system.search(question, top_k=top_k)
        
        if not results:
            print("✗ RAG查询未返回任何结果")
            return ""
        
        print(f"✓ RAG查询返回 {len(results)} 个结果")
        
        # 格式化RAG结果
        rag_context = ""
        valid_results_count = 0
        
        for i, result in enumerate(results, 1):
            score = result.get('score', 0)
            text = result.get('text', '')
            metadata = result.get('metadata', {})
            
            # 使用配置中的相似度阈值
            if score < search_config.similarity_threshold:
                print(f"  - 结果 {i}: 相关度 {score:.3f} (低于阈值{search_config.similarity_threshold}，跳过)")
                continue
            
            valid_results_count += 1
            print(f"  - 结果 {i}: 相关度 {score:.3f}")
            
            # 显示参考资料的前100个字符
            text_preview = text[:100] + "..." if len(text) > 100 else text
            print(f"    内容预览: {text_preview}")
            
            # 显示元数据信息
            if 'source' in metadata:
                print(f"    来源: {metadata['source']}")
            if 'section' in metadata:
                print(f"    章节: {metadata['section']}")
            print()  # 空行分隔
            
            # 添加元数据信息
            if 'source' in metadata:
                rag_context += f"来源：{metadata['source']}\n"
            if 'section' in metadata:
                rag_context += f"章节：{metadata['section']}\n"
            
            # 添加内容
            rag_context += f"内容：{text}\n\n"
        
        if valid_results_count == 0:
            print(f"✗ 所有RAG结果的相关度都低于阈值{search_config.similarity_threshold}")
            return ""
        
        print(f"✓ 共有 {valid_results_count} 个有效的参考资料被使用")
        return rag_context.strip()
        
    except Exception as e:
        print(f"RAG查询出错: {e}")
        return ""

# 全局RAG系统实例，避免重复加载
_global_rag_system = None
_global_rag_name = None

def get_rag_system(rag_name: Optional[str] = None):
    """获取全局RAG系统实例"""
    global _global_rag_system, _global_rag_name
    
    config = get_config()
    
    # 如果没有指定名称，让用户选择
    if rag_name is None:
        rag_name = prompt_user_to_select_rag()
        if rag_name is None:  # 用户取消选择
            print("❌ 未选择RAG系统")
            return None
    
    # 如果RAG名称改变了，重新加载
    if _global_rag_system is None or _global_rag_name != rag_name:
        print(f"🔄 切换到RAG系统: {rag_name}")
        _global_rag_system = load_rag_system(rag_name)
        _global_rag_name = rag_name
    
    return _global_rag_system

def prompt_user_to_select_rag():
    """提示用户选择RAG系统"""
    config = get_config()
    available_rags = config.list_available_rags()
    
    if not available_rags:
        print("❌ 没有可用的RAG系统")
        return None
    
    # 显示可用的RAG系统
    print("\n📋 可用的RAG系统:")
    print("=" * 50)
    
    for i, rag_id in enumerate(available_rags, 1):
        rag_config = config.get_rag_config(rag_id)
        default_mark = " (默认)" if rag_id == config.get_default_rag_name() else ""
        exists_mark = " ✓" if Path(rag_config.path).exists() else " ✗"
        
        print(f"{i}. {rag_id}{default_mark}{exists_mark}")
        print(f"   名称: {rag_config.name}")
        print(f"   描述: {rag_config.description}")
        print(f"   路径: {rag_config.path}")
        print()
    
    # 添加默认选项和退出选项
    default_index = 0
    default_rag = config.get_default_rag_name()
    if default_rag in available_rags:
        default_index = available_rags.index(default_rag) + 1
    
    print(f"d. 使用默认RAG系统 ({default_rag})")
    print("q. 退出/取消")
    print("=" * 50)
    
    while True:
        try:
            choice = input(f"请选择RAG系统 [1-{len(available_rags)}/d/q]: ").strip().lower()
            
            if choice == 'q':
                return None
            elif choice == 'd' or choice == '':
                if default_rag in available_rags:
                    print(f"✅ 选择了默认RAG系统: {default_rag}")
                    return default_rag
                else:
                    print(f"❌ 默认RAG系统不可用: {default_rag}")
                    continue
            elif choice.isdigit():
                index = int(choice) - 1
                if 0 <= index < len(available_rags):
                    selected_rag = available_rags[index]
                    print(f"✅ 选择了RAG系统: {selected_rag}")
                    return selected_rag
                else:
                    print(f"❌ 无效选择，请输入 1-{len(available_rags)} 之间的数字")
            else:
                print("❌ 无效输入，请输入数字、'd'(默认) 或 'q'(退出)")
                
        except KeyboardInterrupt:
            print("\n❌ 用户中断")
            return None
        except Exception as e:
            print(f"❌ 输入错误: {e}")

# 可选：添加一个批量选择函数（用于非交互式环境）
def get_rag_system_non_interactive(rag_name: Optional[str] = None, use_default: bool = True):
    """非交互式获取RAG系统实例"""
    global _global_rag_system, _global_rag_name
    
    config = get_config()
    
    # 如果没有指定名称，根据参数决定行为
    if rag_name is None:
        if use_default:
            rag_name = config.get_default_rag_name()
            print(f"🔄 使用默认RAG系统: {rag_name}")
        else:
            print("❌ 未指定RAG系统且不使用默认值")
            return None
    
    # 如果RAG名称改变了，重新加载
    if _global_rag_system is None or _global_rag_name != rag_name:
        print(f"🔄 切换到RAG系统: {rag_name}")
        _global_rag_system = load_rag_system(rag_name)
        _global_rag_name = rag_name
    
    return _global_rag_system

def simple_rag_query(question: str, rag_name: Optional[str] = None, top_k: Optional[int] = None, interactive: bool = False):
    """
    简化的RAG查询函数，支持动态指定RAG系统
    
    Args:
        question: 用户问题
        rag_name: RAG系统名称，为None时根据interactive参数决定行为
        top_k: 返回的最相关结果数量，为None时使用配置中的默认值
        interactive: 是否启用交互式RAG选择，False时使用默认RAG
        
    Returns:
        str: 格式化的RAG查询结果文本
    """
    if interactive and rag_name is None:
        rag_system = get_rag_system(rag_name)  # 会提示用户选择
    else:
        rag_system = get_rag_system_non_interactive(rag_name)  # 使用默认或指定的RAG
    
    if rag_system is None:
        return ""
    
    return query_rag_for_question(question, rag_system, rag_name, top_k)

def switch_rag_system(rag_name: str):
    """切换RAG系统"""
    global _global_rag_system, _global_rag_name
    
    config = get_config()
    
    if rag_name in config.list_available_rags():
        _global_rag_system = None  # 强制重新加载
        _global_rag_name = None
        get_rag_system(rag_name)  # 加载新的RAG系统
        print(f"✅ 已切换到RAG系统: {rag_name}")
        return True
    else:
        print(f"❌ RAG系统 '{rag_name}' 不存在或已禁用")
        print(f"可用的RAG系统: {config.list_available_rags()}")
        return False

def list_available_rags():
    """列出可用的RAG系统"""
    config = get_config()
    available_rags = config.list_available_rags()
    
    print("📋 可用的RAG系统:")
    for rag_id in available_rags:
        rag_config = config.get_rag_config(rag_id)
        default_mark = " (默认)" if rag_id == config.get_default_rag_name() else ""
        print(f"  ✅ {rag_id}{default_mark}: {rag_config.name}")
        print(f"      描述: {rag_config.description}")
        print(f"      路径: {rag_config.path}")
    
    return available_rags