"""
RAG系统配置文件
使用Python格式，便于动态配置和类型检查
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field

@dataclass
class RAGConfig:
    """RAG系统配置"""
    name: str
    description: str
    path: str
    model_path: str
    enabled: bool = True

@dataclass
class ModelConfig:
    """模型配置"""
    default: str
    alternatives: List[str] = field(default_factory=list)

@dataclass
class PathConfig:
    """路径配置"""
    pdf_input_dir: str = "../pdf"
    pdf_output_dir: str = "extracted_texts"
    pdf_temp_dir: str = "temp_pdf"
    rag_base_dir: str = "."
    models_dir: str = "../models"
    logs_dir: str = "logs"
    rag_log: str = "logs/rag.log"
    pdf_log: str = "logs/pdf_conversion.log"

@dataclass
class TextProcessingConfig:
    """文本处理配置"""
    similarity_threshold: float = 0.1
    min_chunk_size: int = 100
    max_chunk_size: int = 1000
    use_spacy: bool = True
    traditional_chunk_size: int = 500
    traditional_overlap: int = 50
    pdf_method: str = "auto"
    pdf_methods_priority: List[str] = field(default_factory=lambda: ["pdfplumber", "pymupdf", "pypdf2"])

@dataclass
class SearchConfig:
    """搜索配置"""
    default_top_k: int = 3
    similarity_threshold: float = 0.1
    max_results: int = 10

@dataclass
class SystemConfig:
    """系统配置"""
    debug: bool = False
    verbose: bool = True
    auto_backup: bool = True
    temp_cleanup: bool = True

# =============================================================================
# 主要配置区域 - 在这里修改您的配置
# =============================================================================

# 默认RAG系统设置
DEFAULT_RAG_NAME = "jingui_rag"
FALLBACK_RAG_NAME = "wenbingxue_rag"

# 可用的RAG系统列表
AVAILABLE_RAGS = {
    "jingui_rag": RAGConfig(
        name="金匮要略RAG",
        description="金匮要略医学文献检索系统",
        path="jingui_rag_from_pdf",
        model_path="../models/text2vec-base-chinese",
        enabled=True
    ),
    
    "wenbingxue_rag": RAGConfig(
        name="温病学RAG", 
        description="温病学医学文献检索系统",
        path="wenbingxue_rag",
        model_path="../models/text2vec-base-chinese",
        enabled=True
    )
}

# 模型配置
MODEL_CONFIG = ModelConfig(
    default="../models/text2vec-base-chinese",
    alternatives=[]
)

# 路径配置
PATH_CONFIG = PathConfig(
    pdf_input_dir="../pdf",
    pdf_output_dir="../pdf/extracted_texts",
    pdf_temp_dir="temp_pdf",
    rag_base_dir=".",
    models_dir="../models",
    logs_dir="logs",
    rag_log="logs/rag.log",
    pdf_log="logs/pdf_conversion.log"
)

# 文本处理配置
TEXT_PROCESSING_CONFIG = TextProcessingConfig(
    similarity_threshold=0.1,
    min_chunk_size=100,
    max_chunk_size=1000,
    use_spacy=True,
    traditional_chunk_size=500,
    traditional_overlap=50,
    pdf_method="auto",
    pdf_methods_priority=["pdfplumber", "pymupdf", "pypdf2"]
)

# 搜索配置
SEARCH_CONFIG = SearchConfig(
    default_top_k=3,
    similarity_threshold=0.1,
    max_results=10
)

# 系统配置
SYSTEM_CONFIG = SystemConfig(
    debug=False,
    verbose=True,
    auto_backup=True,
    temp_cleanup=True
)

# =============================================================================
# 配置管理器类
# =============================================================================

class ConfigManager:
    """配置管理器"""
    
    def __init__(self):
        self._setup_logging()
    
    def get_default_rag_name(self) -> str:
        """获取默认RAG名称"""
        return DEFAULT_RAG_NAME
    
    def get_fallback_rag_name(self) -> str:
        """获取备用RAG名称"""
        return FALLBACK_RAG_NAME
    
    def get_rag_config(self, rag_name: Optional[str] = None) -> RAGConfig:
        """获取RAG配置"""
        if rag_name is None:
            rag_name = DEFAULT_RAG_NAME
        
        if rag_name in AVAILABLE_RAGS and AVAILABLE_RAGS[rag_name].enabled:
            return AVAILABLE_RAGS[rag_name]
        else:
            print(f"⚠️  RAG '{rag_name}' 不存在或已禁用，使用默认RAG '{DEFAULT_RAG_NAME}'")
            return AVAILABLE_RAGS[DEFAULT_RAG_NAME]
    
    def get_model_config(self) -> ModelConfig:
        """获取模型配置"""
        return MODEL_CONFIG
    
    def get_path_config(self) -> PathConfig:
        """获取路径配置"""
        return PATH_CONFIG
    
    def get_text_processing_config(self) -> TextProcessingConfig:
        """获取文本处理配置"""
        return TEXT_PROCESSING_CONFIG
    
    def get_search_config(self) -> SearchConfig:
        """获取搜索配置"""
        return SEARCH_CONFIG
    
    def get_system_config(self) -> SystemConfig:
        """获取系统配置"""
        return SYSTEM_CONFIG
    
    def list_available_rags(self) -> List[str]:
        """列出可用的RAG系统"""
        return [rag_id for rag_id, config in AVAILABLE_RAGS.items() if config.enabled]
    
    def list_all_rags(self) -> List[str]:
        """列出所有RAG系统（包括禁用的）"""
        return list(AVAILABLE_RAGS.keys())
    
    def is_rag_enabled(self, rag_name: str) -> bool:
        """检查RAG是否启用"""
        return rag_name in AVAILABLE_RAGS and AVAILABLE_RAGS[rag_name].enabled
    
    def enable_rag(self, rag_name: str) -> bool:
        """启用RAG系统"""
        if rag_name in AVAILABLE_RAGS:
            AVAILABLE_RAGS[rag_name].enabled = True
            print(f"✅ 已启用RAG系统: {rag_name}")
            return True
        else:
            print(f"❌ RAG系统不存在: {rag_name}")
            return False
    
    def disable_rag(self, rag_name: str) -> bool:
        """禁用RAG系统"""
        if rag_name in AVAILABLE_RAGS:
            if rag_name == DEFAULT_RAG_NAME:
                print(f"⚠️  不能禁用默认RAG系统: {rag_name}")
                return False
            AVAILABLE_RAGS[rag_name].enabled = False
            print(f"🚫 已禁用RAG系统: {rag_name}")
            return True
        else:
            print(f"❌ RAG系统不存在: {rag_name}")
            return False
    
    def add_rag_config(self, rag_id: str, rag_config: RAGConfig) -> bool:
        """添加新的RAG配置"""
        if rag_id in AVAILABLE_RAGS:
            print(f"⚠️  RAG系统 '{rag_id}' 已存在，将覆盖")
        
        AVAILABLE_RAGS[rag_id] = rag_config
        print(f"✅ 已添加RAG配置: {rag_id}")
        return True
    
    def remove_rag_config(self, rag_id: str) -> bool:
        """移除RAG配置"""
        if rag_id == DEFAULT_RAG_NAME:
            print(f"⚠️  不能移除默认RAG系统: {rag_id}")
            return False
        
        if rag_id in AVAILABLE_RAGS:
            del AVAILABLE_RAGS[rag_id]
            print(f"✅ 已移除RAG配置: {rag_id}")
            return True
        else:
            print(f"❌ RAG系统不存在: {rag_id}")
            return False
    
    def get_rag_info(self, rag_name: Optional[str] = None) -> Dict[str, Any]:
        """获取RAG系统信息"""
        if rag_name is None:
            # 返回所有RAG信息
            return {
                rag_id: {
                    "name": config.name,
                    "description": config.description,
                    "path": config.path,
                    "model_path": config.model_path,
                    "enabled": config.enabled,
                    "exists": Path(config.path).exists()
                }
                for rag_id, config in AVAILABLE_RAGS.items()
            }
        else:
            # 返回指定RAG信息
            if rag_name in AVAILABLE_RAGS:
                config = AVAILABLE_RAGS[rag_name]
                return {
                    "name": config.name,
                    "description": config.description,
                    "path": config.path,
                    "model_path": config.model_path,
                    "enabled": config.enabled,
                    "exists": Path(config.path).exists()
                }
            else:
                return {}
    
    def _setup_logging(self):
        """设置日志"""
        log_dir = Path(PATH_CONFIG.logs_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # 配置日志格式
        logging.basicConfig(
            level=logging.DEBUG if SYSTEM_CONFIG.debug else logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(PATH_CONFIG.rag_log, encoding='utf-8'),
                logging.StreamHandler()
            ] if SYSTEM_CONFIG.verbose else [
                logging.FileHandler(PATH_CONFIG.rag_log, encoding='utf-8')
            ]
        )

# =============================================================================
# 全局配置管理器实例
# =============================================================================

config_manager = ConfigManager()

def get_config() -> ConfigManager:
    """获取配置管理器实例"""
    return config_manager

# =============================================================================
# 便捷函数
# =============================================================================

def get_default_rag() -> str:
    """获取默认RAG名称"""
    return DEFAULT_RAG_NAME

def set_default_rag(rag_name: str) -> bool:
    """设置默认RAG"""
    global DEFAULT_RAG_NAME
    if rag_name in AVAILABLE_RAGS and AVAILABLE_RAGS[rag_name].enabled:
        DEFAULT_RAG_NAME = rag_name
        print(f"✅ 默认RAG已设置为: {rag_name}")
        return True
    else:
        print(f"❌ RAG '{rag_name}' 不存在或已禁用")
        return False

def get_available_rags() -> List[str]:
    """获取可用的RAG系统列表"""
    return config_manager.list_available_rags()

def print_rag_status():
    """打印RAG系统状态"""
    print("\n📋 RAG系统状态:")
    print("=" * 60)
    
    for rag_id, config in AVAILABLE_RAGS.items():
        status_icon = "✅" if config.enabled else "🚫"
        default_mark = " (默认)" if rag_id == DEFAULT_RAG_NAME else ""
        exists_mark = " ✓" if Path(config.path).exists() else " ✗"
        
        print(f"{status_icon} {rag_id}{default_mark}{exists_mark}")
        print(f"   名称: {config.name}")
        print(f"   描述: {config.description}")
        print(f"   路径: {config.path}")
        print(f"   模型: {config.model_path}")
        print()

# =============================================================================
# 配置验证函数
# =============================================================================

def validate_config() -> bool:
    """验证配置"""
    print("🔍 验证配置...")
    
    all_valid = True
    
    # 检查默认RAG是否存在且启用
    if DEFAULT_RAG_NAME not in AVAILABLE_RAGS:
        print(f"❌ 默认RAG系统不存在: {DEFAULT_RAG_NAME}")
        all_valid = False
    elif not AVAILABLE_RAGS[DEFAULT_RAG_NAME].enabled:
        print(f"❌ 默认RAG系统已禁用: {DEFAULT_RAG_NAME}")
        all_valid = False
    
    # 检查备用RAG是否存在
    if FALLBACK_RAG_NAME not in AVAILABLE_RAGS:
        print(f"⚠️  备用RAG系统不存在: {FALLBACK_RAG_NAME}")
    
    # 检查路径是否存在
    paths_to_check = [
        ("模型目录", PATH_CONFIG.models_dir),
        ("PDF输入目录", PATH_CONFIG.pdf_input_dir),
    ]
    
    for name, path in paths_to_check:
        if not Path(path).exists():
            print(f"⚠️  {name}不存在: {path}")
    
    # 检查启用的RAG系统路径
    for rag_id, config in AVAILABLE_RAGS.items():
        if config.enabled and not Path(config.path).exists():
            print(f"⚠️  RAG系统路径不存在: {rag_id} -> {config.path}")
    
    if all_valid:
        print("✅ 配置验证通过")
    else:
        print("❌ 配置验证失败")
    
    return all_valid

if __name__ == "__main__":
    # 配置文件被直接运行时，显示配置状态
    print("🔧 RAG系统配置")
    print_rag_status()
    validate_config()