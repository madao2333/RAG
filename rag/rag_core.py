import re
import json
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any

class RAGCore:
    """RAG系统核心功能（通用）"""
    
    def __init__(self, model_name: str = '../models/text2vec-base-chinese'):
        # 🔥 默认使用本地下载的中文模型
        self.model_name = model_name
        self.model = None
    
    def create_embeddings(self, chunks: List[Dict[str, Any]]) -> tuple:
        """生成文本向量嵌入"""
        if self.model is None:
            try:
                print(f"正在加载模型: {self.model_name}")
                
                # 🔥 尝试加载本地模型
                if self.model_name.startswith('../models/'):
                    # 本地模型路径
                    model_path = Path(self.model_name)
                    if model_path.exists():
                        self.model = SentenceTransformer(str(model_path))
                        print(f"✓ 成功加载本地模型: {model_path}")
                    else:
                        print(f"✗ 本地模型不存在: {model_path}")
                else:
                    # 在线模型
                    self.model = SentenceTransformer(self.model_name)
                    print(f"✓ 成功加载在线模型: {self.model_name}")
                
                # 🔥 测试模型
                test_embedding = self.model.encode(["测试中文文本"])
                print(f"  模型测试通过，向量维度: {test_embedding.shape}")
                
            except Exception as e:
                print(f"✗ 模型加载失败: {e}")
        
        texts = [chunk['text'] for chunk in chunks]
        embeddings = self.model.encode(texts, show_progress_bar=True)
        
        return embeddings, self.model
    
    def build_vector_index(self, embeddings: np.ndarray) -> faiss.Index:
        """构建FAISS向量索引"""
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        faiss.normalize_L2(embeddings)
        index.add(embeddings.astype('float32'))
        return index
    
    def save_chunks(self, chunks: List[Dict[str, Any]], save_path: str):
        """保存chunks到文件"""
        save_file = Path(save_path)
        save_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_file, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
        
        print(f"Chunks已保存到: {save_file}")
        print(f"共保存 {len(chunks)} 个文档块")
    
    def load_chunks(self, load_path: str) -> List[Dict[str, Any]]:
        """从文件加载chunks"""
        load_file = Path(load_path)
        
        if not load_file.exists():
            raise FileNotFoundError(f"Chunks文件不存在: {load_file}")
        
        with open(load_file, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        
        print(f"从 {load_file} 加载了 {len(chunks)} 个文档块")
        return chunks
    
    def build_rag_system(self, text_file: str, save_dir: str,
                        save_chunks: bool = True) -> 'UniversalRAG':
        """构建RAG系统"""
        from text_processors import ProcessorFactory
        from rag_system import UniversalRAG
        
        print(f"构建RAG系统: {text_file}")
        
        # 1. 获取处理器
        processor = ProcessorFactory.get_processor(text_file)
        
        # 2. 清理文本
        print("1. 清理文本...")
        cleaned_text = self._clean_common_text(text_file)
        cleaned_text = processor.clean_text(cleaned_text)
        
        # 3. 文档分块
        print("2. 文档分块...")
        chunks = processor.chunk_text(cleaned_text)
        print(f"生成 {len(chunks)} 个文档块")
        
        if not chunks:
            raise ValueError("未能生成任何文档块，请检查文本格式")
        
        # 🔥 新增：保存chunks（可选）
        if save_chunks:
            chunks_file = Path(save_dir) / "chunks_only.json"
            self.save_chunks(chunks, chunks_file)
        
        # 4. 生成向量嵌入
        print("3. 生成向量嵌入...")
        embeddings, model = self.create_embeddings(chunks)
        
        # 5. 构建向量索引
        print("4. 构建向量索引...")
        index = self.build_vector_index(embeddings)
        
        # 6. 创建RAG系统
        print("5. 创建RAG系统...")
        rag_system = UniversalRAG(chunks, embeddings, index, model)
        
        # 7. 保存系统
        print("6. 保存系统...")
        rag_system.save(save_dir)
        
        return rag_system
    
    def build_rag_from_chunks(self, chunks_file: str, save_dir: str) -> 'UniversalRAG':
        """从已保存的chunks构建RAG系统"""
        from rag_system import UniversalRAG
        
        print(f"从chunks文件构建RAG系统: {chunks_file}")
        
        # 1. 加载chunks
        print("1. 加载chunks...")
        chunks = self.load_chunks(chunks_file)
        
        # 2. 生成向量嵌入
        print("2. 生成向量嵌入...")
        embeddings, model = self.create_embeddings(chunks)
        
        # 3. 构建向量索引
        print("3. 构建向量索引...")
        index = self.build_vector_index(embeddings)
        
        # 4. 创建RAG系统
        print("4. 创建RAG系统...")
        rag_system = UniversalRAG(chunks, embeddings, index, model)
        
        # 5. 保存系统
        print("5. 保存系统...")
        rag_system.save(save_dir)
        
        return rag_system
    
    def _clean_common_text(self, text_file: str) -> str:
        """通用文本清理"""
        with open(text_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 通用清理规则
        # 移除页面标记
        content = re.sub(r'--- 第 \d+ 页 ---', '', content)
        
        # 移除文件头信息
        content = re.sub(r'# 源文件:.*?\n# 文本长度:.*?\n\n', '', content, flags=re.DOTALL)
        
        return content.strip()