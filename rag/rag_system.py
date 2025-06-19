import json
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any

class UniversalRAG:
    """通用RAG检索系统（完全通用，不需要修改）"""
    
    def __init__(self, chunks, embeddings, index, model):
        self.chunks = chunks
        self.embeddings = embeddings
        self.index = index
        self.model = model
    
    def search(self, query: str, top_k: int = 5, filter_metadata: Dict[str, str] = None) -> List[Dict[str, Any]]:
        """检索相关文档"""
        # 编码查询
        query_embedding = self.model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # 搜索
        scores, indices = self.index.search(query_embedding.astype('float32'), top_k * 2)
        
        # 返回结果
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx != -1 and len(results) < top_k:
                chunk = self.chunks[idx]
                
                # 应用元数据过滤
                if filter_metadata:
                    if not all(chunk['metadata'].get(k) == v for k, v in filter_metadata.items()):
                        continue
                
                results.append({
                    'text': chunk['text'],
                    'metadata': chunk['metadata'],
                    'score': float(score),
                    'rank': len(results) + 1
                })
        
        return results
    
    def save(self, save_dir: str):
        """保存RAG系统"""
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)
        
        # 保存chunks
        with open(save_path / 'chunks.json', 'w', encoding='utf-8') as f:
            json.dump(self.chunks, f, ensure_ascii=False, indent=2)
        
        # 保存嵌入向量
        np.save(save_path / 'embeddings.npy', self.embeddings)
        
        # 保存FAISS索引
        faiss.write_index(self.index, str(save_path / 'index.faiss'))
        
        print(f"RAG系统已保存到 {save_dir}")
    
    @classmethod
    def load(cls, save_dir: str, model_name: str = '../models/text2vec-base-chinese'):
        """加载RAG系统"""
        save_path = Path(save_dir)
        
        # 加载chunks
        with open(save_path / 'chunks.json', 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        
        # 加载嵌入向量
        embeddings = np.load(save_path / 'embeddings.npy')
        
        # 加载FAISS索引
        index = faiss.read_index(str(save_path / 'index.faiss'))
        
        # 加载模型
        model = SentenceTransformer(model_name)
        
        return cls(chunks, embeddings, index, model)