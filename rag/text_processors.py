import re
import numpy as np
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
import jieba
import jieba.posseg as pseg
from abc import ABC, abstractmethod

class BaseTextProcessor(ABC):
    """文本处理器基类"""
    
    @abstractmethod
    def clean_text(self, content: str) -> str:
        """清理文本"""
        pass
    
    @abstractmethod
    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[Dict[str, Any]]:
        """分块文本"""
        pass

class GenericProcessor(BaseTextProcessor):
    """通用文本处理器 - 传统分块方法"""
    
    def clean_text(self, content: str) -> str:
        """清理文本"""
        # 移除常见的无关内容
        content = re.sub(r'ISBN.*?\n', '', content)
        content = re.sub(r'版权所有.*?\n', '', content)
        # 清理多余的空白
        content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)
        content = re.sub(r'[ \t]+', ' ', content)
        return content
    
    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[Dict[str, Any]]:
        """传统分块方法"""
        chunks = []
        # 按段落分割
        paragraphs = re.split(r'\n\s*\n', text)
        
        current_chunk = ""
        current_size = 0
        chunk_index = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
                
            para_len = len(para)
            
            # 如果当前段落加上已有内容超过块大小，则保存当前块
            if current_size + para_len > chunk_size and current_size > 0:
                chunks.append({
                    'text': current_chunk,
                    'metadata': {
                        'source': '医学文献',
                        'chunk_index': chunk_index,
                        'method': 'traditional'
                    }
                })
                chunk_index += 1
                
                # 保留重叠部分
                overlap_text = current_chunk[-overlap:] if overlap > 0 else ""
                current_chunk = overlap_text + para
                current_size = len(current_chunk)
            else:
                # 否则将段落添加到当前块
                if current_size > 0:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para
                current_size = len(current_chunk)
        
        # 添加最后一个块
        if current_size > 0:
            chunks.append({
                'text': current_chunk,
                'metadata': {
                    'source': '医学文献',
                    'chunk_index': chunk_index,
                    'method': 'traditional'
                }
            })
            
        return chunks

class SemanticGenericProcessor(BaseTextProcessor):
    """支持语义分块的通用处理器"""
    
    def __init__(self, model_path: str = "../models/text2vec-base-chinese"):
        """
        初始化语义处理器
        
        Args:
            model_path: 向量模型路径
        """
        try:
            self.embedding_model = SentenceTransformer(model_path)
        except Exception as e:
            print(f"加载模型失败: {e}, 将使用传统分块")
            self.embedding_model = None
            
        self.similarity_threshold = 0.6  # 🔥 降低阈值，避免过度合并
        self.min_chunk_size = 100  # 🔥 降低最小块大小，减少内容丢失
        self.max_chunk_size = 1000
        
    def clean_text(self, content: str) -> str:
        """清理文本"""
        # 移除常见的无关内容
        content = re.sub(r'ISBN.*?\n', '', content)
        content = re.sub(r'版权所有.*?\n', '', content)
        content = re.sub(r'目\s*录', '目录', content)
        # 清理多余的空白
        content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)
        content = re.sub(r'[ \t]+', ' ', content)
        return content
    
    def chunk_text(self, text: str, chunk_size: int = 800, overlap: int = 100) -> List[Dict[str, Any]]:
        """
        语义分块主函数 - 移除结构化分块，直接使用语义分块
        
        Args:
            text: 输入文本
            chunk_size: 目标分块大小
            overlap: 重叠大小（语义分块中作为参考）
            
        Returns:
            List[Dict]: 分块结果
        """
        self.max_chunk_size = chunk_size
        
        # 🔥 如果模型加载失败，使用传统分块
        if self.embedding_model is None:
            print("模型未加载，使用传统分块")
            return self._fallback_traditional_chunking(text)
        
        try:
            # 🔥 直接进行语义分块，不再使用结构化分块
            print("开始语义分块...")
            semantic_chunks = self._semantic_chunking(text)
            
            # 验证分块结果
            if not semantic_chunks:
                print("语义分块失败，使用传统分块")
                return self._fallback_traditional_chunking(text)
            
            # 🔥 验证内容完整性
            if self._verify_content_completeness(text, semantic_chunks):
                print(f"✅ 语义分块成功，生成 {len(semantic_chunks)} 个chunks")
                return semantic_chunks
            else:
                print("⚠️ 语义分块内容不完整，使用传统分块")
                return self._fallback_traditional_chunking(text)
                
        except Exception as e:
            print(f"语义分块失败: {e}, 使用传统分块")
            return self._fallback_traditional_chunking(text)

    def _verify_content_completeness(self, original_text: str, chunks: List[Dict[str, Any]]) -> bool:
        """验证分块后内容的完整性"""
        try:
            # 合并所有chunks的文本
            combined_text = ''.join(chunk['text'] for chunk in chunks)
            
            # 移除空白字符后比较
            original_clean = re.sub(r'\s+', '', original_text)
            combined_clean = re.sub(r'\s+', '', combined_text)
            
            # 计算保留率
            retention_rate = len(combined_clean) / len(original_clean) if original_clean else 0
            
            print(f"内容保留率: {retention_rate:.2%}")
            
            if retention_rate < 0.95:  # 如果丢失超过5%
                print(f"⚠️ 内容丢失过多，保留率仅 {retention_rate:.2%}")
                return False
            
            return True
            
        except Exception as e:
            print(f"完整性验证失败: {e}")
            return False

    def _fallback_traditional_chunking(self, text: str) -> List[Dict[str, Any]]:
        """备用传统分块方法"""
        chunks = []
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        if not paragraphs:
            # 如果没有段落，按固定长度分块
            for i in range(0, len(text), self.max_chunk_size - 100):
                chunk_text = text[i:i + self.max_chunk_size]
                if chunk_text.strip():
                    chunks.append({
                        'text': chunk_text.strip(),
                        'metadata': {
                            'source': '医学文献',
                            'type': 'semantic',
                            'chunk_index': len(chunks),
                            'total_chunks': 'unknown',
                            'semantic_method': 'fallback_fixed'
                        }
                    })
        else:
            # 按段落分块
            current_chunk = ""
            chunk_index = 0
            
            for para in paragraphs:
                if len(current_chunk + "\n\n" + para) > self.max_chunk_size and current_chunk:
                    chunks.append({
                        'text': current_chunk.strip(),
                        'metadata': {
                            'source': '医学文献',
                            'type': 'semantic',
                            'chunk_index': chunk_index,
                            'total_chunks': 'unknown',
                            'semantic_method': 'fallback_paragraph'
                        }
                    })
                    chunk_index += 1
                    current_chunk = para
                else:
                    if current_chunk:
                        current_chunk += "\n\n" + para
                    else:
                        current_chunk = para
            
            # 添加最后一个chunk
            if current_chunk:
                chunks.append({
                    'text': current_chunk.strip(),
                    'metadata': {
                        'source': '医学文献',
                        'type': 'semantic',
                        'chunk_index': chunk_index,
                        'total_chunks': chunk_index + 1,
                        'semantic_method': 'fallback_paragraph'
                    }
                })
        
        # 更新总数
        for chunk in chunks:
            chunk['metadata']['total_chunks'] = len(chunks)
        
        print(f"传统备用分块生成 {len(chunks)} 个chunks")
        return chunks
    
    def _structural_chunking(self, text: str) -> List[Dict[str, Any]]:
        """结构化分块 - 先按章节等结构分割"""
        chunks = []
        
        # 按章节分割
        chapter_pattern = r'第[一二三四五六七八九十\d]+[章节篇][^\n]*'
        chapter_matches = list(re.finditer(chapter_pattern, text))
        
        if len(chapter_matches) < 2:  # 章节太少，不进行结构化分块
            return []
        
        for i, match in enumerate(chapter_matches):
            chapter_start = match.start()
            chapter_end = chapter_matches[i + 1].start() if i + 1 < len(chapter_matches) else len(text)
            
            chapter_text = text[chapter_start:chapter_end].strip()
            chapter_title = match.group().strip()
            
            # 提取章节信息
            chapter_match = re.match(r'第([一二三四五六七八九十\d]+)[章节篇]\s*(.+)', chapter_title)
            if chapter_match:
                chapter_num = chapter_match.group(1)
                chapter_name = chapter_match.group(2).strip()
            else:
                chapter_num = str(i + 1)
                chapter_name = chapter_title
            
            chunks.append({
                'text': chapter_text,
                'metadata': {
                    'source': '医学文献',
                    'chapter_num': chapter_num,
                    'chapter_title': chapter_name,
                    'type': 'structural_chapter'
                }
            })
        
        return chunks
    
    def _semantic_chunking(self, text: str, base_metadata: Dict = None) -> List[Dict[str, Any]]:
        """
        语义分块核心方法 - 句子级分块（避免信息丢失版）
        
        Args:
            text: 输入文本
            base_metadata: 基础元数据
            
        Returns:
            List[Dict]: 语义分块结果
        """
        if base_metadata is None:
            base_metadata = {'source': '医学文献', 'type': 'semantic'}
        
        # 🔥 改回句子级分块，但增强分割逻辑
        sentences = self._safe_split_into_sentences(text)
        
        if len(sentences) <= 3:  # 句子太少，直接返回
            return [{
                'text': text,
                'metadata': {
                    **base_metadata,
                    'chunk_index': 0,
                    'total_chunks': 1,
                    'semantic_method': 'too_short'
                }
            }]
        
        print(f"总共 {len(sentences)} 个句子")
        
        # 🔥 验证句子分割的完整性
        combined_sentences = ''.join(sentences)
        original_no_space = re.sub(r'\s+', '', text)
        combined_no_space = re.sub(r'\s+', '', combined_sentences)
        
        if len(combined_no_space) < len(original_no_space) * 0.95:  # 丢失超过5%
            print(f"⚠️ 句子分割可能丢失内容，回退到段落分块")
            return self._simple_paragraph_grouping(text.split('\n\n'), base_metadata)
        
        # 使用句子进行语义分析
        if len(sentences) > 30:  # 句子较多时才使用语义聚类
            try:
                print(f"计算 {len(sentences)} 个句子的向量...")
                sentence_embeddings = self.embedding_model.encode(sentences)
                
                # 🔥 安全的语义聚类
                semantic_groups = self._safe_semantic_clustering(sentences, sentence_embeddings)
                
                # 🔥 验证聚类结果的完整性
                if not self._validate_clustering_completeness(semantic_groups, len(sentences)):
                    print("聚类结果不完整，使用顺序分组")
                    semantic_groups = self._sequential_sentence_grouping(sentences)
                
                # 生成分块
                chunks = self._create_sentence_chunks(semantic_groups, sentences, base_metadata)
                
                return chunks
                
            except Exception as e:
                print(f"句子语义聚类失败: {e}, 使用顺序分组")
                return self._sequential_sentence_chunking(sentences, base_metadata)
        else:
            # 句子较少时直接顺序分组
            return self._sequential_sentence_chunking(sentences, base_metadata)
    
    def _safe_split_into_sentences(self, text, language='zh'):
        """
        安全地将文本分割成句子，优先使用中文友好的库
        """
        if not text or not text.strip():
            print("⚠️  输入文本为空或只包含空白字符")
            return []
        
        text = text.strip()
        print(f"\n🔍 开始分割文本 (语言: {language})")
        print(f"📝 原始文本长度: {len(text)} 字符")
        print(f"📄 原始文本前100字符: {text[:100]}{'...' if len(text) > 100 else ''}")
        
        sentences = []
        
        try:
            if language == 'zh':
                # 🔥 优先尝试中文友好的分割方法
                
                # 方法1: spaCy中文模型 (最推荐)
                try:
                    import spacy
                    nlp = spacy.load("zh_core_web_sm")
                    doc = nlp(text)
                    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
                    print("✅ 使用 spaCy 中文模型分割完成")
                    
                except (ImportError, OSError):
                    print("⚠️  spaCy中文模型未安装")
        except Exception as e:
            print(f"⚠️ 文本分割过程中发生错误: {e}")
            # 使用最基础的分割方法作为最后备选
            sentences = [s.strip() for s in text.split('\n') if s.strip()]
            if not sentences:
                sentences = [text]
            print("🔄 使用基础换行符分割作为备选方案")
        
        # 清理和过滤句子
        print("\n🧹 开始清理句子...")
        original_count = len(sentences)
        
        cleaned_sentences = []
        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if sentence and len(sentence) > 1:  # 过滤掉太短的句子
                cleaned_sentences.append(sentence)
                print(f"  ✓ 句子 {i+1}: {sentence[:50]}{'...' if len(sentence) > 50 else ''}")
            else:
                print(f"  ✗ 跳过句子 {i+1}: '{sentence}' (太短或为空)")
        
        sentences = cleaned_sentences
        
        # 如果没有句子，尝试更宽松的分割
        if not sentences:
            print("⚠️  没有找到有效句子，尝试更宽松的分割...")
            # 按换行符分割
            sentences = [s.strip() for s in text.split('\n') if s.strip()]
            if sentences:
                print(f"✅ 按换行符分割得到 {len(sentences)} 个句子")
            else:
                # 最后手段：整个文本作为一个句子
                sentences = [text]
                print("⚠️  使用整个文本作为单个句子")
        
        print(f"\n📊 分割统计:")
        print(f"  - 清理后句子数: {len(sentences)}")
        
        print(f"\n🎯 最终结果: 共 {len(sentences)} 个句子")
        for i, sentence in enumerate(sentences[:5]):  # 只显示前5个句子
            print(f"  📝 句子 {i+1} ({len(sentence)} 字符): {sentence}")
        
        if len(sentences) > 5:
            print(f"  ... 省略其余 {len(sentences) - 5} 个句子")
        
        print("=" * 60)
        return sentences

    def _safe_semantic_clustering(self, sentences: List[str], embeddings: np.ndarray) -> List[List[int]]:
        """安全的语义聚类 - 确保所有句子都被包含"""
        try:
            if len(sentences) <= 3:
                return [list(range(len(sentences)))]
            
            # 🔥 使用滑动窗口的贪心聚类，确保所有句子都被处理
            groups = []
            used = [False] * len(sentences)  # 使用布尔数组跟踪
            
            for i in range(len(sentences)):
                if used[i]:
                    continue
                
                current_group = [i]
                used[i] = True
                current_text_length = len(sentences[i])
                
                # 🔥 向前和向后搜索相似句子
                search_range = min(10, len(sentences))  # 限制搜索范围
                
                for offset in range(1, search_range):
                    # 向后搜索
                    j = i + offset
                    if j < len(sentences) and not used[j]:
                        similarity = cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]
                        if (similarity > self.similarity_threshold and 
                            current_text_length + len(sentences[j]) <= self.max_chunk_size):
                            current_group.append(j)
                            used[j] = True
                            current_text_length += len(sentences[j])
                    
                    # 向前搜索（如果i>0）
                    k = i - offset
                    if k >= 0 and not used[k]:
                        similarity = cosine_similarity([embeddings[i]], [embeddings[k]])[0][0]
                        if (similarity > self.similarity_threshold and 
                            current_text_length + len(sentences[k]) <= self.max_chunk_size):
                            current_group.append(k)
                            used[k] = True
                            current_text_length += len(sentences[k])
                
                if current_group:
                    groups.append(sorted(current_group))  # 保持句子顺序
            
            # 🔥 检查是否有遗漏的句子
            all_used_indices = set()
            for group in groups:
                all_used_indices.update(group)
            
            missing_indices = set(range(len(sentences))) - all_used_indices
            if missing_indices:
                print(f"发现 {len(missing_indices)} 个未分组的句子，添加到独立组")
                for idx in sorted(missing_indices):
                    groups.append([idx])
            
            return groups
            
        except Exception as e:
            print(f"语义聚类失败: {e}, 使用顺序分组")
            return self._sequential_sentence_grouping(sentences)

    def _sequential_sentence_grouping(self, sentences: List[str]) -> List[List[int]]:
        """顺序句子分组 - 确保不丢失任何句子"""
        groups = []
        current_group = []
        current_length = 0
        
        for i, sentence in enumerate(sentences):
            sentence_length = len(sentence)
            
            if current_length + sentence_length > self.max_chunk_size and current_group:
                groups.append(current_group)
                current_group = [i]
                current_length = sentence_length
            else:
                current_group.append(i)
                current_length += sentence_length
        
        if current_group:
            groups.append(current_group)
        
        return groups

    def _validate_clustering_completeness(self, groups: List[List[int]], total_sentences: int) -> bool:
        """验证聚类结果的完整性"""
        all_indices = set()
        for group in groups:
            all_indices.update(group)
        
        return len(all_indices) == total_sentences

    def _sequential_sentence_chunking(self, sentences: List[str], base_metadata: Dict) -> List[Dict[str, Any]]:
        """顺序句子分块 - 确保不丢失内容"""
        chunks = []
        current_chunk_sentences = []
        current_length = 0
        chunk_index = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            if current_length + sentence_length > self.max_chunk_size and current_chunk_sentences:
                # 保存当前chunk
                chunk_text = ''.join(current_chunk_sentences)
                keywords = self._simple_extract_keywords(chunk_text)
                
                chunks.append({
                    'text': chunk_text.strip(),
                    'metadata': {
                        **base_metadata,
                        'chunk_index': chunk_index,
                        'total_chunks': 'unknown',
                        'semantic_method': 'sequential_sentences',
                        'sentence_count': len(current_chunk_sentences),
                        'keywords': keywords
                    }
                })
                
                chunk_index += 1
                current_chunk_sentences = [sentence]
                current_length = sentence_length
            else:
                current_chunk_sentences.append(sentence)
                current_length += sentence_length
        
        # 添加最后一个chunk
        if current_chunk_sentences:
            chunk_text = ''.join(current_chunk_sentences)
            keywords = self._simple_extract_keywords(chunk_text)
            
            chunks.append({
                'text': chunk_text.strip(),
                'metadata': {
                    **base_metadata,
                    'chunk_index': chunk_index,
                    'total_chunks': chunk_index + 1,
                    'semantic_method': 'sequential_sentences',
                    'sentence_count': len(current_chunk_sentences),
                    'keywords': keywords
                }
            })
        
        # 更新总数
        for chunk in chunks:
            chunk['metadata']['total_chunks'] = len(chunks)
        
        print(f"顺序句子分块生成 {len(chunks)} 个chunks")
        return chunks

    def _create_sentence_chunks(self, groups: List[List[int]], sentences: List[str], base_metadata: Dict) -> List[Dict[str, Any]]:
        """从句子组创建最终chunks - 确保不丢失内容"""
        chunks = []
        
        for i, group in enumerate(groups):
            if not group:  # 跳过空组
                continue
            
            # 🔥 按索引顺序组合句子，保持原有顺序
            group_sentences = [sentences[idx] for idx in sorted(group)]
            chunk_text = ''.join(group_sentences)  # 不添加额外标点，保持原样
            
            # 确保内容不为空
            if not chunk_text.strip():
                continue
            
            keywords = self._simple_extract_keywords(chunk_text)
            
            chunks.append({
                'text': chunk_text.strip(),
                'metadata': {
                    **base_metadata,
                    'chunk_index': i,
                    'total_chunks': len(groups),
                    'semantic_method': 'sentence_clustering',
                    'sentence_count': len(group),
                    'keywords': keywords,
                    'semantic_coherence': self._safe_calculate_coherence(group_sentences)
                }
            })
        
        return chunks

    def _safe_calculate_coherence(self, sentences: List[str]) -> float:
        """安全计算语义连贯性"""
        try:
            if len(sentences) <= 1:
                return 1.0
            
            if self.embedding_model is None:
                return 1.0
            
            # 计算句子向量
            embeddings = self.embedding_model.encode(sentences)
            
            # 计算相邻句子的平均相似度
            similarities = []
            for i in range(len(embeddings) - 1):
                similarity = cosine_similarity([embeddings[i]], [embeddings[i + 1]])[0][0]
                similarities.append(float(similarity))  # 🔥 立即转换
            
            return float(np.mean(similarities)) if similarities else 1.0
            
        except Exception as e:
            print(f"连贯性计算失败: {e}")
            return 1.0
    
    # 🔥 保留其他原有方法但简化逻辑
    def _semantic_clustering(self, sentences: List[str], embeddings: np.ndarray) -> List[List[int]]:
        """语义聚类 - 简化版"""
        try:
            if len(sentences) <= 3:
                return [list(range(len(sentences)))]
            
            # 使用更简单的聚类策略
            groups = []
            used = set()
            
            for i in range(len(sentences)):
                if i in used:
                    continue
                
                current_group = [i]
                used.add(i)
                
                # 只与相邻的句子比较
                for j in range(i + 1, min(i + 5, len(sentences))):  # 限制搜索范围
                    if j in used:
                        continue
                        
                    similarity = cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]
                    if similarity > self.similarity_threshold:
                        current_group.append(j)
                        used.add(j)
                
                groups.append(current_group)
            return groups
        except Exception as e:
            print(f"聚类失败: {e}")
            # 返回简单分组
            return [[i] for i in range(len(sentences))]
    
    def _merge_adjacent_groups(self, groups: List[List[int]], embeddings: np.ndarray) -> List[List[int]]:
        """合并相邻组 - 简化版"""
        try:
            return groups  # 🔥 暂时不合并，避免复杂度
        except Exception as e:
            print(f"合并失败: {e}")
            return groups
    
    def _create_final_chunks(self, groups: List[List[int]], sentences: List[str], base_metadata: Dict) -> List[Dict[str, Any]]:
        """生成最终分块 - 确保不丢失内容"""
        chunks = []
        
        for i, group in enumerate(groups):
            # 按顺序组合句子
            group_sentences = [sentences[idx] for idx in sorted(group)]
            chunk_text = '。'.join(group_sentences) + '。'
            
            # 🔥 移除小块过滤，确保所有内容都保存
            keywords = self._extract_keywords(group_sentences)
            
            chunks.append({
                'text': chunk_text,
                'metadata': {
                    **base_metadata,
                    'chunk_index': i,
                    'total_chunks': len(groups),
                    'semantic_method': 'clustering',
                    'sentence_count': len(group),
                    'keywords': keywords,
                    'semantic_coherence': self._calculate_coherence(group_sentences)
                }
            })
        
        return chunks
    
    def _simple_extract_keywords(self, text: str) -> List[str]:
            """简化的关键词提取 - 增加容错性"""
            try:
                # 尝试使用jieba
                words = jieba.cut(text)
                keywords = []
                for word in words:
                    if len(word) >= 2 and len(word) <= 4:
                        if re.match(r'[\u4e00-\u9fa5]+', word):  # 只要中文
                            keywords.append(word)
                
                # 去重并返回前5个
                return list(dict.fromkeys(keywords))[:5]
                
            except Exception as e:
                # 如果jieba失败，使用正则表达式
                try:
                    words = re.findall(r'[\u4e00-\u9fa5]{2,4}', text)
                    return list(dict.fromkeys(words))[:5]
                except:
                    return []
        
    def _simple_paragraph_grouping(self, paragraphs: List[str], base_metadata: Dict) -> List[Dict[str, Any]]:
        """简单的段落分组"""
        chunks = []
        current_chunk = ""
        chunk_index = 0
        
        for para in paragraphs:
            if len(current_chunk + "\n\n" + para) > self.max_chunk_size and current_chunk:
                # 保存当前chunk
                keywords = self._simple_extract_keywords(current_chunk)
                chunks.append({
                    'text': current_chunk.strip(),
                    'metadata': {
                        **base_metadata,
                        'chunk_index': chunk_index,
                        'total_chunks': 'unknown',
                        'semantic_method': 'simple_grouping',
                        'keywords': keywords
                    }
                })
                chunk_index += 1
                current_chunk = para
            else:
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para
        
        # 添加最后一个chunk
        if current_chunk:
            keywords = self._simple_extract_keywords(current_chunk)
            chunks.append({
                'text': current_chunk.strip(),
                'metadata': {
                    **base_metadata,
                    'chunk_index': chunk_index,
                    'total_chunks': chunk_index + 1,
                    'semantic_method': 'simple_grouping',
                    'keywords': keywords
                }
            })
        
        # 更新总数
        for chunk in chunks:
            chunk['metadata']['total_chunks'] = len(chunks)
        
        return chunks
class HybridGenericProcessor(BaseTextProcessor):
    """混合处理器：结合传统分块和语义分块"""
    
    def __init__(self, model_path: str = "../models/text2vec-base-chinese", use_semantic: bool = True):
        self.use_semantic = use_semantic
        if use_semantic:
            self.semantic_processor = SemanticGenericProcessor(model_path)
        self.traditional_processor = GenericProcessor()
    
    def clean_text(self, content: str) -> str:
        if self.use_semantic:
            return self.semantic_processor.clean_text(content)
        else:
            return self.traditional_processor.clean_text(content)
    
    def chunk_text(self, text: str, chunk_size: int = 800, overlap: int = 100) -> List[Dict[str, Any]]:
        if self.use_semantic and len(text) > 1000:  # 只对较长文本使用语义分块
            print("使用语义分块...")
            return self.semantic_processor.chunk_text(text, chunk_size, overlap)
        else:
            print("使用传统分块...")
            return self.traditional_processor.chunk_text(text, chunk_size, overlap)

class ProcessorFactory:
    """处理器工厂"""
    
    @staticmethod
    def get_processor(text_file: str = "", use_semantic: bool = False) -> BaseTextProcessor:
        """统一使用语义增强的通用处理器"""
        print(f"使用通用语义处理器 (文件: {text_file})")
        return HybridGenericProcessor(use_semantic=True)