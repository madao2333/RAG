import os
import sys
from pathlib import Path
import argparse
from typing import List, Optional
import logging

try:
    from config import get_config
    config = get_config()
    path_config = config.get_path_config()
    
    # 确保日志目录存在
    log_dir = Path(path_config.logs_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    pdf_log_path = path_config.pdf_log
except ImportError:
    # 如果无法导入配置，使用默认路径
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    pdf_log_path = "logs/pdf_conversion.log"

# 导入所需库
try:
    import PyPDF2
    import pdfplumber
    import fitz  # PyMuPDF
except ImportError as e:
    print(f"缺少必要的库: {e}")
    print("请安装: pip install PyPDF2 pdfplumber PyMuPDF")
    sys.exit(1)

# 创建独立的PDF转换logger
def setup_pdf_logger():
    """设置PDF转换专用的logger"""
    pdf_logger = logging.getLogger('pdf_conversion')
    pdf_logger.setLevel(logging.INFO)
    
    # 清除现有的handlers
    for handler in pdf_logger.handlers[:]:
        pdf_logger.removeHandler(handler)
    
    # 创建文件handler
    file_handler = logging.FileHandler(pdf_log_path, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    # 创建console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 创建formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 添加handlers
    pdf_logger.addHandler(file_handler)
    pdf_logger.addHandler(console_handler)
    
    # 防止日志传播到root logger
    pdf_logger.propagate = False
    
    return pdf_logger

# 使用独立的logger
logger = setup_pdf_logger()

class PDFToTextConverter:
    """PDF转文本转换器"""
    
    def __init__(self, output_dir: str = "extracted_texts"):
        """
        初始化转换器
        
        Args:
            output_dir: 输出目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def extract_with_pypdf2(self, pdf_path: str) -> str:
        """使用PyPDF2提取文本"""
        try:
            text = ""
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text.strip():
                            text += f"\n--- 第 {page_num + 1} 页 ---\n"
                            text += page_text + "\n"
                    except Exception as e:
                        logger.warning(f"PyPDF2提取第{page_num + 1}页失败: {e}")
            return text
        except Exception as e:
            logger.error(f"PyPDF2提取失败: {e}")
            return ""
    
    def extract_with_pdfplumber(self, pdf_path: str) -> str:
        """使用pdfplumber提取文本（更好地处理表格）"""
        try:
            text = ""
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text and page_text.strip():
                            text += f"\n--- 第 {page_num + 1} 页 ---\n"
                            text += page_text + "\n"
                        
                        # 提取表格
                        tables = page.extract_tables()
                        if tables:
                            text += "\n--- 表格内容 ---\n"
                            for table_num, table in enumerate(tables):
                                text += f"表格 {table_num + 1}:\n"
                                for row in table:
                                    if row:
                                        text += " | ".join([cell or "" for cell in row]) + "\n"
                                text += "\n"
                    except Exception as e:
                        logger.warning(f"pdfplumber提取第{page_num + 1}页失败: {e}")
            return text
        except Exception as e:
            logger.error(f"pdfplumber提取失败: {e}")
            return ""
    
    def extract_with_pymupdf(self, pdf_path: str) -> str:
        """使用PyMuPDF提取文本（处理复杂布局）"""
        try:
            text = ""
            pdf_document = fitz.open(pdf_path)
            for page_num in range(pdf_document.page_count):
                try:
                    page = pdf_document[page_num]
                    page_text = page.get_text()
                    if page_text.strip():
                        text += f"\n--- 第 {page_num + 1} 页 ---\n"
                        text += page_text + "\n"
                except Exception as e:
                    logger.warning(f"PyMuPDF提取第{page_num + 1}页失败: {e}")
            pdf_document.close()
            return text
        except Exception as e:
            logger.error(f"PyMuPDF提取失败: {e}")
            return ""
    
    def clean_text(self, text: str) -> str:
        """清理提取的文本"""
        if not text:
            return ""
        
        # 基本清理
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if line:
                # 移除过多的空格
                line = ' '.join(line.split())
                cleaned_lines.append(line)
        
        # 合并文本
        cleaned_text = '\n'.join(cleaned_lines)
        
        # 移除重复的换行符
        while '\n\n\n' in cleaned_text:
            cleaned_text = cleaned_text.replace('\n\n\n', '\n\n')
        
        return cleaned_text
    
    def convert_single_pdf(self, pdf_path: str, method: str = "auto") -> bool:
        """
        转换单个PDF文件
        
        Args:
            pdf_path: PDF文件路径
            method: 提取方法 ("pypdf2", "pdfplumber", "pymupdf", "auto")
        
        Returns:
            是否成功转换
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            logger.error(f"文件不存在: {pdf_path}")
            return False
        
        if not pdf_path.suffix.lower() == '.pdf':
            logger.error(f"不是PDF文件: {pdf_path}")
            return False
        
        logger.info(f"开始转换: {pdf_path.name}")
        
        # 根据方法提取文本
        text = ""
        if method == "auto":
            # 自动模式：依次尝试不同方法
            methods = [
                ("pdfplumber", self.extract_with_pdfplumber),
                ("pymupdf", self.extract_with_pymupdf),
                ("pypdf2", self.extract_with_pypdf2)
            ]
            
            for method_name, extract_func in methods:
                logger.info(f"尝试使用 {method_name} 提取...")
                text = extract_func(str(pdf_path))
                if text and len(text.strip()) > 100:  # 如果提取到足够的文本
                    logger.info(f"使用 {method_name} 成功提取")
                    break
                else:
                    logger.warning(f"{method_name} 提取结果不理想，尝试下一种方法")
        else:
            # 指定方法
            extract_methods = {
                "pypdf2": self.extract_with_pypdf2,
                "pdfplumber": self.extract_with_pdfplumber,
                "pymupdf": self.extract_with_pymupdf
            }
            
            if method in extract_methods:
                text = extract_methods[method](str(pdf_path))
            else:
                logger.error(f"未知的提取方法: {method}")
                return False
        
        if not text or len(text.strip()) < 10:
            logger.error(f"未能从PDF中提取到有效文本: {pdf_path.name}")
            return False
        
        # 清理文本
        cleaned_text = self.clean_text(text)
        
        # 保存文本
        output_file = self.output_dir / f"{pdf_path.stem}.txt"
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(f"# 源文件: {pdf_path.name}\n")
                f.write(f"# 提取时间: {logging.Formatter().formatTime(logging.LogRecord('', 0, '', 0, '', (), None))}\n")
                f.write(f"# 文本长度: {len(cleaned_text)} 字符\n\n")
                f.write(cleaned_text)
            
            logger.info(f"成功转换并保存: {output_file}")
            logger.info(f"提取文本长度: {len(cleaned_text)} 字符")
            return True
            
        except Exception as e:
            logger.error(f"保存文件失败: {e}")
            return False
    
    def convert_multiple_pdfs(self, pdf_paths: List[str], method: str = "auto") -> dict:
        """
        批量转换PDF文件
        
        Args:
            pdf_paths: PDF文件路径列表
            method: 提取方法
        
        Returns:
            转换结果统计
        """
        results = {
            "total": len(pdf_paths),
            "success": 0,
            "failed": 0,
            "failed_files": []
        }
        
        for pdf_path in pdf_paths:
            try:
                if self.convert_single_pdf(pdf_path, method):
                    results["success"] += 1
                else:
                    results["failed"] += 1
                    results["failed_files"].append(pdf_path)
            except Exception as e:
                logger.error(f"处理文件 {pdf_path} 时出错: {e}")
                results["failed"] += 1
                results["failed_files"].append(pdf_path)
        
        return results

def find_pdf_files(directory: str) -> List[str]:
    """在目录中查找所有PDF文件"""
    pdf_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.pdf'):
                pdf_files.append(os.path.join(root, file))
    return pdf_files

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="PDF转文本工具")
    parser.add_argument("input", help="输入PDF文件或目录路径")
    parser.add_argument("-o", "--output", default="extracted_texts", help="输出目录")
    parser.add_argument("-m", "--method", default="auto", 
                       choices=["auto", "pypdf2", "pdfplumber", "pymupdf"],
                       help="提取方法")
    
    args = parser.parse_args()
    
    # 创建转换器
    converter = PDFToTextConverter(args.output)
    
    input_path = Path(args.input)
    
    if input_path.is_file():
        # 单个文件
        success = converter.convert_single_pdf(str(input_path), args.method)
        if success:
            print(f"✓ 成功转换文件: {input_path.name}")
        else:
            print(f"✗ 转换失败: {input_path.name}")
    
    elif input_path.is_dir():
        # 目录批量处理
        pdf_files = find_pdf_files(str(input_path))
        if not pdf_files:
            print(f"在目录 {input_path} 中未找到PDF文件")
            return
        
        print(f"找到 {len(pdf_files)} 个PDF文件，开始批量转换...")
        results = converter.convert_multiple_pdfs(pdf_files, args.method)
        
        print(f"\n转换完成！")
        print(f"总计: {results['total']} 个文件")
        print(f"成功: {results['success']} 个文件")
        print(f"失败: {results['failed']} 个文件")
        
        if results['failed_files']:
            print(f"\n失败的文件:")
            for failed_file in results['failed_files']:
                print(f"  - {failed_file}")
    
    else:
        print(f"输入路径不存在: {input_path}")

if __name__ == "__main__":
    main()