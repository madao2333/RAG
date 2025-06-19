import sys
import argparse
from pathlib import Path
from rag_core import RAGCore
import logging

# 导入配置管理器
from config import get_config

# 导入PDF转换功能
try:
    from pdf_to_text import PDFToTextConverter
    PDF_AVAILABLE = True
except ImportError as e:
    print(f"PDF转换模块导入失败: {e}")
    print("请确保pdf_to_text.py在同一目录下")
    PDF_AVAILABLE = False

def process_input_file(input_path: str, pdf_method: str = None, output_dir: str = None) -> str:
    """
    处理输入文件，如果是PDF则转换为文本
    
    Args:
        input_path: 输入文件路径
        pdf_method: PDF提取方法，None时使用配置默认值
        output_dir: PDF转换输出目录，None时使用配置默认值
    
    Returns:
        文本文件路径
    """
    # 使用配置默认值
    config = get_config()
    path_config = config.get_path_config()
    text_config = config.get_text_processing_config()
    
    if pdf_method is None:
        pdf_method = text_config.pdf_method
    if output_dir is None:
        output_dir = path_config.pdf_output_dir
    
    input_path = Path(input_path)
    
    if not input_path.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_path}")
    
    # 如果已经是文本文件，直接返回
    if input_path.suffix.lower() == '.txt':
        print(f"📄 检测到文本文件，直接使用: {input_path}")
        return str(input_path)
    
    # 如果是PDF文件，进行转换
    elif input_path.suffix.lower() == '.pdf':
        if not PDF_AVAILABLE:
            raise ImportError("PDF转换模块不可用，无法处理PDF文件")
        
        print(f"📑 检测到PDF文件，开始转换...")
        
        # 🔥 直接使用pdf_to_text.py中的PDFToTextConverter
        converter = PDFToTextConverter(output_dir)
        
        # 调用转换方法
        success = converter.convert_single_pdf(str(input_path), pdf_method)
        
        if success:
            # 生成对应的文本文件路径
            text_file_path = Path(output_dir) / f"{input_path.stem}.txt"
            print(f"✅ PDF转换成功: {text_file_path}")
            return str(text_file_path)
        else:
            raise RuntimeError(f"PDF转换失败: {input_path}")
    
    else:
        raise ValueError(f"不支持的文件格式: {input_path.suffix}")

def main():
    """主函数 - 集成PDF转换和RAG构建"""
    # 🔥 从配置获取默认值
    config = get_config()
    path_config = config.get_path_config()
    model_config = config.get_model_config()
    text_config = config.get_text_processing_config()
    search_config = config.get_search_config()
    
    # 配置日志（使用配置中的设置）
    logger = logging.getLogger(__name__)
    
    parser = argparse.ArgumentParser(description="集成PDF转换和RAG系统构建器")
    
    # 将input_file改为可选参数
    parser.add_argument("input_file", nargs='?', help="输入文件路径（支持PDF或TXT）")
    parser.add_argument("-o", "--output", help="输出RAG系统目录")  
    
    parser.add_argument("-m", "--model", default=model_config.default,
                       help=f"向量模型路径 (默认: {model_config.default})")
    
    # PDF转换参数（使用配置默认值）
    parser.add_argument("--pdf-method", default=text_config.pdf_method,
                       choices=["auto", "pypdf2", "pdfplumber", "pymupdf"],
                       help=f"PDF提取方法 (默认: {text_config.pdf_method})")
    parser.add_argument("--pdf-output", default=path_config.pdf_output_dir,
                       help=f"PDF转换临时文件输出目录 (默认: {path_config.pdf_output_dir})")
    
    # 测试参数（使用配置默认值）
    parser.add_argument("--test-query", help="测试查询")
    parser.add_argument("--test-top-k", type=int, default=search_config.default_top_k,
                       help=f"测试查询返回结果数 (默认: {search_config.default_top_k})")
    parser.add_argument("--keep-txt", action="store_true", 
                       help="保留转换的文本文件")
    
    # 添加配置相关参数
    parser.add_argument("--show-config", action="store_true", 
                       help="显示当前配置并退出")
    parser.add_argument("--validate-config", action="store_true",
                       help="验证配置并退出")
    
    args = parser.parse_args()
    
    # 🔥 配置相关操作
    if args.show_config:
        from config import print_rag_status
        print_rag_status()
        return
    
    if args.validate_config:
        from config import validate_config
        validate_config()
        return
    
    # 🔥 检查必需参数（只在非配置命令时检查）
    if not args.input_file:
        parser.error("当执行构建任务时，需要提供 input_file 参数")
    
    if not args.output:
        parser.error("当执行构建任务时，需要提供 -o/--output 参数")
    
    try:
        # 🔥 第一步：处理输入文件（PDF转换或直接使用文本）
        print("=" * 60)
        print("🚀 步骤1: 处理输入文件")
        print("=" * 60)
        
        text_file_path = process_input_file(
            input_path=args.input_file,
            pdf_method=args.pdf_method,
            output_dir=args.pdf_output
        )
        print(f"📝 使用文本文件: {text_file_path}")
        
        # 🔥 第二步：构建RAG系统
        print("\n" + "=" * 60)
        print("🚀 步骤2: 构建RAG系统")
        print("=" * 60)
        
        rag_core = RAGCore(args.model)
        rag_system = rag_core.build_rag_system(
            text_file=text_file_path,
            save_dir=args.output
        )
        
        print(f"✅ RAG系统构建成功: {args.output}")
        
        # 🔥 第三步：自动将新建的RAG系统添加到配置中
        rag_name = Path(args.output).name
        if rag_name not in config.list_all_rags():
            from config import RAGConfig
            new_rag_config = RAGConfig(
                name=f"{rag_name.replace('_', ' ').title()}",
                description=f"基于 {Path(args.input_file).name} 构建的RAG系统",
                path=args.output,
                model_path=args.model,
                enabled=True
            )
            config.add_rag_config(rag_name, new_rag_config)
            print(f"📝 已将新RAG系统添加到配置: {rag_name}")
        
        # 🔥 第四步：测试查询（可选）
        if args.test_query:
            print("\n" + "=" * 60)
            print("🚀 步骤3: 测试查询")
            print("=" * 60)
            print(f"🔍 查询: {args.test_query}")
            
            results = rag_system.search(args.test_query, top_k=args.test_top_k)
            
            for i, result in enumerate(results, 1):
                print(f"\n📋 结果 {i}:")
                print(f"  相关度: {result['score']:.3f}")
                print(f"  来源: {result['metadata']}")
                print(f"  内容: {result['text'][:200]}...")
                print("-" * 40)
        
        # 🔥 清理临时文件（使用配置设置）
        system_config = config.get_system_config()
        if system_config.temp_cleanup and not args.keep_txt and args.input_file.lower().endswith('.pdf'):
            temp_txt = Path(text_file_path)
            if temp_txt.exists() and temp_txt.parent.name == args.pdf_output:
                try:
                    temp_txt.unlink()
                    print(f"🗑️  已清理临时文本文件: {temp_txt}")
                except Exception as e:
                    print(f"⚠️  清理临时文件失败: {e}")
        
        print(f"\n🎉 全部完成！RAG系统已保存到: {args.output}")
        
        # 🔥 显示可用的RAG系统
        print("\n📋 当前可用的RAG系统:")
        for rag_id in config.list_available_rags():
            rag_config_info = config.get_rag_config(rag_id)
            default_mark = " (默认)" if rag_id == config.get_default_rag_name() else ""
            print(f"  ✅ {rag_id}{default_mark}: {rag_config_info.name}")
        
    except Exception as e:
        logger.error(f"处理失败: {e}")
        print(f"❌ 错误: {e}")
        sys.exit(1)

def batch_process():
    """批量处理示例 - 使用配置文件"""
    config = get_config()
    model_config = config.get_model_config()
    path_config = config.get_path_config()
    
    rag_core = RAGCore(model_config.default)
    
    # 🔥 从配置文件读取批量处理任务
    # 您可以在这里定义批量处理的文件列表
    input_files = [
        (f"{path_config.pdf_input_dir}/extracted_texts/金.txt", "jingui_rag"),
        (f"{path_config.pdf_input_dir}/金匮要略.pdf", "jingui_rag_from_pdf"),
    ]
    
    print(f"🔄 开始批量处理 {len(input_files)} 个文件...")
    print(f"📁 PDF输入目录: {path_config.pdf_input_dir}")
    print(f"📁 PDF输出目录: {path_config.pdf_output_dir}")
    print(f"🤖 使用模型: {model_config.default}")
    print("=" * 60)
    
    for i, (input_file, save_dir) in enumerate(input_files, 1):
        input_path = Path(input_file)
        print(f"\n📂 [{i}/{len(input_files)}] 处理: {input_file}")
        
        if input_path.exists():
            try:
                # 处理输入文件（使用pdf_to_text.py的功能）
                text_file_path = process_input_file(input_file)
                
                # 构建RAG系统
                rag_system = rag_core.build_rag_system(text_file_path, save_dir)
                print(f"✓ 成功构建: {save_dir}")
                
                # 🔥 自动添加到配置
                if save_dir not in config.list_all_rags():
                    from config import RAGConfig
                    new_rag_config = RAGConfig(
                        name=f"{save_dir.replace('_', ' ').title()}",
                        description=f"基于 {input_path.name} 批量构建的RAG系统",
                        path=save_dir,
                        model_path=model_config.default,
                        enabled=True
                    )
                    config.add_rag_config(save_dir, new_rag_config)
                
            except Exception as e:
                print(f"✗ 构建失败: {e}")
        else:
            print(f"⚠️  文件不存在: {input_file}")
    
    print("\n🎉 批量处理完成！")
    print("\n📋 当前可用的RAG系统:")
    from config import print_rag_status
    print_rag_status()

def show_help():
    """显示帮助信息"""
    print("🔧 RAG系统构建器 - 使用说明")
    print("=" * 60)
    print()
    print("📋 基本用法:")
    print("  python main.py <输入文件> -o <输出目录>")
    print()
    print("📋 示例:")
    print("  # 从PDF构建RAG")
    print("  python main.py 金匮要略.pdf -o jingui_rag --test-query '桂枝汤的功效'")
    print()
    print("  # 从文本构建RAG")
    print("  python main.py 金匮要略.txt -o jingui_rag -t jingui")
    print()
    print("  # 查看配置状态")
    print("  python main.py --show-config")
    print()
    print("  # 验证配置")
    print("  python main.py --validate-config")
    print()
    print("📋 配置管理:")
    print("  # 查看配置状态")
    print("  python config.py")
    print()
    print("  # 在Python中管理配置")
    print("  from config import get_config, set_default_rag")
    print("  config = get_config()")
    print("  set_default_rag('wenbingxue_rag')")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # 如果有参数但是是help相关的
        if sys.argv[1] in ['-h', '--help', 'help']:
            show_help()
        else:
            main()
    else:
        # 无参数时显示选项
        print("🔧 RAG系统构建器")
        print("=" * 40)
        print("请选择操作:")
        print("1. 批量处理示例")
        print("2. 显示配置状态")
        print("3. 验证配置")
        print("4. 显示帮助")
        print()
        
        try:
            choice = input("请输入选择 (1-4): ").strip()
            
            if choice == "1":
                print("\n🔄 运行批量处理示例...")
                batch_process()
            elif choice == "2":
                from config import print_rag_status
                print_rag_status()
            elif choice == "3":
                from config import validate_config
                validate_config()
            elif choice == "4":
                show_help()
            else:
                print("无效选择，显示帮助信息...")
                show_help()
                
        except KeyboardInterrupt:
            print("\n\n👋 再见！")
        except Exception as e:
            print(f"\n❌ 发生错误: {e}")