"""
多模态解析器测试脚本
用于验证第一阶段功能实现
"""

import os
import sys
from pathlib import Path

# 添加父目录到路径
parent_dir = str(Path(__file__).parent.parent)
sys.path.insert(0, parent_dir)
os.chdir(parent_dir)  # 切换工作目录到项目根目录

def test_imports():
    """测试模块导入"""
    print("=" * 60)
    print("测试 1: 模块导入检查")
    print("=" * 60)
    
    try:
        import multimodal_parser
        print("✓ multimodal_parser 模块导入成功")
    except ImportError as e:
        print(f"✗ multimodal_parser 导入失败: {e}")
    
    try:
        import storage_optimizer
        print("✓ storage_optimizer 模块导入成功")
    except ImportError as e:
        print(f"✗ storage_optimizer 导入失败: {e}")
    
    print()

def test_parser_initialization():
    """测试解析器初始化"""
    print("=" * 60)
    print("测试 2: 解析器初始化")
    print("=" * 60)
    
    try:
        from multimodal_parser import create_parser
        parser = create_parser()
        print("✓ 多模态解析器初始化成功")
        print()
    except Exception as e:
        print(f"✗ 初始化失败: {e}")
        print()

def test_dependencies():
    """测试依赖库可用性"""
    print("=" * 60)
    print("测试 3: 依赖库检查")
    print("=" * 60)
    
    dependencies = {
        "PyMuPDF (PDF解析)": "fitz",
        "python-docx (Word解析)": "docx",
        "python-pptx (PPT解析)": "pptx",
        "Pillow (图片处理)": "PIL",
        "OpenCV (图片预处理)": "cv2",
        "Tesseract (本地OCR)": "pytesseract",
        "Whisper (音频转写)": "whisper"
    }
    
    available_count = 0
    for name, module in dependencies.items():
        try:
            __import__(module)
            print(f"✓ {name:<30} 可用")
            available_count += 1
        except ImportError:
            print(f"✗ {name:<30} 未安装")
    
    print()
    print(f"依赖库可用率: {available_count}/{len(dependencies)} ({available_count/len(dependencies)*100:.0f}%)")
    print()

def test_parse_text_file():
    """测试解析文本文件"""
    print("=" * 60)
    print("测试 4: 文本文件解析")
    print("=" * 60)
    
    try:
        from multimodal_parser import create_parser
        parser = create_parser()
        
        # 创建临时测试文件
        test_file = "test_sample.txt"
        test_content = "这是一个测试文件。\n用于验证多模态解析器的文本处理功能。\nThis is a test file."
        
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(test_content)
        
        # 解析文件
        result = parser.parse_file(test_file)
        
        if result.success:
            print("✓ 文件解析成功")
            print(f"  - 提取内容长度: {len(result.content)} 字符")
            print(f"  - 内容预览: {result.content[:50]}...")
        else:
            print(f"✗ 解析失败: {result.error}")
        
        # 清理
        os.remove(test_file)
        print()
        
    except Exception as e:
        print(f"✗ 测试异常: {e}")
        print()

def test_provenance():
    """测试溯源记录"""
    print("=" * 60)
    print("测试 5: 溯源记录创建")
    print("=" * 60)
    
    try:
        from storage_optimizer import ProvenanceManager
        
        # 创建溯源记录
        provenance = ProvenanceManager.create_provenance(
            source_type='upload',
            original_filename='test.pdf',
            importer='test_user',
            metadata={'test': True}
        )
        
        print("✓ 溯源记录创建成功")
        print(f"  - 来源类型: {provenance['source_type']}")
        print(f"  - 原始文件名: {provenance['original_filename']}")
        print(f"  - 导入者: {provenance['importer']}")
        print(f"  - 导入时间: {provenance['import_time']}")
        print()
        
    except Exception as e:
        print(f"✗ 测试异常: {e}")
        print()

def run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("InsightVault 第一阶段功能测试")
    print("=" * 60)
    print()
    
    test_imports()
    test_parser_initialization()
    test_dependencies()
    test_parse_text_file()
    test_provenance()
    
    print("=" * 60)
    print("测试完成")
    print("=" * 60)
    print()
    print("提示：")
    print("- 如果有依赖缺失，请运行: pip install -r requirements.txt")
    print("- Tesseract OCR 需要单独安装系统包")
    print("- Whisper 是可选依赖（约1GB），按需安装")
    print()

if __name__ == "__main__":
    run_all_tests()
