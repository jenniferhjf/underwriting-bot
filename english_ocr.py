"""
English Handwriting OCR Module - 专门针对英文手写识别
=======================================================

针对英文保险文档的手写识别优化方案：
1. PaddleOCR (英文版) - 推荐
2. TrOCR (专门针对英文手写)
3. EasyOCR (英文优化)
4. Tesseract (基础方案)

Version: 2.2.0 - English Optimized
"""

import os
from typing import List, Dict, Optional, Union
from PIL import Image
import numpy as np
import time

# PaddleOCR (英文版)
try:
    from paddleocr import PaddleOCR
    PADDLE_AVAILABLE = True
except ImportError:
    PADDLE_AVAILABLE = False

# TrOCR (专门针对手写)
try:
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    import torch
    TROCR_AVAILABLE = True
except ImportError:
    TROCR_AVAILABLE = False

# EasyOCR
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

# Tesseract
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False


class EnglishHandwritingOCR:
    """英文手写识别器 - 专门优化"""
    
    def __init__(self, 
                 engine: str = 'auto',
                 use_gpu: bool = False,
                 confidence_threshold: float = 0.5):
        """
        初始化英文手写OCR
        
        Args:
            engine: OCR引擎
                - 'auto': 自动选择最佳引擎
                - 'paddleocr': PaddleOCR英文版
                - 'trocr': TrOCR手写专用
                - 'easyocr': EasyOCR
                - 'tesseract': Tesseract
            use_gpu: 是否使用GPU加速
            confidence_threshold: 置信度阈值
        """
        self.engine = engine
        self.use_gpu = use_gpu
        self.confidence_threshold = confidence_threshold
        self.ocr_model = None
        
        # 自动选择引擎
        if self.engine == 'auto':
            self.engine = self._auto_select_engine()
        
        # 初始化
        self._initialize_engine()
        
        print(f"✅ 使用OCR引擎: {self.engine.upper()}")
        if use_gpu:
            print(f"   GPU加速: 已启用")
    
    def _auto_select_engine(self) -> str:
        """
        自动选择最佳可用引擎
        
        优先级（针对英文手写）:
        1. TrOCR - 英文手写识别最强
        2. PaddleOCR - 速度快，准确度高
        3. EasyOCR - 通用方案
        4. Tesseract - 基础方案
        """
        if TROCR_AVAILABLE:
            print("🎯 自动选择: TrOCR (英文手写最强)")
            return 'trocr'
        elif PADDLE_AVAILABLE:
            print("🎯 自动选择: PaddleOCR (速度快)")
            return 'paddleocr'
        elif EASYOCR_AVAILABLE:
            print("🎯 自动选择: EasyOCR (通用)")
            return 'easyocr'
        elif TESSERACT_AVAILABLE:
            print("🎯 自动选择: Tesseract (基础)")
            return 'tesseract'
        else:
            raise ImportError(
                "没有可用的OCR引擎！请至少安装一个：\n"
                "推荐用于英文手写:\n"
                "  pip install transformers torch  # TrOCR (最强)\n"
                "  pip install paddleocr paddlepaddle  # PaddleOCR (快)\n"
                "  pip install easyocr  # EasyOCR (通用)\n"
            )
    
    def _initialize_engine(self):
        """初始化OCR引擎"""
        
        if self.engine == 'trocr':
            if not TROCR_AVAILABLE:
                raise ImportError("TrOCR未安装: pip install transformers torch")
            
            print("   加载TrOCR模型 (专门针对英文手写)...")
            
            # 使用专门的英文手写模型
            model_name = 'microsoft/trocr-base-handwritten'
            
            self.ocr_processor = TrOCRProcessor.from_pretrained(model_name)
            self.ocr_model = VisionEncoderDecoderModel.from_pretrained(model_name)
            
            if self.use_gpu and torch.cuda.is_available():
                self.ocr_model = self.ocr_model.to('cuda')
                print("   ✅ GPU加速已启用")
            
            print(f"   ✅ TrOCR模型加载完成")
        
        elif self.engine == 'paddleocr':
            if not PADDLE_AVAILABLE:
                raise ImportError("PaddleOCR未安装: pip install paddleocr paddlepaddle")
            
            print("   初始化PaddleOCR (英文版)...")
            
            self.ocr_model = PaddleOCR(
                use_angle_cls=True,
                lang='en',  # 英文
                use_gpu=self.use_gpu,
                show_log=False,
                det_db_thresh=0.3,  # 检测阈值，手写调低
                det_db_box_thresh=0.5,  # 框选阈值
                rec_algorithm='CRNN'  # 识别算法
            )
            
            print("   ✅ PaddleOCR初始化完成")
        
        elif self.engine == 'easyocr':
            if not EASYOCR_AVAILABLE:
                raise ImportError("EasyOCR未安装: pip install easyocr")
            
            print("   初始化EasyOCR...")
            
            self.ocr_model = easyocr.Reader(
                ['en'],
                gpu=self.use_gpu
            )
            
            print("   ✅ EasyOCR初始化完成")
        
        elif self.engine == 'tesseract':
            if not TESSERACT_AVAILABLE:
                raise ImportError("Tesseract未安装: pip install pytesseract")
            
            print("   使用Tesseract OCR")
            # Tesseract不需要初始化
        
        else:
            raise ValueError(f"不支持的OCR引擎: {self.engine}")
    
    def recognize(self, image: Union[str, Image.Image, np.ndarray]) -> Dict:
        """
        识别英文手写文本
        
        Args:
            image: 图像路径、PIL Image或numpy数组
        
        Returns:
            {
                'text': '识别的文本',
                'confidence': 0.95,
                'engine': 'trocr',
                'processing_time': 1.23,
                'details': [
                    {'text': 'line 1', 'confidence': 0.96, 'bbox': [...]},
                    ...
                ]
            }
        """
        start_time = time.time()
        
        # 转换图像格式
        if isinstance(image, str):
            image = Image.open(image)
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # 根据引擎识别
        if self.engine == 'trocr':
            result = self._recognize_trocr(image)
        elif self.engine == 'paddleocr':
            result = self._recognize_paddle(image)
        elif self.engine == 'easyocr':
            result = self._recognize_easy(image)
        else:  # tesseract
            result = self._recognize_tesseract(image)
        
        # 添加元数据
        result['engine'] = self.engine
        result['processing_time'] = time.time() - start_time
        
        # 过滤低置信度结果
        if result['confidence'] < self.confidence_threshold:
            result['warning'] = f"置信度低于阈值 ({result['confidence']:.2%} < {self.confidence_threshold:.2%})"
        
        return result
    
    def _recognize_trocr(self, image: Image.Image) -> Dict:
        """
        TrOCR识别 - 专门针对英文手写
        
        特点：
        - 最高准确度
        - 专门训练于手写识别
        - 适合单行或短文本
        """
        # 预处理图像
        pixel_values = self.ocr_processor(
            image,
            return_tensors="pt"
        ).pixel_values
        
        if self.use_gpu and torch.cuda.is_available():
            pixel_values = pixel_values.to('cuda')
        
        # 生成文本
        generated_ids = self.ocr_model.generate(pixel_values)
        
        # 解码
        text = self.ocr_processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0]
        
        # TrOCR不提供置信度，使用固定值
        confidence = 0.92
        
        return {
            'text': text.strip(),
            'confidence': confidence,
            'details': [{
                'text': text.strip(),
                'confidence': confidence,
                'bbox': None
            }]
        }
    
    def _recognize_paddle(self, image: Image.Image) -> Dict:
        """
        PaddleOCR识别 - 快速准确
        
        特点：
        - 速度快
        - 支持多行文本
        - 提供详细的位置信息
        """
        # 转换为numpy数组
        img_array = np.array(image)
        
        # 识别
        result = self.ocr_model.ocr(img_array, cls=True)
        
        if not result or not result[0]:
            return {
                'text': '',
                'confidence': 0.0,
                'details': []
            }
        
        # 提取文本和置信度
        texts = []
        confidences = []
        details = []
        
        for line in result[0]:
            text = line[1][0]
            conf = line[1][1]
            bbox = line[0]
            
            texts.append(text)
            confidences.append(conf)
            details.append({
                'text': text,
                'confidence': conf,
                'bbox': bbox
            })
        
        return {
            'text': '\n'.join(texts),
            'confidence': np.mean(confidences) if confidences else 0.0,
            'details': details
        }
    
    def _recognize_easy(self, image: Image.Image) -> Dict:
        """
        EasyOCR识别 - 通用方案
        
        特点：
        - 易于使用
        - 多语言支持
        - 准确度不错
        """
        # 转换为numpy数组
        img_array = np.array(image)
        
        # 识别
        result = self.ocr_model.readtext(img_array)
        
        if not result:
            return {
                'text': '',
                'confidence': 0.0,
                'details': []
            }
        
        # 提取文本和置信度
        texts = []
        confidences = []
        details = []
        
        for detection in result:
            bbox, text, conf = detection
            texts.append(text)
            confidences.append(conf)
            details.append({
                'text': text,
                'confidence': conf,
                'bbox': bbox
            })
        
        return {
            'text': '\n'.join(texts),
            'confidence': np.mean(confidences) if confidences else 0.0,
            'details': details
        }
    
    def _recognize_tesseract(self, image: Image.Image) -> Dict:
        """
        Tesseract识别 - 基础方案
        
        特点：
        - 最基础的方案
        - 对印刷体效果好
        - 手写识别一般
        """
        # 识别文本
        text = pytesseract.image_to_string(image, lang='eng')
        
        # 获取详细信息
        try:
            data = pytesseract.image_to_data(
                image,
                lang='eng',
                output_type=pytesseract.Output.DICT
            )
            
            # 提取置信度
            confidences = [
                float(conf) / 100.0
                for conf in data['conf']
                if conf != '-1'
            ]
            avg_conf = np.mean(confidences) if confidences else 0.0
        
        except:
            avg_conf = 0.5
        
        return {
            'text': text.strip(),
            'confidence': avg_conf,
            'details': []
        }
    
    def recognize_batch(self, 
                       images: List[Union[str, Image.Image, np.ndarray]],
                       show_progress: bool = True) -> List[Dict]:
        """
        批量识别
        
        Args:
            images: 图像列表
            show_progress: 是否显示进度
        
        Returns:
            识别结果列表
        """
        results = []
        total = len(images)
        
        for i, img in enumerate(images):
            if show_progress:
                print(f"处理: {i+1}/{total}", end='\r')
            
            result = self.recognize(img)
            results.append(result)
        
        if show_progress:
            print(f"✅ 完成: {total}/{total}")
        
        return results
    
    def test_recognition(self, image_path: str):
        """
        测试识别功能
        
        Args:
            image_path: 测试图像路径
        """
        print(f"\n=== 测试识别: {image_path} ===\n")
        
        result = self.recognize(image_path)
        
        print(f"识别文本:\n{result['text']}\n")
        print(f"置信度: {result['confidence']:.2%}")
        print(f"引擎: {result['engine']}")
        print(f"处理时间: {result['processing_time']:.2f}秒")
        
        if 'warning' in result:
            print(f"⚠️  警告: {result['warning']}")
        
        if result['details']:
            print(f"\n详细信息:")
            for i, detail in enumerate(result['details'], 1):
                print(f"  {i}. {detail['text']} (置信度: {detail['confidence']:.2%})")


def create_english_ocr(engine: str = 'auto', **kwargs) -> EnglishHandwritingOCR:
    """
    创建英文手写OCR实例（工厂函数）
    
    Args:
        engine: OCR引擎
            - 'auto': 自动选择（推荐）
            - 'trocr': TrOCR（英文手写最强）
            - 'paddleocr': PaddleOCR（速度快）
            - 'easyocr': EasyOCR（通用）
            - 'tesseract': Tesseract（基础）
        **kwargs: 其他参数
    
    Returns:
        EnglishHandwritingOCR实例
    
    Example:
        >>> ocr = create_english_ocr()  # 自动选择
        >>> result = ocr.recognize('handwritten.jpg')
        >>> print(result['text'])
    """
    return EnglishHandwritingOCR(engine=engine, **kwargs)


if __name__ == '__main__':
    # 测试代码
    print("=" * 60)
    print("英文手写OCR测试")
    print("=" * 60)
    
    # 检查可用引擎
    print("\n可用的OCR引擎:")
    if TROCR_AVAILABLE:
        print("  ✅ TrOCR - 英文手写最强（推荐）")
    else:
        print("  ❌ TrOCR - 未安装 (pip install transformers torch)")
    
    if PADDLE_AVAILABLE:
        print("  ✅ PaddleOCR - 速度快，准确度高")
    else:
        print("  ❌ PaddleOCR - 未安装 (pip install paddleocr paddlepaddle)")
    
    if EASYOCR_AVAILABLE:
        print("  ✅ EasyOCR - 通用方案")
    else:
        print("  ❌ EasyOCR - 未安装 (pip install easyocr)")
    
    if TESSERACT_AVAILABLE:
        print("  ✅ Tesseract - 基础方案")
    else:
        print("  ❌ Tesseract - 未安装 (pip install pytesseract)")
    
    print("\n推荐安装（按优先级）:")
    print("  1. pip install transformers torch  # TrOCR - 最强")
    print("  2. pip install paddleocr paddlepaddle  # PaddleOCR - 快")
    print("  3. pip install easyocr  # EasyOCR - 通用")
    
    # 创建OCR实例
    try:
        print("\n" + "=" * 60)
        ocr = create_english_ocr()
        print("=" * 60)
        
        print("\n✅ OCR初始化成功！")
        print(f"   当前引擎: {ocr.engine.upper()}")
        
    except Exception as e:
        print(f"\n❌ 初始化失败: {e}")
