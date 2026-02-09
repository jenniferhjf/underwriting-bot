"""
Enhanced OCR Module - 多种开源OCR方案
======================================

支持的OCR引擎：
1. Tesseract - 通用OCR（默认）
2. PaddleOCR - 中文手写识别最强（推荐）
3. EasyOCR - 多语言支持
4. TrOCR - Transformer based（手写识别）

Version: 2.1.0
"""

import os
from typing import List, Dict, Optional, Union
from PIL import Image
import numpy as np

# Tesseract (默认)
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    print("⚠️  Tesseract未安装")

# PaddleOCR (推荐用于中文)
try:
    from paddleocr import PaddleOCR
    PADDLE_AVAILABLE = True
except ImportError:
    PADDLE_AVAILABLE = False

# EasyOCR
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

# TrOCR
try:
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    import torch
    TROCR_AVAILABLE = True
except ImportError:
    TROCR_AVAILABLE = False


class EnhancedOCR:
    """增强型OCR识别器 - 支持多种开源引擎"""
    
    def __init__(self, 
                 engine: str = 'paddleocr',
                 language: str = 'ch',
                 use_gpu: bool = False):
        """
        初始化OCR引擎
        
        Args:
            engine: OCR引擎 ('tesseract', 'paddleocr', 'easyocr', 'trocr')
            language: 语言 ('ch'=中文, 'en'=英文, 'ch+en'=中英文)
            use_gpu: 是否使用GPU加速
        """
        self.engine = engine.lower()
        self.language = language
        self.use_gpu = use_gpu
        self.ocr_model = None
        
        # 初始化对应的OCR引擎
        self._initialize_engine()
    
    def _initialize_engine(self):
        """初始化OCR引擎"""
        
        if self.engine == 'paddleocr':
            if not PADDLE_AVAILABLE:
                print("⚠️  PaddleOCR未安装，切换到Tesseract")
                self.engine = 'tesseract'
            else:
                print("✅ 使用PaddleOCR（推荐用于中文手写）")
                self.ocr_model = PaddleOCR(
                    use_angle_cls=True,
                    lang='ch' if 'ch' in self.language else 'en',
                    use_gpu=self.use_gpu,
                    show_log=False
                )
        
        elif self.engine == 'easyocr':
            if not EASYOCR_AVAILABLE:
                print("⚠️  EasyOCR未安装，切换到Tesseract")
                self.engine = 'tesseract'
            else:
                print("✅ 使用EasyOCR")
                # 语言映射
                langs = []
                if 'ch' in self.language:
                    langs.append('ch_sim')
                if 'en' in self.language:
                    langs.append('en')
                
                self.ocr_model = easyocr.Reader(
                    langs,
                    gpu=self.use_gpu
                )
        
        elif self.engine == 'trocr':
            if not TROCR_AVAILABLE:
                print("⚠️  TrOCR未安装，切换到Tesseract")
                self.engine = 'tesseract'
            else:
                print("✅ 使用TrOCR（手写识别）")
                model_name = 'microsoft/trocr-base-handwritten'
                self.ocr_processor = TrOCRProcessor.from_pretrained(model_name)
                self.ocr_model = VisionEncoderDecocoderModel.from_pretrained(model_name)
                
                if self.use_gpu and torch.cuda.is_available():
                    self.ocr_model = self.ocr_model.to('cuda')
        
        elif self.engine == 'tesseract':
            if not TESSERACT_AVAILABLE:
                raise ImportError("Tesseract未安装！请安装: pip install pytesseract")
            print("✅ 使用Tesseract OCR")
        
        else:
            raise ValueError(f"不支持的OCR引擎: {self.engine}")
    
    def recognize(self, image: Union[str, Image.Image, np.ndarray]) -> Dict:
        """
        识别图像中的文字
        
        Args:
            image: 图像路径、PIL Image或numpy数组
        
        Returns:
            {
                'text': '识别的文本',
                'confidence': 0.95,  # 置信度
                'details': [...]      # 详细信息（可选）
            }
        """
        # 转换图像格式
        if isinstance(image, str):
            image = Image.open(image)
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # 根据引擎调用对应的识别方法
        if self.engine == 'paddleocr':
            return self._recognize_paddle(image)
        elif self.engine == 'easyocr':
            return self._recognize_easy(image)
        elif self.engine == 'trocr':
            return self._recognize_trocr(image)
        else:  # tesseract
            return self._recognize_tesseract(image)
    
    def _recognize_paddle(self, image: Image.Image) -> Dict:
        """PaddleOCR识别"""
        # 转换为numpy数组
        img_array = np.array(image)
        
        # 识别
        result = self.ocr_model.ocr(img_array, cls=True)
        
        if not result or not result[0]:
            return {'text': '', 'confidence': 0.0, 'details': []}
        
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
        """EasyOCR识别"""
        # 转换为numpy数组
        img_array = np.array(image)
        
        # 识别
        result = self.ocr_model.readtext(img_array)
        
        if not result:
            return {'text': '', 'confidence': 0.0, 'details': []}
        
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
    
    def _recognize_trocr(self, image: Image.Image) -> Dict:
        """TrOCR识别"""
        # 预处理
        pixel_values = self.ocr_processor(
            image,
            return_tensors="pt"
        ).pixel_values
        
        if self.use_gpu and torch.cuda.is_available():
            pixel_values = pixel_values.to('cuda')
        
        # 生成
        generated_ids = self.ocr_model.generate(pixel_values)
        
        # 解码
        text = self.ocr_processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0]
        
        return {
            'text': text,
            'confidence': 0.9,  # TrOCR不直接提供置信度
            'details': [{'text': text, 'confidence': 0.9}]
        }
    
    def _recognize_tesseract(self, image: Image.Image) -> Dict:
        """Tesseract识别"""
        # 语言映射
        lang_map = {
            'ch': 'chi_sim',
            'en': 'eng',
            'ch+en': 'chi_sim+eng'
        }
        lang = lang_map.get(self.language, 'eng')
        
        # 识别
        text = pytesseract.image_to_string(image, lang=lang)
        
        # 获取详细信息（包含置信度）
        try:
            data = pytesseract.image_to_data(
                image,
                lang=lang,
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
            avg_conf = 0.0
        
        return {
            'text': text.strip(),
            'confidence': avg_conf,
            'details': []
        }
    
    def recognize_batch(self, images: List[Union[str, Image.Image, np.ndarray]]) -> List[Dict]:
        """批量识别"""
        results = []
        for img in images:
            result = self.recognize(img)
            results.append(result)
        return results


def create_ocr(engine: str = None, **kwargs) -> EnhancedOCR:
    """
    创建OCR实例（工厂函数）
    
    Args:
        engine: OCR引擎，如果为None则自动选择最佳可用引擎
        **kwargs: 其他参数
    
    Returns:
        EnhancedOCR实例
    """
    # 自动选择最佳引擎
    if engine is None:
        if PADDLE_AVAILABLE:
            engine = 'paddleocr'
            print("🎯 自动选择：PaddleOCR（中文手写最强）")
        elif EASYOCR_AVAILABLE:
            engine = 'easyocr'
            print("🎯 自动选择：EasyOCR")
        elif TROCR_AVAILABLE:
            engine = 'trocr'
            print("🎯 自动选择：TrOCR（手写识别）")
        elif TESSERACT_AVAILABLE:
            engine = 'tesseract'
            print("🎯 自动选择：Tesseract")
        else:
            raise ImportError("没有可用的OCR引擎！请至少安装一个：\n"
                            "pip install paddleocr  # 推荐\n"
                            "pip install easyocr\n"
                            "pip install pytesseract")
    
    return EnhancedOCR(engine=engine, **kwargs)


if __name__ == '__main__':
    # 测试代码
    print("=== Enhanced OCR 测试 ===\n")
    
    # 检查可用的引擎
    print("可用的OCR引擎：")
    if PADDLE_AVAILABLE:
        print("  ✅ PaddleOCR - 推荐用于中文手写")
    if EASYOCR_AVAILABLE:
        print("  ✅ EasyOCR - 多语言支持")
    if TROCR_AVAILABLE:
        print("  ✅ TrOCR - Transformer based")
    if TESSERACT_AVAILABLE:
        print("  ✅ Tesseract - 通用OCR")
    
    print("\n推荐安装：")
    print("  pip install paddleocr paddlepaddle")
    print("  pip install easyocr")
    
    # 自动创建OCR
    ocr = create_ocr(language='ch+en')
    print(f"\n当前使用引擎: {ocr.engine}")
