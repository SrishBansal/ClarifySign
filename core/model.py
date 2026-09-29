"""
ClarifySign - Backward Compatibility Module for core.model
Delegates directly to core.recognizer.ISLRecognizer.
"""

from core.recognizer import ISLRecognizer, ISLBiLSTM

__all__ = ["ISLRecognizer", "ISLBiLSTM"]
