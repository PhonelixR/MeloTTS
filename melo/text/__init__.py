"""
Módulo principal de procesamiento de texto para MeloTTS
Mantiene compatibilidad total con versiones existentes
"""

# Importar TODO de symbols para compatibilidad completa
from .symbols import *

# Importar todos los módulos de idiomas
from . import chinese
from . import japanese
from . import english
from . import chinese_mix
from . import korean
from . import french
from . import spanish

# Diccionario de símbolos (compatibilidad)
_symbol_to_id = {s: i for i, s in enumerate(symbols)}


def cleaned_text_to_sequence(cleaned_text, tones, language, symbol_to_id=None):
    """Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
    Args:
      cleaned_text: list of symbols
      tones: list of tone values
      language: language code
      symbol_to_id: optional custom symbol to id mapping
    Returns:
      phones, tones, lang_ids
    """
    symbol_to_id_map = symbol_to_id if symbol_to_id else _symbol_to_id
    phones = [symbol_to_id_map[symbol] for symbol in cleaned_text]
    tone_start = language_tone_start_map[language]
    tones = [i + tone_start for i in tones]
    lang_id = language_id_map[language]
    lang_ids = [lang_id for _ in phones]
    return phones, tones, lang_ids


def get_bert(norm_text, word2ph, language, device):
    """Obtiene características BERT para el texto normalizado."""
    # Importación condicional por eficiencia
    if language == "ZH":
        from .chinese_bert import get_bert_feature as bert_func
    elif language == "EN":
        from .english_bert import get_bert_feature as bert_func
    elif language == "JP":
        from .japanese_bert import get_bert_feature as bert_func
    elif language == "ZH_MIX_EN":
        from .chinese_mix import get_bert_feature as bert_func
    elif language in ["SP", "ES"]:
        from .spanish_bert import get_bert_feature as bert_func
    elif language == "FR":
        from .french_bert import get_bert_feature as bert_func
    elif language == "KR":
        from .korean import get_bert_feature as bert_func
    else:
        raise ValueError(f"Language '{language}' not supported for BERT. "
                       f"Available: ZH, EN, JP, ZH_MIX_EN, SP/ES, FR, KR")

    return bert_func(norm_text, word2ph, device)


# Exportar las funciones y módulos principales
__all__ = [
    # Funciones principales
    'cleaned_text_to_sequence',
    'get_bert',
    
    # Módulos de idiomas
    'chinese',
    'japanese', 
    'english',
    'chinese_mix',
    'korean',
    'french',
    'spanish',
    
    # Símbolos principales
    'symbols',
    'punctuation',
    'language_tone_start_map',
    'language_id_map',
    'num_languages',
    'num_tones',
    'sil_phonemes_ids',
]
