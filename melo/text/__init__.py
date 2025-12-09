# Importar todo de symbols para mantener compatibilidad
from .symbols import *

# Crear el diccionario de símbolos
_symbol_to_id = {s: i for i, s in enumerate(symbols)}

def cleaned_text_to_sequence(cleaned_text, tones, language, symbol_to_id=None):
    """Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
    Args:
      text: string to convert to a sequence
    Returns:
      List of integers corresponding to the symbols in the text
    """
    symbol_to_id_map = symbol_to_id if symbol_to_id else _symbol_to_id
    phones = [symbol_to_id_map[symbol] for symbol in cleaned_text]
    tone_start = language_tone_start_map[language]
    tones = [i + tone_start for i in tones]
    lang_id = language_id_map[language]
    lang_ids = [lang_id for i in phones]
    return phones, tones, lang_ids


def get_bert(norm_text, word2ph, language, device):
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
        raise ValueError(f"Language '{language}' not supported for BERT. Available: ZH, EN, JP, ZH_MIX_EN, SP/ES, FR, KR")
    
    bert = bert_func(norm_text, word2ph, device)
    return bert
