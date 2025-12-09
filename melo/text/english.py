import pickle
import os
import re
import nltk
import warnings

# Intentar importar g2p_en, pero con manejo robusto de errores
try:
    from g2p_en import G2p
    HAS_G2P = True
except ImportError:
    HAS_G2P = False
    warnings.warn("⚠️ g2p_en no está instalado. Usando diccionario CMU y fallback simple.")

from . import symbols

from .english_utils.abbreviations import expand_abbreviations
from .english_utils.time_norm import expand_time_english
from .english_utils.number_norm import normalize_numbers
from .japanese import distribute_phone

from transformers import AutoTokenizer

current_file_path = os.path.dirname(__file__)
CMU_DICT_PATH = os.path.join(current_file_path, "cmudict.rep")
CACHE_PATH = os.path.join(current_file_path, "cmudict_cache.pickle")

# ====== INICIALIZACIÓN ROBUSTA ======
def _init_g2p():
    """Inicialización con múltiples intentos y fallbacks"""
    if not HAS_G2P:
        return None
    
    try:
        # Intento 1: Inicialización normal
        return G2p()
    except Exception as e:
        # Intento 2: Descargar recursos NLTK y reintentar
        try:
            nltk.download('averaged_perceptron_tagger', quiet=True)
            nltk.download('punkt', quiet=True)
            nltk.download('cmudict', quiet=True)
            return G2p()
        except Exception as e2:
            warnings.warn(f"⚠️ g2p_en falló: {str(e)[:100]}")
            return None

_g2p = _init_g2p()

# ====== FALLBACK PARA PALABRAS NO ENCONTRADAS ======
def _simple_g2p_fallback(word):
    """Fallback simple para palabras cuando g2p_en no funciona"""
    # Reglas simples de pronunciación para caracteres comunes
    simple_map = {
        '-': ['-'],
        '.': ['.'],
        ',': [','],
        '!': ['!'],
        '?': ['?'],
        ';': [';'],
        ':': [':'],
        "'": ["'"],
        '"': ['"'],
        '(': ['('],
        ')': [')'],
        '[': ['['],
        ']': [']'],
        '{': ['{'],
        '}': ['}'],
    }
    
    if word in simple_map:
        return simple_map[word]
    
    # Para palabras reales, intentar dividir por sílabas simples
    # Esto es muy básico, pero mejor que nada
    word_lower = word.lower()
    phones = []
    
    # Reglas de conversión muy simples (no precisa, solo para emergencia)
    vowel_sounds = ['AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'EH', 'ER', 'EY', 'IH', 'IY', 'OW', 'OY', 'UH', 'UW']
    consonant_sounds = ['B', 'CH', 'D', 'DH', 'F', 'G', 'HH', 'JH', 'K', 'L', 'M', 'N', 'NG', 'P', 'R', 'S', 'SH', 'T', 'TH', 'V', 'W', 'Y', 'Z', 'ZH']
    
    # Si la palabra es corta, devolver un fonema simple
    if len(word_lower) <= 3:
        return ['AH']  # Sonido de schwa como fallback
    
    # Devolver una secuencia simple basada en letras
    return ['AH', 'N', 'D']  # "and" como fallback genérico

arpa = {
    "AH0", "S", "AH1", "EY2", "AE2", "EH0", "OW2", "UH0", "NG", "B",
    "G", "AY0", "M", "AA0", "F", "AO0", "ER2", "UH1", "IY1", "AH2",
    "DH", "IY0", "EY1", "IH0", "K", "N", "W", "IY2", "T", "AA1",
    "ER1", "EH2", "OY0", "UH2", "UW1", "Z", "AW2", "AW1", "V", "UW2",
    "AA2", "ER", "AW0", "UW0", "R", "OW1", "EH1", "ZH", "AE0", "IH2",
    "IH", "Y", "JH", "P", "AY1", "EY0", "OY2", "TH", "HH", "D", "ER0",
    "CH", "AO1", "AE1", "AO2", "OY1", "AY2", "IH1", "OW0", "L", "SH",
}


def post_replace_ph(ph):
    rep_map = {
        "：": ",", "；": ",", "，": ",", "。": ".", "！": "!", "？": "?",
        "\n": ".", "·": ",", "、": ",", "...": "…", "v": "V",
    }
    if ph in rep_map.keys():
        ph = rep_map[ph]
    if ph in symbols:
        return ph
    if ph not in symbols:
        ph = "UNK"
    return ph


def read_dict():
    g2p_dict = {}
    start_line = 49
    with open(CMU_DICT_PATH) as f:
        line = f.readline()
        line_index = 1
        while line:
            if line_index >= start_line:
                line = line.strip()
                word_split = line.split("  ")
                word = word_split[0]

                syllable_split = word_split[1].split(" - ")
                g2p_dict[word] = []
                for syllable in syllable_split:
                    phone_split = syllable.split(" ")
                    g2p_dict[word].append(phone_split)

            line_index = line_index + 1
            line = f.readline()

    return g2p_dict


def cache_dict(g2p_dict, file_path):
    with open(file_path, "wb") as pickle_file:
        pickle.dump(g2p_dict, pickle_file)


def get_dict():
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "rb") as pickle_file:
            g2p_dict = pickle.load(pickle_file)
    else:
        g2p_dict = read_dict()
        cache_dict(g2p_dict, CACHE_PATH)

    return g2p_dict


eng_dict = get_dict()


def refine_ph(phn):
    tone = 0
    if re.search(r"\d$", phn):
        tone = int(phn[-1]) + 1
        phn = phn[:-1]
    return phn.lower(), tone


def refine_syllables(syllables):
    tones = []
    phonemes = []
    for phn_list in syllables:
        for i in range(len(phn_list)):
            phn = phn_list[i]
            phn, tone = refine_ph(phn)
            phonemes.append(phn)
            tones.append(tone)
    return phonemes, tones


def text_normalize(text):
    text = text.lower()
    text = expand_time_english(text)
    text = normalize_numbers(text)
    text = expand_abbreviations(text)
    return text

model_id = 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_id)


def g2p(text, pad_start_end=True, tokenized=None):
    """Versión robusta de g2p que no depende críticamente de g2p_en"""
    if tokenized is None:
        tokenized = tokenizer.tokenize(text)
    
    ph_groups = []
    for t in tokenized:
        if not t.startswith("#"):
            ph_groups.append([t])
        else:
            ph_groups[-1].append(t.replace("#", ""))
    
    phones = []
    tones = []
    word2ph = []
    for group in ph_groups:
        w = "".join(group)
        phone_len = 0
        word_len = len(group)
        
        # PRIMERO: Intentar con el diccionario CMU
        if w.upper() in eng_dict:
            phns, tns = refine_syllables(eng_dict[w.upper()])
            phones += phns
            tones += tns
            phone_len += len(phns)
        else:
            # SEGUNDO: Intentar con g2p_en si está disponible
            if _g2p is not None:
                try:
                    phone_list = list(filter(lambda p: p != " ", _g2p(w)))
                    for ph in phone_list:
                        if ph in arpa:
                            ph, tn = refine_ph(ph)
                            phones.append(ph)
                            tones.append(tn)
                        else:
                            phones.append(ph)
                            tones.append(0)
                        phone_len += 1
                except Exception:
                    # Si g2p_en falla, usar fallback simple
                    phone_list = _simple_g2p_fallback(w)
                    for ph in phone_list:
                        if ph in arpa:
                            ph, tn = refine_ph(ph)
                            phones.append(ph)
                            tones.append(tn)
                        else:
                            phones.append(ph)
                            tones.append(0)
                        phone_len += 1
            else:
                # TERCERO: Fallback simple sin g2p_en
                phone_list = _simple_g2p_fallback(w)
                for ph in phone_list:
                    if ph in arpa:
                        ph, tn = refine_ph(ph)
                        phones.append(ph)
                        tones.append(tn)
                    else:
                        phones.append(ph)
                        tones.append(0)
                    phone_len += 1
        
        aaa = distribute_phone(phone_len, word_len)
        word2ph += aaa
    
    phones = [post_replace_ph(i) for i in phones]

    if pad_start_end:
        phones = ["_"] + phones + ["_"]
        tones = [0] + tones + [0]
        word2ph = [1] + word2ph + [1]
    
    return phones, tones, word2ph


def get_bert_feature(text, word2ph, device=None):
    from text import english_bert
    return english_bert.get_bert_feature(text, word2ph, device=device)


if __name__ == "__main__":
    from text.english_bert import get_bert_feature
    text = "In this paper, we propose 1 DSPGAN, a N-F-T GAN-based universal vocoder."
    text = text_normalize(text)
    phones, tones, word2ph = g2p(text)
    bert = get_bert_feature(text, word2ph)
    print(phones, tones, word2ph, bert.shape)
