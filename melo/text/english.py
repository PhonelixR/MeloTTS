import pickle
import os
import re
import nltk
import warnings
from g2p_en import G2p

from . import symbols

from .english_utils.abbreviations import expand_abbreviations
from .english_utils.time_norm import expand_time_english
from .english_utils.number_norm import normalize_numbers
from .japanese import distribute_phone

from transformers import AutoTokenizer

current_file_path = os.path.dirname(__file__)
CMU_DICT_PATH = os.path.join(current_file_path, "cmudict.rep")
CACHE_PATH = os.path.join(current_file_path, "cmudict_cache.pickle")

# ====== SOLUCIÓN DEFINITIVA PARA G2P_EN 2.1.0 ======
def _fix_nltk_resource():
    """Arregla el problema de compatibilidad entre g2p_en 2.1.0 y NLTK"""
    try:
        # Descargar el recurso correcto
        nltk.download('averaged_perceptron_tagger', quiet=True)
        
        # Encontrar la ruta del recurso
        resource_path = nltk.data.find('taggers/averaged_perceptron_tagger')
        
        # g2p_en 2.1.0 busca 'averaged_perceptron_tagger_eng' en lugar de 'averaged_perceptron_tagger'
        # Crear un alias simbólico si es posible
        import shutil
        from pathlib import Path
        
        # Obtener el directorio padre
        parent_dir = Path(resource_path).parent
        target_name = 'averaged_perceptron_tagger_eng'
        target_path = parent_dir / target_name
        
        if not target_path.exists():
            # Crear una copia con el nombre que espera g2p_en
            if Path(resource_path).is_dir():
                shutil.copytree(resource_path, target_path)
            else:
                shutil.copy2(resource_path, target_path)
            print(f"✅ Creado alias para recurso NLTK: {target_name}")
            
    except Exception as e:
        warnings.warn(f"⚠️ No se pudo arreglar recurso NLTK: {e}")

def _init_g2p():
    """Inicialización robusta de G2p con compatibilidad completa"""
    try:
        # Descargar todos los recursos necesarios
        resources = ['averaged_perceptron_tagger', 'punkt', 'cmudict']
        for resource in resources:
            try:
                if resource == 'averaged_perceptron_tagger':
                    nltk.data.find(f'taggers/{resource}')
                else:
                    nltk.data.find(f'tokenizers/{resource}' if resource == 'punkt' else f'corpora/{resource}')
            except LookupError:
                print(f"📥 Descargando recurso NLTK: {resource}")
                nltk.download(resource, quiet=True)
        
        # Intentar arreglar el problema de compatibilidad
        _fix_nltk_resource()
        
        # Inicializar G2p
        g2p_instance = G2p()
        
        # Probar que funciona
        test_result = g2p_instance("test")
        return g2p_instance
        
    except Exception as e:
        warnings.warn(f"⚠️ No se pudo inicializar g2p_en: {e}")
        print("\n💡 **SOLUCIÓN MANUAL:** Ejecuta esto en Colab:")
        print("""
import nltk
import shutil
from pathlib import Path

# Descargar recurso
nltk.download('averaged_perceptron_tagger')

# Crear alias
resource_path = nltk.data.find('taggers/averaged_perceptron_tagger')
parent_dir = Path(resource_path).parent
target_path = parent_dir / 'averaged_perceptron_tagger_eng'

if not target_path.exists():
    if Path(resource_path).is_dir():
        shutil.copytree(resource_path, target_path)
    else:
        shutil.copy2(resource_path, target_path)
    print("✅ Recurso NLTK arreglado")
""")
        return None

# Inicializar G2p
_g2p = _init_g2p()

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

def g2p_old(text):
    tokenized = tokenizer.tokenize(text)
    phones = []
    tones = []
    words = re.split(r"([,;.\-\?\!\s+])", text)
    for w in words:
        if w.upper() in eng_dict:
            phns, tns = refine_syllables(eng_dict[w.upper()])
            phones += phns
            tones += tns
        else:
            if _g2p is None:
                raise RuntimeError("❌ G2p no está inicializado. Ejecuta: pip install g2p_en==2.1.0")
            phone_list = list(filter(lambda p: p != " ", _g2p(w)))
            for ph in phone_list:
                if ph in arpa:
                    ph, tn = refine_ph(ph)
                    phones.append(ph)
                    tones.append(tn)
                else:
                    phones.append(ph)
                    tones.append(0)
    word2ph = [1 for _ in phones]

    phones = [post_replace_ph(i) for i in phones]
    return phones, tones, word2ph


def g2p(text, pad_start_end=True, tokenized=None):
    if _g2p is None:
        raise RuntimeError("❌ G2p no está inicializado. Ejecuta: pip install g2p_en==2.1.0")
    
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
        if w.upper() in eng_dict:
            phns, tns = refine_syllables(eng_dict[w.upper()])
            phones += phns
            tones += tns
            phone_len += len(phns)
        else:
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
            except Exception as e:
                # Solo mostrar warning para palabras significativas, no puntuación
                if w.strip() and w not in ['-', '.', ',', '!', '?', ';', ':']:
                    warnings.warn(f"⚠️ Error en g2p para palabra '{w}': {e}")
                # Fallback: agregar un fonema UNK
                phones.append("UNK")
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
