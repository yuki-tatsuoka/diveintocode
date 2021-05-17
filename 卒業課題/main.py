from keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from transformers import AutoTokenizer, AutoModelWithLMHead
import streamlit as st
import streamlit.components.v1 as stc
import numpy as np
import MeCab 
import torch
import re
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.patheffects as path_effects
from matplotlib.font_manager import FontProperties

# 色変更class
class Color:
    BLACK          = '\033[30m'#(文字)黒
    RED            = '\033[31m'#(文字)赤
    GREEN          = '\033[32m'#(文字)緑
    YELLOW         = '\033[33m'#(文字)黄
    BLUE           = '\033[34m'#(文字)青
    MAGENTA        = '\033[35m'#(文字)マゼンタ
    CYAN           = '\033[36m'#(文字)シアン
    WHITE          = '\033[37m'#(文字)白
    COLOR_DEFAULT  = '\033[39m'#文字色をデフォルトに戻す
    BOLD           = '\033[1m'#太字
    UNDERLINE      = '\033[4m'#下線
    INVISIBLE      = '\033[08m'#不可視
    REVERCE        = '\033[07m'#文字色と背景色を反転
    BG_BLACK       = '\033[40m'#(背景)黒
    BG_RED         = '\033[41m'#(背景)赤
    BG_GREEN       = '\033[42m'#(背景)緑
    BG_YELLOW      = '\033[43m'#(背景)黄
    BG_BLUE        = '\033[44m'#(背景)青
    BG_MAGENTA     = '\033[45m'#(背景)マゼンタ
    BG_CYAN        = '\033[46m'#(背景)シアン
    BG_WHITE       = '\033[47m'#(背景)白
    BG_DEFAULT     = '\033[49m'#背景色をデフォルトに戻す
    RESET          = '\033[0m'#全てリセット


 
# bertサブワード削除
word_level_tokens = []    

def joint_sub_word_tokens(tokens):
    for token in tokens:
        if token.startswith('##'):
            token = re.sub("^##", "", token)
            word_level_tokens[-1]+=token
        else:
            word_level_tokens.append(token)
    return word_level_tokens


# テキストエリアの入力
st.title('文章校正ツール')
placeholder = st.empty()
text_area = placeholder.text_area('誤字脱字を検出したい文章を入力してください')

# ボタンのリセット
botton = st.button('ボタン', key=1)
clear = st.button('クリア')

if clear == True:
    text_area = placeholder.text_area('誤字脱字を検出したい文章を入力してください', value='', key=1)
else:
    pass


# モデル
tokenizer = AutoTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
model = AutoModelWithLMHead.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')

#入力されたテキストを形態素化
if botton == True:
    area = text_area.split('\n')
    for i,sub_text in enumerate(area):
        tokenized = tokenizer.tokenize(sub_text) # 後半のif文でoredictionを検出するために利用す
        tokens = tokenizer.convert_tokens_to_ids(tokenized)
        len_text = len(tokens) # mask処理を行うための回数
            
        for masked_index in range(len_text):
            # 形態素化
            tokenized_text = tokenizer.tokenize(sub_text) # 形態素化        
            tokenized_text[masked_index] = '[MASK]' # mask化
            # id化 
            index_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
            tokens_tensor = torch.tensor([index_tokens])

            with torch.no_grad():           
                outputs = model(tokens_tensor)
                predictions = outputs[0][0, masked_index].topk(50).indices# 次元は(1,14,32000)
                if tokens[masked_index] not in predictions:
                    word_color = tokenizer.convert_ids_to_tokens(tokens[masked_index]) # 指摘されたID
                    #change_color = f'<font color="#ff4500">{word_color}</font><nobr>'# 色変更したID
                    change_color = f'<nobr><span style="color:#ff0000;">{word_color}</span></nobr>'                      
                    pre_sentence = tokenizer.convert_ids_to_tokens(tokens) # 元の形態素化された文章
                    pre_sentence[masked_index] = change_color
                    #st.write(pre_sentence)
                    result = joint_sub_word_tokens(pre_sentence)
                    #st.text_area('出力結果', value=''.join(result))
                    #st.write(result)
                    stc.html(''.join(result), height=80)
                    #st.write(b)

     


