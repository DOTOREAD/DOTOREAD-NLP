from flask import Flask, request
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

import pandas as pd
from gensim import corpora
from gensim.models import LdaModel
from konlpy.tag import Okt
import re
import json
from flask_cors import CORS
import requests
from bs4 import BeautifulSoup

app = Flask(__name__)
CORS(app)

# 전처리 관련 함수
def clean_text(text):
    """텍스트에서 특수문자와 이모지 제거"""
    text = re.sub(r'[^가-힣a-zA-Z0-9\s]', ' ', text)  # 한글, 영어, 숫자, 공백만 유지
    text = re.sub(r'\s+', ' ', text).strip()  # 공백 여러 개를 하나로 줄임
    return text

def extract_nouns(text):
    """주어진 텍스트에서 명사만 추출"""
    okt = Okt()
    tokens = okt.pos(text)
    valid_nouns = [word for word, pos in tokens if pos in ['Noun', 'Adjective']]
    return valid_nouns

def preprocess_text(text, stopwords=[]):
    """텍스트 전처리 및 명사 추출"""
    cleaned_text = clean_text(text)
    tokenized_text = extract_nouns(cleaned_text)
    tokenized_text = [word for word in tokenized_text if word not in stopwords]
    tokenized_text = [word for word in tokenized_text if len(word) > 1]
    return tokenized_text

def lda_modeling(tokenized_articles, num_topics=5):
    """토큰화된 텍스트에 대해 LDA 모델링"""
    dictionary = corpora.Dictionary(tokenized_articles)
    corpus = [dictionary.doc2bow(text) for text in tokenized_articles]
    lda_model = LdaModel(corpus=corpus, num_topics=num_topics, id2word=dictionary, passes=15)
    return lda_model, corpus, dictionary

# 가장 중요한 키워드 추출 함수
def get_top_keyword(lda_model):
    """LDA 모델에서 가장 중요한 키워드를 추출"""
    topics = lda_model.show_topics(num_words=10)
    top_keywords = [word.split("*")[1].strip('" ') for word in topics[0][1].split("+")]
    return top_keywords[0]  # 가장 중요한 키워드 반환

# 웹 크롤링 함수 추가
def crawl_url(url):
    """주어진 URL의 웹 페이지를 크롤링하여 텍스트 추출"""
    try:
        response = requests.get(url)
        response.raise_for_status()  # 오류 발생 시 예외 발생
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 텍스트 추출 (예: <p> 태그 내의 텍스트)
        paragraphs = soup.find_all('p')
        text = ' '.join([p.get_text() for p in paragraphs])
        
        return text
    except requests.RequestException as e:
        print(f"크롤링 중 오류 발생: {e}")
        return None

@app.route('/lda', methods=['POST'])
def lda_topic_extraction():
    data = request.json
    raw_text = data.get('text')
    url = data.get('url')
    stopwords = data.get('stopwords', [])

    if url:
        # URL이 제공된 경우 크롤링 수행
        raw_text = crawl_url(url)
        if raw_text is None:
            return json.dumps({"error": "크롤링 실패"}, ensure_ascii=False), 400

    if not raw_text:
        return json.dumps({"error": "텍스트 또는 유효한 URL을 제공해주세요"}, ensure_ascii=False), 400

    # 전처리 수행
    preprocessed_text = preprocess_text(raw_text, stopwords)

    # LDA 모델링 수행
    lda_model, corpus, dictionary = lda_modeling([preprocessed_text])

    # 가장 적절한 키워드 추출
    top_keyword = get_top_keyword(lda_model)

    # ensure_ascii=False로 한국어를 그대로 반환
    return json.dumps({"topic": top_keyword}, ensure_ascii=False)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001)