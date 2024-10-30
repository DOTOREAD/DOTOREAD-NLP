from flask import Flask, request, jsonify
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
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# 불용어 리스트
def load_stopwords(filepath="stopwordlist.txt"):
    with open(filepath, "r", encoding="utf-8") as file:
        stopwords = file.read().splitlines()
    return stopwords

default_stopwords = load_stopwords() 

# 웹 크롤링 함수
def crawl_website(url):
    """주어진 URL의 웹사이트를 크롤링하여 제목과 텍스트 추출"""
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # 페이지 제목 추출
    title = soup.title.string if soup.title else ""
    
    # 주요 콘텐츠가 있는 태그를 선택 (예: <p>, <article> 등)
    text_elements = soup.find_all(['p', 'article', 'div', 'span'])
    
    # 추출된 텍스트를 하나의 문자열로 결합
    text = ' '.join([elem.get_text() for elem in text_elements])
    
    # 제목과 본문 텍스트를 결합
    full_text = title + " " + text
    
    return full_text

# 제목 크롤링 함수
def get_website_title(url):
    """주어진 URL의 웹사이트 제목을 추출"""
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    title = soup.title.string if soup.title else ""
    return title

# 이미지 추출 함수
def get_image_url(url):
    """URL의 본문에서 첫 번째 이미지 URL을 추출"""
    response = requests.get(url)

    response = requests.get(url) # url 본문에서 외부 url을 가진 첫 번째 이미지 url만 가져옴
    soup = BeautifulSoup(response.content, 'html.parser')

    content = soup.find('article') or soup.find('div', class_='content') or soup.find('div', id='main-content')

    # 본문 영역이 없을 경우
    if not content:
        return None

    # 첫번째 이미지 URL 추출
    for img_tag in content.find_all('img'):
        img_url = img_tag.get('src')
        if img_url and (img_url.startswith('http://') or img_url.startswith('https://')):
            return img_url

    return None 

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

def preprocess_text(text, stopwords=default_stopwords):
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

@app.route('/keyword', methods=['POST'])
def lda_topic_extraction():
    url = request.json.get('url')
    if not url:
        return json.dumps({"error": "URL이 필요합니다."}, ensure_ascii=False), 400

    stopwords = request.json.get('stopwords', default_stopwords)

    # URL에서 텍스트 크롤링 (제목 포함)
    raw_text = crawl_website(url)

    # 전처리 수행
    preprocessed_text = preprocess_text(raw_text, stopwords)

    # LDA 모델링 수행
    lda_model, corpus, dictionary = lda_modeling([preprocessed_text])

    # 가장 적절한 키워드 추출
    top_keyword = get_top_keyword(lda_model)

    # ensure_ascii=False로 한국어를 그대로 반환
    return json.dumps({"topic": top_keyword}, ensure_ascii=False)

@app.route('/title', methods=['POST'])
def get_title():
    # 요청에서 URL 가져오기
    url = request.json.get('url')
    if not url:
        return json.dumps({"error": "URL이 필요합니다."}, ensure_ascii=False), 400

    title = get_website_title(url)
    return json.dumps({"title": title}, ensure_ascii=False)

@app.route('/images', methods=['POST'])
def image_urls():
    blog_url = request.json.get('url')
    if not blog_url:
        return jsonify({"error": "URL이 필요합니다."}), 400

    image_url = get_image_url(blog_url)
    return jsonify({"image_url": image_url})


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001)