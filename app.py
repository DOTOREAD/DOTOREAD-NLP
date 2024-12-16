from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer, util
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from bs4 import BeautifulSoup

import re
import json
import requests
from flask_cors import CORS

model = SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS")

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

def load_stopwords(filepath="stopwordlist.txt"):
    with open(filepath, "r", encoding="utf-8") as file:
        stopwords = file.read().splitlines()
    return stopwords

stopwords = load_stopwords()

def error_response(message, status_code=400):
    return jsonify({
        "error": message,
        "status_code": status_code
    }), status_code

def crawl_website_title(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    title = soup.title.string if soup.title else ""
    return title

def crawl_website_text(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    text_elements = soup.find_all(['p', 'article', 'div', 'span'])
    text = ' '.join([elem.get_text() for elem in text_elements])
    return text

def preprocess_text(text, stopwords):
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'[^가-힣a-zA-Z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    filtered_words = [word for word in text.split() if word not in stopwords]
    return ' '.join(filtered_words)

def get_image_url(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')

        img_tags = soup.find_all('img')

        image_urls = []
        for img_tag in img_tags:
            img_url = img_tag.get('src')
            if img_url:
                if not (img_url.startswith('http://') or img_url.startswith('https://')):
                    img_url = requests.compat.urljoin(url, img_url)

                if any(keyword in img_url.lower() for keyword in ['icon', 'logo', 'favicon', 'small']):
                    continue

                if img_url.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.svg')):
                    width = img_tag.get('width')
                    height = img_tag.get('height')
                    if width and height and (int(width) < 50 or int(height) < 50):  # 크기 제한
                        continue

                    image_urls.append(img_url)

        if not image_urls:
            return None

        if len(image_urls) > 3:
            return image_urls[len(image_urls) // 2]
        elif len(image_urls) > 1:
            return image_urls[-1]
        else:
            return image_urls[0]

    except Exception as e:
        return None

def extract_new_topics(text, num_topics=1):
    try:
        vectorizer = CountVectorizer(stop_words=stopwords, max_df=1.0, min_df=1)
        text_vectorized = vectorizer.fit_transform([text])

        lda_model = LatentDirichletAllocation(n_components=num_topics, random_state=42)
        lda_model.fit(text_vectorized)

        topic = lda_model.components_[0]
        topic_keyword = vectorizer.get_feature_names_out()[topic.argsort()[-1]]
        return topic_keyword

    except Exception as e:
            return error_response(f"폴더 추출에 실패했습니다. {str(e)}", 500)

@app.route('/title', methods=['POST'])
def get_title():
    url = request.json.get('url')
    if not url:
        return error_response("URL이 필요합니다.", 400)

    title = crawl_website_title(url)
    return jsonify({"title": title})

@app.route('/images', methods=['POST'])
def image_urls():
    blog_url = request.json.get('url')
    if not blog_url:
        return error_response("URL이 필요합니다.", 400)

    image_url = get_image_url(blog_url)
    return jsonify({"image_url": image_url})

@app.route('/classify', methods=['POST'])
def recommend_folder():
    try:
        data = request.json
        url = data.get("url")
        existing_folders = data.get("folders", [])
        if not url or not existing_folders:
            return error_response("URL과 기존 폴더 목록을 모두 제공해야 합니다.", 400)
        raw_text = crawl_website_text(url)
        if not raw_text.strip():
            return jsonify({
                "전처리 텍스트": "",
                "폴더 분류": None,
                "폴더 분류 유사도": 0,
                "새 폴더 추천": None,
                "error": "크롤링된 텍스트가 비어 있습니다."
            }), 200

        processed_text = preprocess_text(raw_text, stopwords)

        text_embedding = model.encode(processed_text, convert_to_tensor=True)
        folder_embeddings = model.encode(existing_folders, convert_to_tensor=True)
        cosine_scores = util.pytorch_cos_sim(text_embedding, folder_embeddings)

        best_match_index = cosine_scores.argmax()
        best_match_score = cosine_scores[0][best_match_index].item()
        best_match_folder = existing_folders[best_match_index]

        threshold = 0.15
        if best_match_score < threshold:
            new_topics = extract_new_topics(processed_text)
            return jsonify({
                "전처리 텍스트": processed_text,
                "폴더 분류": None,
                "폴더 분류 유사도": best_match_score,
                "새 폴더 추천": new_topics
            })
        else:
            return jsonify({
                "전처리 텍스트": processed_text,
                "폴더 분류": best_match_folder,
                "폴더 분류 유사도": best_match_score,
                "새 폴더 추천": None
            })
    except Exception as e:
            return error_response(f"서버 내부 오류: {str(e)}", 500)

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.set_start_method("fork", force=True)

    app.run(host="0.0.0.0", port=5001)