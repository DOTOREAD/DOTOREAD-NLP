from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer, util
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from bs4 import BeautifulSoup
import requests
import re

model = SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS")

app = Flask(__name__)

def load_stopwords(filepath="stopwordlist.txt"):
    try:
        with open(filepath, "r", encoding="utf-8") as file:
            stopwords = file.read().splitlines()
        return stopwords
    except Exception as e:
        print(f"불용어 로드 실패: {e}")
        return []

stopwords = load_stopwords()

def crawl_website(url):
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
        return f"폴더 추출에 실패했습니다. {str(e)}"

@app.route('/classify', methods=['POST'])
def recommend_folder():
    try:
        data = request.json
        url = data.get("url")
        existing_folders = data.get("folders", [])
        if not url or not existing_folders:
            return jsonify({"error": "URL과 기존 폴더 목록을 모두 제공해야 합니다."}), 400

        raw_text = crawl_website(url)
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
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.set_start_method("fork", force=True)

    app.run(host="0.0.0.0", port=5001)