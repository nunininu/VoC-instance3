import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from pyvis.network import Network
from jinja2 import Template

# ==============================
# 설정값
# ==============================
NUM_TOPICS = 5
TOPN = 10
MIN_DF = 2
OUTPUT_HTML = "lda_topic_network.html"

# ==============================
# 템플릿 패치
# ==============================
import pyvis.network
template_path = os.path.join(os.path.dirname(pyvis.network.__file__), "templates", "template.html")
with open(template_path, "r", encoding="utf-8") as f:
    html_template = Template(f.read())

# ==============================
# 환경변수 로드 및 DB 연결
# ==============================
print("📦 환경변수 로드 중...")
os.environ.clear()
load_dotenv(dotenv_path=".env", override=True)

DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME")

engine = create_engine(f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}")

# ==============================
# 키워드 데이터 불러오기
# ==============================
print("📥 LDA 학습용 키워드 로딩 중...")
query = "SELECT keywords FROM dummy_analysis_result WHERE keywords IS NOT NULL AND keywords != ''"
df = pd.read_sql(query, engine)
documents = df["keywords"].dropna().tolist()

# ==============================
# CountVectorizer + LDA 학습
# ==============================
print("🧠 LDA 학습 중...")
vectorizer = CountVectorizer(tokenizer=lambda x: x.split(), min_df=MIN_DF)
X = vectorizer.fit_transform(documents)

lda_model = LatentDirichletAllocation(n_components=NUM_TOPICS, random_state=42)
lda_model.fit(X)

feature_names = vectorizer.get_feature_names_out()

# ==============================
# 네트워크 시각화
# ==============================
print("🖼️ 토픽별 그래프 생성 중...")
net = Network(height="800px", width="100%", bgcolor="#222222", font_color="white", notebook=False)
net.template = html_template
net.repulsion(node_distance=180, spring_length=100, spring_strength=0.03)

added_nodes = set()
topic_terms = lda_model.show_topics(num_topics=NUM_TOPICS, num_words=TOPN, formatted=False)

for topic_idx, terms in topic_terms:
    topic_node = f"Topic {topic_idx}"
    net.add_node(topic_node, label=topic_node, color="lightblue", size=25)

    for word, score in terms:
        if score < 0.001:
            continue
        word_node = word.strip()
        if word_node not in added_nodes:
            net.add_node(word_node, label=word_node, color="white", size=15)
            added_nodes.add(word_node)
        net.add_edge(topic_node, word_node, value=float(score))

# ==============================
# 저장
# ==============================
net.save_graph(OUTPUT_HTML)
print(f"✅ {OUTPUT_HTML} 저장 완료")
