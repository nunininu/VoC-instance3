import os
from dotenv import load_dotenv
import psycopg2
import pandas as pd
from gensim.models import Word2Vec
from gensim import corpora, models
import networkx as nx

# .env 로드
load_dotenv()
conn = psycopg2.connect(
    host=os.getenv("DB_HOST"),
    port=os.getenv("DB_PORT"),
    dbname=os.getenv("DB_NAME"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD")
)

# 데이터 불러오기
df = pd.read_sql("""
    SELECT analysis_result_id, client_id, consulting_id, keywords, positive, negative
    FROM analysis_result
    WHERE keywords IS NOT NULL AND positive IS NOT NULL AND negative IS NOT NULL
""", conn)

# 분류 라벨 부여
df["label"] = (df["positive"] > df["negative"]).astype(int)

# 부정 응답자 필터링 및 키워드 전처리
df_neg = df[df["label"] == 0].copy()
df_neg["tokenized"] = df_neg["keywords"].apply(lambda x: [w.strip() for w in x.split(",") if w.strip()])

# LDA 분석
dictionary = corpora.Dictionary(df_neg["tokenized"])
corpus = [dictionary.doc2bow(text) for text in df_neg["tokenized"]]
lda_model = models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=5)
print("🧠 LDA Topics:")
for idx, topic in lda_model.print_topics(-1):
    print(f"Topic {idx}: {topic}")

# Word2Vec 학습
w2v_model = Word2Vec(sentences=df_neg["tokenized"], vector_size=100, window=3, min_count=1, workers=4)

# 네트워크 그래프 구성
G = nx.Graph()
for kw in df_neg["tokenized"].explode().value_counts().head(5).index:
    if kw in w2v_model.wv:
        for sim_kw, score in w2v_model.wv.most_similar(kw, topn=5):
            G.add_edge(kw, sim_kw, weight=score)

# Superset용 테이블로 저장
df["keyword"] = df["keywords"].str.split(",")
exploded = df.explode("keyword")
exploded["keyword"] = exploded["keyword"].str.strip()
exploded = exploded[exploded["keyword"] != ""]

from sqlalchemy import create_engine
engine = create_engine(f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}")
exploded.to_sql("analysis_keywords_exploded", engine, index=False, if_exists="replace")
