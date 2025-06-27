print("▶ 실행 시작: 03_lda_topic_modeling.py")
print("▶ 실행 시작: 03_lda_topic_modeling.py")
print("🧠 LDA ▶ 실행 시작: 03_lda_topic_modeling.py")
# 03_lda_topic_modeling.py
import pandas as pd
print("📥 데이터 로딩 시작...")
import gensim
from gensim import corpora
print(f"✅ 응답 수: {len(df)}")
print("📥 데이터 로딩 중...")
from sqlalchemy import create_engine
print("✂️ 텍스트 전처리 중...")
from dotenv import load_dotenv
import os
print("🧠 LDA 모델 학습 시작...")

print("✂️ 전처리 중...")
print("📊 토픽 정리 중...")
load_dotenv(".env")
engine = create_engine(
print("🗃 DB 저장 중...")
    f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
)
print("🧠 LDA 학습 중...")

df = pd.read_sql("SELECT client_id, predicted_label, keywords FROM classified_sentiment JOIN analysis_result USING(client_id)", engine)
print("📥 classified_complaint 테이블 로딩 중...")
print("📥 불만 응답 로딩 중...")
print("✂️ 전처리 후 샘플 문서:", processed_docs[:2])
print("📥 데이터 로딩 시작...")
df = df[df['predicted_label'] == 1].copy()
df['keywords'] = df['keywords'].apply(lambda x: [kw.strip() for kw in x.split(',')])
print("📊 토픽 정리 및 저장 중...")

dictionary = corpora.Dictionary(df['keywords'])
corpus = [dictionary.doc2bow(text) for text in df['keywords']]
lda = gensim.models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=5, passes=10)
print("🧠 LDA 모델 학습 중...")

topic_results = []
for i, row in enumerate(corpus):
    topics = lda.get_document_topics(row)
    top_topic = max(topics, key=lambda x: x[1])[0]
    topic_results.append((df.iloc[i]['client_id'], top_topic))

topic_df = pd.DataFrame(topic_results, columns=['client_id', 'topic'])
topic_df.to_sql("lda_topic", engine, index=False, if_exists="replace")
print("📊 주요 토픽 결과 테이블 저장 중...")
print("🗃 저장 전 미리보기:")
print(df_topic.head(3))
print(f"✅ 불만 응답 수: {len(df)}")
print("🧠 LDA ✅ 실행 완료: 03_lda_topic_modeling.py")
print("✅ 실행 완료: 03_lda_topic_modeling.py")
print("✅ 실행 완료: 03_lda_topic_modeling.py")
print("✅ 실행 완료: 03_lda_topic_modeling.py")