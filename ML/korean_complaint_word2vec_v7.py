import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
from pyvis.network import Network
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
import re
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from sqlalchemy import create_engine
from dotenv import load_dotenv

# DB 연결
load_dotenv()
DB_URL = f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@" \
         f"{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
engine = create_engine(DB_URL)


# 한글 폰트 설정
plt.rcParams['font.family'] = 'NanumSquareRound'
plt.rcParams['axes.unicode_minus'] = False

class Word2VecTrainingCallback(CallbackAny2Vec):
    """Word2Vec 학습 진행상황을 모니터링하는 콜백"""
    def __init__(self):
        self.epoch = 0
        self.losses = []

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        self.losses.append(loss)
        self.epoch += 1
        if self.epoch % 10 == 0:
            print(f'Epoch {self.epoch}: Loss = {loss}')

class WikiComplaintWord2VecSystem:
    def __init__(self):
        self.base_model = None  # 위키피디아 기반 사전학습 모델
        self.complaint_model = None  # 고객불만 추가학습 모델
        self.vocabulary = None
        self.word_vectors = None
        load_dotenv()
        DB_URL = f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@" \
                 f"{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
        self.engine = create_engine(DB_URL)

        
    def setup_mecab(self):
        """MeCab 설치 및 설정 (Colab 환경용)"""
        try:
            from mecab import MeCab
            self.mecab = MeCab()
            print("MeCab이 이미 설치되어 있습니다.")
            return True
        except:
            print("MeCab 설치가 필요합니다. 다음 명령어를 실행하세요:")
            print("!git clone https://github.com/SOMJANG/Mecab-ko-for-Google-Colab.git")
            print("!cd Mecab-ko-for-Google-Colab && bash install_mecab-ko_on_colab190912.sh")
            return False
    
    def download_and_process_wikipedia(self, force_download=False):
        """위키피디아 데이터 다운로드 및 전처리"""
        print("=== 위키피디아 데이터 다운로드 ===")
        
        import subprocess
        import os
        
        # 파일 존재 확인
        wiki_dump_file = "kowiki-latest-pages-articles.xml.bz2"
        extracted_dir = "text"
        
        if not force_download and os.path.exists(extracted_dir):
            print(f"추출된 위키피디아 데이터가 이미 존재합니다: {extracted_dir}")
            return True
        
        try:
            # 1. wikiextractor 설치
            print("1. wikiextractor 설치 중...")
            subprocess.run(["pip", "install", "wikiextractor"], check=False)
            
            # 2. 위키피디아 덤프 다운로드 (파일이 없는 경우만)
            if not os.path.exists(wiki_dump_file):
                print("2. 위키피디아 덤프 다운로드 중... (시간이 오래 걸릴 수 있습니다)")
                download_cmd = [
                    "wget", 
                    "https://dumps.wikimedia.org/kowiki/latest/kowiki-latest-pages-articles.xml.bz2"
                ]
                result = subprocess.run(download_cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    print(f"다운로드 실패: {result.stderr}")
                    print("수동으로 다음 명령어를 실행하세요:")
                    print("wget https://dumps.wikimedia.org/kowiki/latest/kowiki-latest-pages-articles.xml.bz2")
                    return False
            else:
                print(f"2. 위키피디아 덤프 파일이 이미 존재합니다: {wiki_dump_file}")
            
            # 3. 텍스트 추출
            if not os.path.exists(extracted_dir):
                print("3. 위키피디아 텍스트 추출 중...")
                extract_cmd = [
                    "python", "-m", "wikiextractor.WikiExtractor", 
                    wiki_dump_file,
                    "--output", extracted_dir,
                    "--bytes", "100M",
                    "--processes", "4"
                ]
                result = subprocess.run(extract_cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    print(f"텍스트 추출 실패: {result.stderr}")
                    return False
            else:
                print(f"3. 추출된 텍스트가 이미 존재합니다: {extracted_dir}")
            
            print("위키피디아 데이터 처리 완료!")
            return True
            
        except Exception as e:
            print(f"위키피디아 데이터 처리 중 오류 발생: {e}")
            print("\n수동 설치 방법:")
            print("1. pip install wikiextractor")
            print("2. wget https://dumps.wikimedia.org/kowiki/latest/kowiki-latest-pages-articles.xml.bz2")
            print("3. python -m wikiextractor.WikiExtractor kowiki-latest-pages-articles.xml.bz2")
            return False
    
    def list_wiki_files(self, dirname='text'):
        """위키피디아 텍스트 파일들의 경로 리스트 생성"""
        def list_wiki_recursive(dirname):
            filepaths = []
            if not os.path.exists(dirname):
                return filepaths
                
            filenames = os.listdir(dirname)
            for filename in filenames:
                filepath = os.path.join(dirname, filename)
                if os.path.isdir(filepath):
                    # 재귀 함수
                    filepaths.extend(list_wiki_recursive(filepath))
                else:
                    find = re.findall(r"wiki_[0-9][0-9]", filepath)
                    if len(find) > 0:
                        filepaths.append(filepath)
            return sorted(filepaths)
        
        return list_wiki_recursive(dirname)
    
    def merge_wiki_files(self, filepaths, output_file="wiki_merged.txt"):
        """위키피디아 파일들을 하나로 통합"""
        print(f"총 {len(filepaths)}개의 파일을 통합합니다...")
        
        with open(output_file, "w", encoding="utf-8") as outfile:
            for filepath in tqdm(filepaths, desc="파일 통합"):
                try:
                    with open(filepath, encoding="utf-8") as infile:
                        contents = infile.read()
                        outfile.write(contents)
                except Exception as e:
                    print(f"파일 읽기 오류: {filepath}, {e}")
        
        print(f"통합 완료: {output_file}")
        return output_file
    
    def preprocess_wiki_text(self, wiki_file="wiki_merged.txt", max_lines=100000):
        """위키피디아 텍스트 전처리 및 형태소 분석"""
        if not hasattr(self, 'mecab'):
            print("MeCab이 설치되지 않았습니다.")
            return []
        
        print("위키피디아 텍스트 전처리 시작...")
        
        # 메모리 효율을 위해 스트리밍 방식으로 처리
        result = []
        line_count = 0
        
        try:
            with open(wiki_file, 'r', encoding='utf-8') as f:
                print(f"최대 {max_lines}개 줄까지 처리합니다...")
                
                # 진행 상황 표시를 위한 카운터
                processed_count = 0
                
                for line in f:
                    if line_count >= max_lines:
                        print(f"최대 처리 줄 수({max_lines})에 도달했습니다.")
                        break
                    
                    line_count += 1
                    
                    # 빈 문자열이 아니고, XML 태그가 아닌 경우에만 처리
                    line = line.strip()
                    if line and not line.startswith('<') and not line.startswith('</'):
                        try:
                            morphs = self.mecab.morphs(line)
                            # 최소 3개 이상의 형태소가 있고, 길이가 적절한 경우만
                            if len(morphs) >= 3 and len(morphs) <= 100:
                                # 불용어와 특수문자 제거
                                filtered_morphs = [
                                    word for word in morphs 
                                    if len(word) > 1 and word.isalnum()
                                ]
                                if len(filtered_morphs) >= 2:
                                    result.append(filtered_morphs)
                                    processed_count += 1
                        except Exception as e:
                            continue
                    
                    # 진행 상황 출력
                    if line_count % 10000 == 0:
                        print(f"처리된 줄: {line_count:,}, 유효한 문장: {processed_count:,}")
                
        except Exception as e:
            print(f"파일 처리 중 오류 발생: {e}")
            return []
        
        print(f"전처리 완료: 총 {line_count:,}줄 처리, {len(result):,}개의 유효한 문장 생성")
        return result
    
    def train_base_wiki_model(self, processed_sentences, model_path="wiki_word2vec.model"):
        """위키피디아 데이터로 기본 Word2Vec 모델 학습"""
        print("=== 위키피디아 Word2Vec 모델 학습 시작 ===")
        
        # Word2Vec 모델 설정 (원본 예시와 동일한 파라미터)
        self.base_model = Word2Vec(
            sentences=processed_sentences,
            vector_size=100,  # size -> vector_size (최신 버전)
            window=5,
            min_count=5,
            workers=4,
            sg=0  # CBOW 모델 사용
        )
        
        print(f"모델 학습 완료!")
        print(f"어휘 크기: {len(self.base_model.wv.key_to_index)}")
        
        # 모델 저장
        self.base_model.save(model_path)
        print(f"모델 저장 완료: {model_path}")
        
        # 학습 결과 예시 출력
        self.test_base_model()
        
        return self.base_model
    
    def test_base_model(self):
        """기본 모델 테스트"""
        print("\n=== 위키피디아 모델 테스트 ===")
        
        test_words = ["대한민국", "한국", "서울", "부산", "정치", "경제", "문화", "기술"]
        
        for word in test_words:
            if word in self.base_model.wv.key_to_index:
                try:
                    similar_words = self.base_model.wv.most_similar(word, topn=5)
                    print(f"\n'{word}'와 유사한 단어들:")
                    for sim_word, score in similar_words:
                        print(f"  {sim_word}: {score:.3f}")
                except:
                    print(f"'{word}' 유사도 계산 실패")
    
    def load_base_model(self, model_path="wiki_word2vec.model"):
        """사전 학습된 위키피디아 모델 로드"""
        try:
            self.base_model = Word2Vec.load(model_path)
            print(f"사전 학습된 모델 로드 완료: {model_path}")
            print(f"어휘 크기: {len(self.base_model.wv.key_to_index)}")
            return True
        except Exception as e:
            print(f"모델 로드 실패: {e}")
            return False
    
    def load_complaints_from_db(self, limit=10000):
        query = f"""
            SELECT c.content
            FROM consulting c
            JOIN analysis_result ar ON c.consulting_id = ar.consulting_id
            WHERE ar.is_negative = TRUE
            AND c.content IS NOT NULL
            AND LENGTH(c.content) > 5
            LIMIT {limit}
        """
        df = pd.read_sql(query, self.engine)
        return pd.DataFrame({'complaint': df['content'].tolist()})



    # def create_sample_complaint_data(self):
    #     """고객 불만 샘플 데이터 생성"""
    #     sample_complaints = [
    #         "배송이 너무 늦어서 불만입니다. 약속한 날짜보다 일주일이나 지연되었어요.",
    #         "상품 품질이 기대와 달라서 실망했습니다. 사진과 실제 제품이 너무 달라요.",
    #         "고객서비스 직원의 응대가 불친절했습니다. 문제 해결도 제대로 안 해주네요.",
    #         "환불 처리가 너무 복잡하고 시간이 오래 걸립니다. 간단하게 처리해 주세요.",
    #         "웹사이트 주문 시스템에 오류가 많아서 불편합니다. 결제도 제대로 안 되고요.",
    #         "상품 포장이 엉망이어서 제품이 손상되었습니다. 포장 상태를 개선해 주세요.",
    #         "가격 대비 품질이 떨어집니다. 이 가격이면 더 좋은 품질을 기대했어요.",
    #         "AS 서비스가 부실합니다. 수리 기간도 너무 길고 비용도 비싸요.",
    #         "직원 교육이 부족한 것 같습니다. 제품에 대한 정보도 제대로 모르네요.",
    #         "배송 추적 시스템이 정확하지 않습니다. 실제 배송 상황과 다르게 표시되어요.",
    #         "상품 설명이 부정확해서 잘못 주문했습니다. 정확한 정보를 제공해 주세요.",
    #         "반품 정책이 너무 까다롭습니다. 고객 입장을 좀 더 고려해 주세요.",
    #         "전화 상담 대기시간이 너무 깁니다. 빠른 응답을 원합니다.",
    #         "제품 하자가 있는데 교환이 어렵다고 하네요. 품질 관리 좀 해주세요.",
    #         "온라인 주문과 실제 받은 상품이 다릅니다. 주문 시스템을 점검해 주세요.",
    #         "매장 직원이 제품 지식이 부족해서 제대로 설명을 못해주네요.",
    #         "결제 시스템 오류로 인해 중복 결제가 되었습니다. 빠른 환불 바랍니다.",
    #         "상품 재고 관리가 엉망입니다. 주문 후에 품절이라고 하네요.",
    #         "배송비가 너무 비쌉니다. 합리적인 배송비 정책이 필요해요.",
    #         "고객센터 운영시간이 너무 짧습니다. 24시간 서비스를 원합니다."
    #     ]
        
        # 데이터 확장
        extended_complaints = []
        for complaint in sample_complaints:
            extended_complaints.append(complaint)
            # 동일한 불만을 다른 방식으로 표현
            if "배송" in complaint:
                extended_complaints.append("물건이 언제 도착하는지 알 수가 없어요. 배송 서비스 개선이 필요합니다.")
            elif "품질" in complaint:
                extended_complaints.append("제품 퀄리티가 너무 떨어져서 돈이 아깝네요.")
            elif "서비스" in complaint:
                extended_complaints.append("직원 서비스 교육이 시급해 보입니다. 너무 불친절해요.")
        
        return pd.DataFrame({'complaint': extended_complaints})
    
    def preprocess_complaint_text(self, text):
        """고객 불만 텍스트 전처리"""
        if pd.isna(text) or not hasattr(self, 'mecab'):
            return []
        
        try:
            # MeCab 형태소 분석
            morphs = self.mecab.morphs(str(text))
            
            # 불용어 제거
            stopwords = {'은', '는', '이', '가', '을', '를', '의', '에', '에서', '로', '으로', 
                        '와', '과', '도', '만', '에게', '한테', '부터', '까지', '처럼', '같이',
                        '하다', '있다', '없다', '되다', '아니다', '그리다', '오다', '가다', '해요', '네요'}
            
            # 길이 필터링 및 불용어 제거
            words = [word for word in morphs if len(word) > 1 and word not in stopwords]
            return words
        except:
            return []
    
    def train_complaint_model(self, complaint_data, epochs=100):
        """고객 불만 데이터로 추가 학습"""
        if self.base_model is None:
            print("기본 모델이 없습니다. 먼저 위키피디아 모델을 로드하세요.")
            return None
        
        print("=== 고객 불만 데이터 전처리 ===")
        
        # 고객 불만 데이터 전처리
        processed_complaints = []
        for complaint in tqdm(complaint_data['complaint'], desc="불만 데이터 전처리"):
            words = self.preprocess_complaint_text(complaint)
            if len(words) > 1:
                processed_complaints.append(words)
        
        print(f"전처리된 불만 문장 수: {len(processed_complaints)}")
        
        # 기존 모델 복사하여 새 모델 생성
        print("=== 추가 학습 시작 ===")
        self.complaint_model = Word2Vec(
            vector_size=self.base_model.vector_size,
            window=self.base_model.window,
            min_count=1,  # 불만 데이터는 작으므로 min_count를 낮춤
            workers=4,
            sg=self.base_model.sg
        )
        
        # 기본 모델의 가중치를 새 모델에 복사
        self.complaint_model.build_vocab(processed_complaints)
        
        # 기존 어휘에 대한 벡터를 복사
        for word in self.complaint_model.wv.key_to_index:
            if word in self.base_model.wv.key_to_index:
                self.complaint_model.wv.vectors[self.complaint_model.wv.key_to_index[word]] = \
                    self.base_model.wv.vectors[self.base_model.wv.key_to_index[word]]
        
        # 추가 학습
        callback = Word2VecTrainingCallback()
        self.complaint_model.train(
            processed_complaints,
            total_examples=len(processed_complaints),
            epochs=epochs,
            callbacks=[callback]
        )
        
        print(f"추가 학습 완료!")
        print(f"최종 어휘 크기: {len(self.complaint_model.wv.key_to_index)}")
        
        # 어휘와 벡터 저장
        self.vocabulary = list(self.complaint_model.wv.key_to_index.keys())
        self.word_vectors = np.array([self.complaint_model.wv[word] for word in self.vocabulary])
        
        return processed_complaints
    
    def compare_models(self):
        """기본 모델과 불만 추가 학습 모델 비교"""
        print("\n=== 모델 비교 분석 ===")
        
        test_words = ["품질", "서비스", "배송", "가격", "직원", "고객", "문제", "불만"]
        
        for word in test_words:
            print(f"\n--- '{word}' 유사 단어 비교 ---")
            
            # 기본 모델
            if word in self.base_model.wv.key_to_index:
                try:
                    base_similar = self.base_model.wv.most_similar(word, topn=5)
                    print("위키피디아 모델:")
                    for sim_word, score in base_similar:
                        print(f"  {sim_word}: {score:.3f}")
                except:
                    print("위키피디아 모델: 유사도 계산 실패")
            else:
                print("위키피디아 모델: 단어 없음")
            
            # 불만 추가 학습 모델
            if self.complaint_model and word in self.complaint_model.wv.key_to_index:
                try:
                    complaint_similar = self.complaint_model.wv.most_similar(word, topn=5)
                    print("불만 추가학습 모델:")
                    for sim_word, score in complaint_similar:
                        print(f"  {sim_word}: {score:.3f}")
                except:
                    print("불만 추가학습 모델: 유사도 계산 실패")
            else:
                print("불만 추가학습 모델: 단어 없음")
    
    def visualize_complaint_words(self, complaint_words=None, n_words=30):
        """불만 관련 단어들의 관계 시각화"""
        if self.complaint_model is None:
            print("불만 학습 모델이 없습니다.")
            return
        
        if complaint_words is None:
            # 불만 관련 주요 단어들
            complaint_words = ['불만', '문제', '서비스', '품질', '배송', '가격', '직원', 
                             '고객', '처리', '개선', '요청', '지연', '오류', '손상', 
                             '환불', '교환', '수리', '반품', '상품', '제품']
            
            # 실제 모델에 있는 단어들만 필터링
            complaint_words = [word for word in complaint_words 
                             if word in self.complaint_model.wv.key_to_index]
        
        if len(complaint_words) < 2:
            print("시각화할 단어가 부족합니다.")
            return
        
        # 선택된 단어들의 벡터 추출
        word_vectors = np.array([self.complaint_model.wv[word] for word in complaint_words])
        
        # t-SNE 적용
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(10, len(complaint_words)-1))
        word_vectors_2d = tsne.fit_transform(word_vectors)
        
        # 시각화
        plt.figure(figsize=(15, 10))
        
        # 단어 타입별 색상 구분
        colors = []
        for word in complaint_words:
            if word in ['불만', '문제', '오류', '손상']:
                colors.append('red')  # 부정적 단어
            elif word in ['개선', '처리', '해결', '수리']:
                colors.append('green')  # 해결 관련 단어
            elif word in ['서비스', '품질', '배송']:
                colors.append('blue')  # 서비스 관련 단어
            else:
                colors.append('gray')  # 기타
        
        scatter = plt.scatter(word_vectors_2d[:, 0], word_vectors_2d[:, 1], 
                            c=colors, alpha=0.7, s=100)
        
        # 단어 라벨 추가
        for i, word in enumerate(complaint_words):
            plt.annotate(word, (word_vectors_2d[i, 0], word_vectors_2d[i, 1]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=12, alpha=0.8, fontweight='bold')
        
        plt.title('고객 불만 단어 관계 시각화 (추가학습 모델)', fontsize=16, fontweight='bold')
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.grid(True, alpha=0.3)
        
        # 범례 추가
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='red', label='부정적 단어'),
                          Patch(facecolor='green', label='해결 관련'),
                          Patch(facecolor='blue', label='서비스 관련'),
                          Patch(facecolor='gray', label='기타')]
        plt.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        import time
        import os

        # 저장 경로 지정
        save_path = "complaint_tsne_plot.png"

        # 그림 저장
        plt.savefig(save_path)

        # 인터랙티브하게 창 띄우되, 3초 후 자동 닫힘
        plt.show(block=False)
        time.sleep(3)
        plt.close()

        # 저장 완료 메시지 출력
        print(f" TSNE 시각화 이미지 저장 완료: {save_path}")
        print(f"    • 전체 경로: {os.path.abspath(save_path)}")
        
        # 반환값으로 2D 벡터와 단어 리스트 반환
        return word_vectors_2d, complaint_words
    
    def create_pyvis_network_graph(self, target_words=None, max_words=50, similarity_threshold=0.3, output_file="word_network.html"):
        """PyVis를 사용한 Force-directed 네트워크 그래프 생성"""
        if self.complaint_model is None:
            print("불만 학습 모델이 없습니다.")
            return None
        
        print("=== PyVis 네트워크 그래프 생성 ===")
        
        # 타겟 단어 설정
        if target_words is None:
            target_words = ['불만', '문제', '서비스', '품질', '배송', '가격', '직원', 
                           '고객', '처리', '개선', '요청', '지연', '오류', '손상', 
                           '환불', '교환', '수리', '반품', '상품', '제품', '만족',
                           '실망', '불편', '화나다', '답답', '빠르다', '느리다']
        
        # 모델에 실제 존재하는 단어들만 필터링
        available_words = [word for word in target_words if word in self.complaint_model.wv.key_to_index]
        
        if len(available_words) < 2:
            print("시각화할 단어가 부족합니다.")
            return None
        
        print(f"사용 가능한 단어 수: {len(available_words)}")
        
        # PyVis 네트워크 객체 생성
        net = Network(
            height="800px", 
            width="100%", 
            bgcolor="#222222", 
            font_color="white",
            directed=False
        )
        
        # 물리학 시뮬레이션 설정 (Force-directed)
        net.set_options("""
        var options = {
            "physics": {
                "enabled": true,
                "forceAtlas2Based": {
                    "gravitationalConstant": -50,
                    "centralGravity": 0.01,
                    "springLength": 100,
                    "springConstant": 0.08,
                    "damping": 0.4,
                    "avoidOverlap": 0.5
                },
                "maxVelocity": 50,
                "minVelocity": 0.1,
                "solver": "forceAtlas2Based",
                "stabilization": {
                    "enabled": true,
                    "iterations": 500,
                    "updateInterval": 25
                }
            },
            "edges": {
                "smooth": {
                    "enabled": true,
                    "type": "continuous"
                }
            }
        }
        """)
        
        # 단어별 카테고리 및 색상 정의
        word_categories = {
            '불만': {'category': 'negative', 'color': '#ff4444', 'size': 30},
            '문제': {'category': 'negative', 'color': '#ff4444', 'size': 25},
            '오류': {'category': 'negative', 'color': '#ff4444', 'size': 20},
            '손상': {'category': 'negative', 'color': '#ff4444', 'size': 20},
            '실망': {'category': 'negative', 'color': '#ff6666', 'size': 20},
            '불편': {'category': 'negative', 'color': '#ff6666', 'size': 20},
            '화나다': {'category': 'negative', 'color': '#ff6666', 'size': 15},
            '답답': {'category': 'negative', 'color': '#ff6666', 'size': 15},
            '개선': {'category': 'solution', 'color': '#44ff44', 'size': 25},
            '처리': {'category': 'solution', 'color': '#44ff44', 'size': 20},
            '해결': {'category': 'solution', 'color': '#44ff44', 'size': 20},
            '수리': {'category': 'solution', 'color': '#66ff66', 'size': 15},
            '서비스': {'category': 'service', 'color': '#4444ff', 'size': 25},
            '품질': {'category': 'service', 'color': '#4444ff', 'size': 25},
            '배송': {'category': 'service', 'color': '#6666ff', 'size': 20},
            '직원': {'category': 'service', 'color': '#6666ff', 'size': 20},
            '고객': {'category': 'neutral', 'color': '#ffff44', 'size': 25},
            '상품': {'category': 'neutral', 'color': '#ffaa44', 'size': 20},
            '제품': {'category': 'neutral', 'color': '#ffaa44', 'size': 20},
            '가격': {'category': 'neutral', 'color': '#ffaa44', 'size': 20},
            '만족': {'category': 'positive', 'color': '#44ffaa', 'size': 20},
            '빠르다': {'category': 'positive', 'color': '#44ffaa', 'size': 15},
            '느리다': {'category': 'negative', 'color': '#ff8888', 'size': 15}
        }
        
        # 노드 추가
        node_positions = {}
        for word in available_words:
            word_info = word_categories.get(word, {'category': 'other', 'color': '#cccccc', 'size': 15})
            
            net.add_node(
                word, 
                label=word,
                color=word_info['color'],
                size=word_info['size'],
                title=f"카테고리: {word_info['category']}<br>단어: {word}",
                font={'size': 14, 'color': 'white'}
            )
        
        # 단어 간 유사도 계산 및 간선 추가
        print("단어 간 유사도 계산 중...")
        edge_count = 0
        
        for i, word1 in enumerate(available_words):
            for j, word2 in enumerate(available_words[i+1:], i+1):
                try:
                    # 두 단어 간 유사도 계산
                    similarity = self.complaint_model.wv.similarity(word1, word2)
                    
                    # 임계값보다 높은 유사도만 간선으로 추가
                    if similarity > similarity_threshold:
                        # 유사도에 따른 간선 굵기 및 색상 설정
                        edge_width = max(1, similarity * 10)  # 1~10 사이의 굵기
                        
                        if similarity > 0.7:
                            edge_color = '#ff4444'  # 높은 유사도 - 빨간색
                        elif similarity > 0.5:
                            edge_color = '#ffaa44'  # 중간 유사도 - 주황색
                        else:
                            edge_color = '#4444ff'  # 낮은 유사도 - 파란색
                        
                        net.add_edge(
                            word1, 
                            word2, 
                            width=edge_width,
                            color=edge_color,
                            title=f"{word1} ↔ {word2}<br>유사도: {similarity:.3f}",
                            length=max(50, (1-similarity) * 200)  # 유사도가 높을수록 짧은 간선
                        )
                        edge_count += 1
                        
                except Exception as e:
                    continue
        
        print(f"생성된 간선 수: {edge_count}")
        
        # HTML 파일로 저장
        try:
            net.save_graph(output_file)
            print(f"네트워크 그래프 저장 완료: {output_file}")
            
            # ▶ CSV 파일 저장
            nodes_df = pd.DataFrame([{
                'id': node['id'],
                'label': node['label'],
                'category': node.get('title', '').split('<br>')[0].replace("카테고리: ", ''),
                'size': node.get('size', 15)
            } for node in net.nodes])
            
            edges_df = pd.DataFrame([{
                'source': edge['from'],
                'target': edge['to'],
                'weight': edge.get('width', 1),
                'similarity': float(edge['title'].split("유사도: ")[-1]) if "유사도" in edge['title'] else None
            } for edge in net.edges])
            
            nodes_df.to_csv("complaint_nodes_clean.csv", index=False)
            edges_df.to_csv("complaint_edges_clean.csv", index=False)
            
            print("📁 complaint_nodes_clean.csv / complaint_edges_clean.csv 파일 저장 완료")
            
            # 웹브라우저에서 자동 열기
            import webbrowser
            import os
            file_path = os.path.abspath(output_file)
            webbrowser.open(f'file://{file_path}')
            print(f"웹브라우저에서 열기: file://{file_path}")
            
            return output_file
            
        except Exception as e:
            print(f"파일 저장 중 오류 발생: {e}")
            return None
    
    def create_interactive_similarity_network(self, focus_word="불만", top_n=15, output_file="similarity_network.html"):
        """특정 단어를 중심으로 한 유사도 네트워크 생성"""
        if self.complaint_model is None:
            print("불만 학습 모델이 없습니다.")
            return None
        
        if focus_word not in self.complaint_model.wv.key_to_index:
            print(f"단어 '{focus_word}'가 모델에 없습니다.")
            return None
        
        print(f"=== '{focus_word}' 중심 유사도 네트워크 생성 ===")
        
        # 중심 단어와 유사한 단어들 찾기
        try:
            similar_words = self.complaint_model.wv.most_similar(focus_word, topn=top_n)
            all_words = [focus_word] + [word for word, _ in similar_words]
            
        except Exception as e:
            print(f"유사 단어 찾기 실패: {e}")
            return None
        
        # PyVis 네트워크 생성
        net = Network(
            height="700px", 
            width="100%", 
            bgcolor="#111111", 
            font_color="white",
            directed=False
        )
        
        # 물리학 설정
        net.set_options("""
        var options = {
            "physics": {
                "enabled": true,
                "barnesHut": {
                    "gravitationalConstant": -30000,
                    "centralGravity": 1,
                    "springLength": 150,
                    "springConstant": 0.05,
                    "damping": 0.09,
                    "avoidOverlap": 0.5
                },
                "maxVelocity": 50,
                "solver": "barnesHut",
                "stabilization": {
                    "enabled": true,
                    "iterations": 200
                }
            }
        }
        """)
        
        # 중심 노드 추가
        net.add_node(
            focus_word,
            label=focus_word,
            color='#ff4444',
            size=40,
            title=f"중심 단어: {focus_word}",
            font={'size': 18, 'color': 'white'}
        )
        
        # 유사 단어 노드 및 간선 추가
        for word, similarity in similar_words:
            # 유사도에 따른 노드 크기 및 색상
            node_size = 15 + similarity * 20  # 15~35 사이
            
            if similarity > 0.7:
                node_color = '#ff6666'
            elif similarity > 0.5:
                node_color = '#ffaa66'
            elif similarity > 0.3:
                node_color = '#66aaff'
            else:
                node_color = '#aaaaaa'
            
            # 노드 추가
            net.add_node(
                word,
                label=word,
                color=node_color,
                size=node_size,
                title=f"단어: {word}<br>유사도: {similarity:.3f}",
                font={'size': 12, 'color': 'white'}
            )
            
            # 중심 단어와 간선 연결
            edge_width = max(2, similarity * 8)
            net.add_edge(
                focus_word,
                word,
                width=edge_width,
                color='#ffffff',
                title=f"유사도: {similarity:.3f}",
                length=max(100, (1-similarity) * 300)
            )
        
        # 유사 단어들 간의 관계도 추가 (옵션)
        for i, (word1, sim1) in enumerate(similar_words[:10]):  # 상위 10개만
            for j, (word2, sim2) in enumerate(similar_words[i+1:10], i+1):
                try:
                    cross_similarity = self.complaint_model.wv.similarity(word1, word2)
                    if cross_similarity > 0.4:  # 높은 유사도만
                        net.add_edge(
                            word1,
                            word2,
                            width=cross_similarity * 3,
                            color='#666666',
                            title=f"{word1} ↔ {word2}<br>유사도: {cross_similarity:.3f}",
                            length=200
                        )
                except:
                    continue
        
        # 저장 및 열기
        try:
            net.save_graph(output_file)
            print(f"유사도 네트워크 저장 완료: {output_file}")
            
            import webbrowser
            import os
            file_path = os.path.abspath(output_file)
            webbrowser.open(f'file://{file_path}')
            
            return output_file
            
        except Exception as e:
            print(f"파일 저장 중 오류 발생: {e}")
            return None
    
    def analyze_complaint_sentiment_words(self):
        """불만 감정 단어 분석"""
        if self.complaint_model is None:
            print("불만 학습 모델이 없습니다.")
            return
        
        print("=== 불만 감정 단어 분석 ===")
        
        # 감정 키워드별 유사 단어 분석
        emotion_keywords = {
            '불만': '부정적 감정',
            '실망': '실망감',
            '불편': '불편함',
            '화': '분노',
            '답답': '답답함',
            '짜증': '짜증',
            '개선': '개선 요구',
            '요청': '요청사항'
        }
        
        for keyword, category in emotion_keywords.items():
            if keyword in self.complaint_model.wv.key_to_index:
                try:
                    similar_words = self.complaint_model.wv.most_similar(keyword, topn=10)
                    print(f"\n[{category}] '{keyword}' 관련 단어들:")
                    for word, score in similar_words:
                        print(f"  {word}: {score:.3f}")
                except:
                    print(f"'{keyword}' 유사도 계산 실패")
    
    def save_models(self, base_path="wiki_word2vec.model", complaint_path="complaint_word2vec.model"):
        """모델들 저장"""
        if self.base_model:
            self.base_model.save(base_path)
            print(f"기본 모델 저장: {base_path}")
        
        if self.complaint_model:
            self.complaint_model.save(complaint_path)
            print(f"불만 학습 모델 저장: {complaint_path}")
    
    def quick_demo(self):
        """빠른 데모를 위한 함수"""
        print("=== 빠른 데모 실행 ===")
        
        # 기본 모델로 시작
        if self.base_model is None:
            print("기본 샘플 모델 생성 중...")
            base_sentences = [
                ['대한민국', '한국', '서울', '부산'],
                ['품질', '서비스', '고객', '만족', '불만'],
                ['배송', '주문', '결제', '환불'],
                ['직원', '응답', '처리', '시간'],
                ['상품', '제품', '가격', '비용'],
                ['문제', '오류', '해결', '도움']
            ]
            
            self.base_model = Word2Vec(
                sentences=base_sentences,
                vector_size=50,  # 빠른 처리를 위해 축소
                window=3,
                min_count=1,
                workers=2,
                sg=0
            )
        
        # 간단한 불만 데이터 생성
        simple_complaints = [
            "배송이 늦어서 불만입니다",
            "상품 품질이 나쁩니다",
            "서비스가 불친절합니다",
            "가격이 너무 비쌉니다",
            "문제가 해결되지 않습니다"
        ]
        
        # simple_complaints = [c for c in simple_complaints if '배송' not in c]

        
        complaint_df = pd.DataFrame({'complaint': simple_complaints})
        complaint_df = complaint_df[~complaint_df['complaint'].str.contains('배송', na=False)]

        
        # 빠른 학습 (10 에포크)
        self.train_complaint_model(complaint_df, epochs=10)
        
        # 간단한 테스트
        if self.complaint_model:
            test_words = ['배송', '품질', '서비스']
            for word in test_words:
                if word in self.complaint_model.wv.key_to_index:
                    similar = self.complaint_model.wv.most_similar(word, topn=2)
                    print(f"{word}: {similar}")
            
            # 간단한 PyVis 네트워크 생성
            try:
                print("\n간단한 네트워크 그래프 생성 중...")
                self.create_interactive_similarity_network(
                    focus_word="불만",
                    top_n=8,
                    output_file="demo_network.html"
                )
            except Exception as e:
                print(f"네트워크 그래프 생성 실패: {e}")
        
        print("빠른 데모 완료!")

# 메인 실행 함수
def main_wiki_complaint_system(use_wiki_data=True, force_retrain=False):
    """전체 시스템 실행"""
    print("=== 위키피디아 + 고객불만 Word2Vec 시스템 ===")
    
    # 시스템 초기화
    system = WikiComplaintWord2VecSystem()
    
    # MeCab 설정 확인
    if not system.setup_mecab():
        print("MeCab 설치 후 다시 실행하세요.")
        return None
    
    # 사전 학습된 모델 로드 시도
    model_exists = system.load_base_model("wiki_word2vec.model")
    
    if not model_exists or force_retrain:
        if not model_exists:
            print("\n사전 학습된 모델이 없습니다.")
        else:
            print("\n강제 재학습 모드입니다.")
            
        if use_wiki_data:
            print("위키피디아 데이터로 새 모델을 학습합니다...")
            
            # 위키피디아 데이터 다운로드 및 처리
            wiki_success = system.download_and_process_wikipedia()
            
            if wiki_success:
                # 위키피디아 파일 목록 가져오기
                wiki_files = system.list_wiki_files('text')
                print(f"발견된 위키피디아 파일 수: {len(wiki_files)}")
                
                if len(wiki_files) > 0:
                    # 파일들을 하나로 통합 (메모리 효율을 위해 일부만 처리)
                    print("위키피디아 파일 통합 중...")
                    # 처리할 파일 수 제한 (메모리 절약)
                    limited_files = wiki_files[:min(10, len(wiki_files))]
                    merged_file = system.merge_wiki_files(limited_files)
                    
                    # 위키피디아 텍스트 전처리
                    print("위키피디아 텍스트 전처리 및 형태소 분석 중...")
                    processed_sentences = system.preprocess_wiki_text(merged_file)
                    
                    if len(processed_sentences) > 0:
                        # 위키피디아 모델 학습
                        print("위키피디아 Word2Vec 모델 학습 시작...")
                        system.train_base_wiki_model(processed_sentences)
                    else:
                        print("전처리된 문장이 없습니다. 기본 모델을 사용합니다.")
                        use_wiki_data = False
                else:
                    print("위키피디아 파일을 찾을 수 없습니다. 기본 모델을 사용합니다.")
                    use_wiki_data = False
            else:
                print("위키피디아 데이터 처리에 실패했습니다. 기본 모델을 사용합니다.")
                use_wiki_data = False
        
        # 위키피디아 학습에 실패했거나 사용하지 않는 경우 기본 모델 생성
        if not use_wiki_data or system.base_model is None:
            print("기본 샘플 모델을 생성합니다...")
            
            # 기본 한국어 문장들로 초기 모델 생성
            base_sentences = [
                ['대한민국', '한국', '서울', '부산', '정치', '경제'],
                ['품질', '서비스', '고객', '만족', '불만', '개선'],
                ['배송', '주문', '결제', '환불', '교환', '반품'],
                ['직원', '응답', '처리', '시간', '빠른', '느린'],
                ['상품', '제품', '가격', '비용', '저렴', '비싼'],
                ['문제', '오류', '해결', '도움', '지원', '상담'],
                ['웹사이트', '시스템', '온라인', '오프라인', '매장'],
                ['포장', '배송', '택배', '운송', '도착', '지연']
            ]
            
            system.base_model = Word2Vec(
                sentences=base_sentences,
                vector_size=100,
                window=5,
                min_count=1,
                workers=4,
                sg=0
            )
            print("기본 모델 생성 완료")
            
            # 기본 모델도 저장
            system.base_model.save("wiki_word2vec.model")
            print("기본 모델 저장 완료")
    else:
        print("기존 위키피디아 모델을 사용합니다.")
    
    # 고객 불만 샘플 데이터 생성
    # print("\n=== 고객 불만 데이터 생성 ===")
    # complaint_data = system.create_sample_complaint_data()
    # print(f"생성된 불만 데이터 수: {len(complaint_data)}")
    
    # 🔁 고객 불만 데이터 로딩: DB에서 가져옴
    print("\n=== DB에서 고객 불만 데이터 로드 ===")
    complaint_data = system.load_complaints_from_db(limit=10000)
    complaint_data = complaint_data[~complaint_data['complaint'].str.contains("배송")]
    print(f"🚫 '배송' 제외 후 불만 데이터 수: {len(complaint_data)}")
    print(f"▶ 불러온 불만 데이터 수: {len(complaint_data)}")
    
    # 🔁 Word2Vec 추가 학습
    print("\n=== 고객 불만 추가 학습 ===")
    processed_complaints = system.train_complaint_model(complaint_data, epochs=50)
    
    # 모델 비교 분석
    system.compare_models()
    
    # 불만 단어 관계 시각화
    print("\n=== 불만 단어 관계 시각화 ===")
    system.visualize_complaint_words()
    
    # PyVis 네트워크 그래프 시각화
    print("\n=== PyVis 네트워크 그래프 시각화 ===")
    try:
        # 전체 단어 네트워크 그래프
        system.create_pyvis_network_graph(
            similarity_threshold=0.2,
            output_file="complaint_word_network.html"
        )
        
        # 특정 단어 중심 유사도 네트워크
        system.create_interactive_similarity_network(
            focus_word="불만",
            top_n=20,
            output_file="complaint_similarity_network.html"
        )
        
    except ImportError:
        print("PyVis 라이브러리가 설치되지 않았습니다.")
        print("설치 명령어: pip install pyvis")
    except Exception as e:
        print(f"PyVis 시각화 중 오류 발생: {e}")
    
    # 불만 감정 단어 분석
    system.analyze_complaint_sentiment_words()
    
    # 모델 저장
    print("\n=== 모델 저장 ===")
    system.save_models()
    
    return system

# 실행 예시
if __name__ == "__main__":
    import argparse
    
    # 명령행 인자 파싱
    parser = argparse.ArgumentParser(description='Korean Complaint Word2Vec System')
    parser.add_argument('--no-wiki', action='store_true', 
                       help='위키피디아 데이터 사용하지 않고 기본 모델만 사용')
    parser.add_argument('--force-retrain', action='store_true', 
                       help='기존 모델이 있어도 강제로 재학습')
    parser.add_argument('--sample-only', action='store_true',
                       help='빠른 테스트를 위해 기본 샘플 모델만 사용')
    
    args = parser.parse_args()
    
    if args.sample_only:
        print("=== 빠른 테스트 모드 (기본 샘플 모델만 사용) ===")
        system = main_wiki_complaint_system(use_wiki_data=False, force_retrain=False)
    else:
        # 전체 시스템 실행
        use_wiki = not args.no_wiki
        force_retrain = args.force_retrain
        
        print(f"=== 시스템 설정 ===")
        print(f"위키피디아 데이터 사용: {'예' if use_wiki else '아니오'}")
        print(f"강제 재학습: {'예' if force_retrain else '아니오'}")
        
        system = main_wiki_complaint_system(use_wiki_data=use_wiki, force_retrain=force_retrain)
    
    # 개별 기능 테스트
    if system and system.complaint_model:
        print("\n=== 개별 단어 유사도 테스트 ===")
        test_words = ['배송', '품질', '서비스', '불만', '개선']
        for word in test_words:
            if word in system.complaint_model.wv.key_to_index:
                similar = system.complaint_model.wv.most_similar(word, topn=3)
                print(f"{word}: {[f'{w}({s:.2f})' for w, s in similar]}")
        
        # 추가 PyVis 테스트 (선택사항)
        try:
            print("\n=== PyVis 추가 테스트 ===")
            
            # 커스텀 단어 목록으로 네트워크 생성
            custom_words = ['배송', '품질', '서비스', '불만', '개선', '고객', '만족']
            system.create_pyvis_network_graph(
                target_words=custom_words,
                similarity_threshold=0.1,
                output_file="custom_network.html"
            )
            
            # 다른 중심 단어로 유사도 네트워크 생성
            for focus in ['서비스', '품질']:
                if focus in system.complaint_model.wv.key_to_index:
                    system.create_interactive_similarity_network(
                        focus_word=focus,
                        top_n=10,
                        output_file=f"{focus}_network.html"
                    )
            
        except Exception as e:
            print(f"PyVis 추가 테스트 실패: {e}")
    
    print("\n=== 실행 옵션 안내 ===")
    print("기본 실행: python korean_complaint_word2vec.py")
    print("위키피디아 없이: python korean_complaint_word2vec.py --no-wiki")
    print("강제 재학습: python korean_complaint_word2vec.py --force-retrain")
    print("빠른 테스트: python korean_complaint_word2vec.py --sample-only")
    
    print("\n=== PyVis 네트워크 그래프 파일 ===")
    print("📊 complaint_word_network.html - 전체 단어 관계 네트워크")
    print("🎯 complaint_similarity_network.html - '불만' 중심 유사도 네트워크")
    print("⚡ demo_network.html - 빠른 데모 네트워크")
    print("🔧 custom_network.html - 커스텀 단어 네트워크")
    print("📈 [단어]_network.html - 특정 단어 중심 네트워크")
    print("\n💡 Tip: 웹브라우저에서 HTML 파일을 열어 인터랙티브 그래프를 확인하세요!")
