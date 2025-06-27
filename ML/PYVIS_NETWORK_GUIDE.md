# 🌐 PyVis Force-directed Graph 시각화 가이드

## 📋 PyVis 네트워크 그래프 기능 개요

**PyVis**를 사용한 인터랙티브 Force-directed graph 기능이 추가되어, Word2Vec 모델의 단어 관계를 역동적이고 직관적으로 시각화할 수 있습니다.

### 🎯 **주요 특징**

- **🌟 인터랙티브**: 마우스로 노드를 드래그하여 위치 조정 가능
- **⚡ Force-directed**: 물리학 시뮬레이션으로 자동 노드 배치
- **🎨 시각적 표현**: 유사도에 따른 색상, 크기, 간선 굵기 차별화
- **🔍 상세 정보**: 호버 시 단어 정보 및 유사도 표시
- **📱 반응형**: 웹브라우저에서 실시간 상호작용

## 🛠️ 새로 추가된 함수들

### 1. **create_pyvis_network_graph() - 전체 단어 네트워크**

```python
def create_pyvis_network_graph(self, 
                              target_words=None, 
                              max_words=50, 
                              similarity_threshold=0.3, 
                              output_file="word_network.html"):
```

#### **기능**
- 지정된 단어들 간의 전체 관계를 네트워크로 시각화
- 유사도 임계값 이상의 단어들을 간선으로 연결
- 카테고리별 색상 분류 (부정적/해결/서비스/중립/긍정)

#### **매개변수**
- `target_words`: 시각화할 단어 목록 (기본: 불만 관련 27개 단어)
- `similarity_threshold`: 간선 생성 임계값 (기본: 0.3)
- `output_file`: 출력 HTML 파일명

#### **사용 예시**
```python
# 기본 사용
system.create_pyvis_network_graph()

# 커스텀 단어 목록으로 생성
custom_words = ['배송', '품질', '서비스', '불만', '개선']
system.create_pyvis_network_graph(
    target_words=custom_words,
    similarity_threshold=0.2,
    output_file="my_network.html"
)
```

### 2. **create_interactive_similarity_network() - 중심 단어 유사도 네트워크**

```python
def create_interactive_similarity_network(self, 
                                        focus_word="불만", 
                                        top_n=15, 
                                        output_file="similarity_network.html"):
```

#### **기능**
- 특정 단어를 중심으로 한 방사형 유사도 네트워크
- 중심 단어와 가장 유사한 N개 단어 표시
- 유사 단어들 간의 교차 관계도 표시

#### **매개변수**
- `focus_word`: 중심이 될 단어
- `top_n`: 표시할 유사 단어 개수 (기본: 15개)
- `output_file`: 출력 HTML 파일명

#### **사용 예시**
```python
# '서비스' 중심 네트워크
system.create_interactive_similarity_network(
    focus_word="서비스",
    top_n=20,
    output_file="service_network.html"
)
```

## 🎨 시각적 특징

### **노드 (단어) 표현**

#### **색상 코드**
- 🔴 **빨간색** (#ff4444): 부정적 감정 (불만, 문제, 오류)
- 🟠 **주황색** (#ff6666): 경미한 부정 (실망, 불편)
- 🟢 **초록색** (#44ff44): 해결/개선 (개선, 처리, 해결)
- 🔵 **파란색** (#4444ff): 서비스 관련 (서비스, 품질, 배송)
- 🟡 **노란색** (#ffff44): 중립 (고객, 상품, 가격)
- 🟢 **연두색** (#44ffaa): 긍정적 (만족, 빠르다)
- ⚪ **회색** (#cccccc): 기타 단어

#### **크기 기준**
- **대형** (30-40px): 핵심 단어 (불만, 서비스, 품질)
- **중형** (20-25px): 중요 단어 (문제, 개선, 고객)
- **소형** (15px): 일반 단어

### **간선 (관계) 표현**

#### **굵기**
- 유사도 × 10 = 간선 굵기 (1-10px)
- 높은 유사도 → 굵은 간선

#### **색상**
- 🔴 **빨간색**: 매우 높은 유사도 (0.7+)
- 🟠 **주황색**: 중간 유사도 (0.5-0.7)
- 🔵 **파란색**: 낮은 유사도 (임계값-0.5)

#### **길이**
- 높은 유사도 → 짧은 간선 (가까운 배치)
- 낮은 유사도 → 긴 간선 (먼 배치)

## ⚙️ 물리학 시뮬레이션 설정

### **Force-directed 알고리즘 특징**

1. **중력**: 노드들을 중앙으로 끌어당기는 힘
2. **반발력**: 노드들 간의 충돌 방지
3. **스프링**: 연결된 노드들을 적절한 거리로 유지
4. **감쇠**: 움직임의 점진적 안정화

### **설정 옵션**

#### **전체 네트워크 (forceAtlas2Based)**
```javascript
"gravitationalConstant": -50,     // 중력 강도
"centralGravity": 0.01,          // 중앙 끌림
"springLength": 100,             // 기본 간선 길이
"springConstant": 0.08,          // 스프링 강도
"damping": 0.4,                  // 감쇠 계수
"avoidOverlap": 0.5             // 충돌 방지
```

#### **중심 네트워크 (barnesHut)**
```javascript
"gravitationalConstant": -30000,  // 강한 중력
"centralGravity": 1,             // 중앙 집중
"springLength": 150,             // 긴 간선
"springConstant": 0.05,          // 약한 스프링
"damping": 0.09                  // 낮은 감쇠
```

## 🚀 실행 및 사용 방법

### **1. 기본 실행**
```bash
python korean_complaint_word2vec.py
```
→ 2개의 HTML 파일 자동 생성 및 브라우저 열기

### **2. 생성되는 파일들**
```
📊 complaint_word_network.html      # 전체 단어 관계 네트워크
🎯 complaint_similarity_network.html # '불만' 중심 유사도 네트워크
⚡ demo_network.html                 # 빠른 데모용 (--sample-only)
🔧 custom_network.html               # 커스텀 단어 네트워크
📈 [단어]_network.html               # 특정 단어 중심 네트워크
```

### **3. 브라우저에서 상호작용**

#### **기본 조작**
- **🖱️ 드래그**: 노드 위치 이동
- **🔍 줌**: 마우스 휠로 확대/축소
- **👆 호버**: 단어 정보 및 유사도 표시
- **⏸️ 일시정지**: 물리학 시뮬레이션 중단/재개

#### **UI 컨트롤**
- **Physics**: 물리학 시뮬레이션 온/오프
- **Configure**: 실시간 설정 조정
- **Generate options**: 현재 설정 코드 생성

## 📊 활용 사례

### **1. 감정 분석**
```python
# 부정적 감정 단어들의 클러스터 확인
negative_words = ['불만', '실망', '화나다', '답답', '짜증']
system.create_pyvis_network_graph(target_words=negative_words)
```

### **2. 도메인 분석**
```python
# 특정 서비스 영역 분석
service_words = ['배송', '포장', '운송', '택배', '도착', '지연']
system.create_pyvis_network_graph(target_words=service_words)
```

### **3. 고객 여정 매핑**
```python
# 고객 경험 단계별 네트워크
for stage in ['주문', '배송', '사용', '서비스']:
    if stage in system.complaint_model.wv.key_to_index:
        system.create_interactive_similarity_network(
            focus_word=stage,
            output_file=f"customer_journey_{stage}.html"
        )
```

## 🔧 고급 설정 및 커스터마이징

### **1. 색상 테마 변경**
```python
# 다크 테마
bgcolor="#222222", font_color="white"

# 라이트 테마  
bgcolor="#ffffff", font_color="black"

# 커스텀 테마
bgcolor="#1a1a2e", font_color="#16213e"
```

### **2. 물리학 파라미터 조정**
```python
# 더 타이트한 클러스터링
"springLength": 50,
"centralGravity": 0.1

# 더 넓은 분산
"springLength": 200,
"centralGravity": 0.001
```

### **3. 노드 크기 및 스타일**
```python
# 균일한 크기
size=20

# 유사도 기반 크기
size=15 + similarity * 25
```

## 🎯 모범 사례 및 팁

### **✅ 권장사항**

1. **적절한 임계값 설정**: 너무 낮으면 복잡, 너무 높으면 빈약
2. **단어 수 제한**: 50개 이하로 유지하여 가독성 확보
3. **카테고리별 분석**: 비슷한 성격의 단어들로 그룹핑
4. **반복 실험**: 다양한 파라미터로 최적 시각화 찾기

### **⚠️ 주의사항**

1. **메모리 사용량**: 대용량 네트워크는 브라우저 성능 저하
2. **유사도 계산**: 모든 단어 쌍에 대한 계산으로 시간 소요
3. **브라우저 호환성**: 최신 브라우저에서 최적 동작

### **🚀 성능 최적화**

```python
# 빠른 처리를 위한 설정
similarity_threshold = 0.4  # 높은 임계값
max_words = 20             # 적은 단어 수
top_n = 10                 # 적은 유사 단어
```

## 📝 결론

PyVis Force-directed graph 기능을 통해 **Korean Complaint Word2Vec** 시스템이 다음과 같이 향상되었습니다:

1. **🎨 직관적 시각화**: 복잡한 단어 관계를 한눈에 파악
2. **🌟 인터랙티브 탐색**: 사용자가 직접 조작하며 관계 탐색
3. **📊 다층적 분석**: 전체 네트워크와 중심 네트워크 동시 제공
4. **🔧 높은 커스터마이징**: 용도에 맞는 다양한 설정 가능

이제 고객 불만 분석이 **정적인 수치**에서 **역동적인 관계 탐색**으로 진화했습니다! 🎊 