# 📈 주식 예측 및 추천 시스템

AI 기반 주가 예측 및 종목 추천 시스템입니다.

## 🌟 주요 기능

- **실시간 증권 데이터 연동**: 한국 주식시장 데이터 자동 수집
- **AI 주가 예측**: 머신러닝 기반 주가 상승/하락 예측
- **Top 10 추천**: 상승 가능성이 높은 종목 순위
- **카테고리별 조회**: 업종별, 시가총액별 필터링
- **대화형 대시보드**: Streamlit 기반 직관적인 UI

## 🛠️ 기술 스택

- **Frontend**: Streamlit
- **Backend**: Python 3.8+
- **Data Source**: pykrx (한국거래소 데이터)
- **ML Models**: 
  - Random Forest (상승 확률 예측)
  - Technical Indicators (기술적 분석)
- **Visualization**: Plotly, Matplotlib

## 📦 설치 방법

### 1. 저장소 클론 및 의존성 설치

```bash
pip install -r requirements.txt
```

### 2. 환경변수 설정

`.env.example` 파일을 `.env`로 복사하고 필요한 값을 설정하세요:

### 설치오류시
python -m pip install --upgrade pip setuptools wheel


```bash
cp .env.example .env
```

`.env` 파일 예시:
```
# API 설정 (선택사항 - 추가 데이터 소스 사용시)
ALPHA_VANTAGE_API_KEY=your_api_key_here
PREDICTION_DAYS=30
TOP_N_STOCKS=10

# 모델 설정
MODEL_CONFIDENCE_THRESHOLD=0.6
USE_ADVANCED_MODEL=False
```

## 🚀 실행 방법

```bash
streamlit run app.py
```

브라우저에서 자동으로 `http://localhost:8501` 열립니다.

## 📊 사용 방법

1. **대시보드 메인**
   - 전체 시장 개요 확인
   - Top 10 추천 종목 실시간 조회

2. **카테고리별 분석**
   - 업종 선택 (IT, 금융, 제조 등)
   - 시가총액 범위 설정
   - 필터링된 종목 분석

3. **개별 종목 분석**
   - 종목 코드 또는 이름 검색
   - 상세 차트 및 예측 결과
   - 기술적 지표 확인

4. **포트폴리오 구성**
   - 추천 종목 기반 포트폴리오 생성
   - 리스크/수익률 시뮬레이션

## 📁 프로젝트 구조

```
stock_prediction_system/
├── app.py                 # Streamlit 메인 애플리케이션
├── config/
│   └── settings.py        # 설정 관리
├── data/
│   ├── data_loader.py     # 데이터 수집
│   └── preprocessor.py    # 데이터 전처리
├── models/
│   ├── predictor.py       # 예측 모델
│   └── recommender.py     # 추천 엔진
├── utils/
│   ├── helpers.py         # 유틸리티 함수
│   └── visualizer.py      # 시각화
└── requirements.txt       # 패키지 의존성
```

## ⚠️ 주의사항

- **투자 참고용**: 이 시스템은 투자 참고용이며, 투자 결정의 책임은 사용자에게 있습니다.
- **데이터 지연**: 실시간 데이터가 아닌 지연된 데이터를 사용할 수 있습니다.
- **예측 정확도**: ML 모델의 예측은 100% 정확하지 않습니다.

## 🔄 업데이트 계획

- [ ] 해외 주식 지원 (미국, 중국 등)
- [ ] 뉴스 감성 분석 통합
- [ ] 실시간 알림 기능
- [ ] 백테스팅 기능
- [ ] API 서버 모드

## 📝 라이선스

MIT License

## 👥 기여

Pull Request는 언제나 환영합니다!

## 📞 문의

이슈 페이지를 통해 문의해주세요.
