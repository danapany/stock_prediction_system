# 🚀 빠른 시작 가이드

## 1단계: 환경 설정

### Python 버전 확인
```bash
python --version  # Python 3.8 이상 필요
```

### 가상환경 생성 (권장)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

## 2단계: 의존성 설치

```bash
pip install -r requirements.txt
```

설치 시간: 약 2-3분 소요

## 3단계: 환경변수 설정 (선택사항)

```bash
# .env.example을 .env로 복사
cp .env.example .env

# .env 파일 편집 (필요시)
# 기본 설정으로도 실행 가능합니다
```

## 4단계: 애플리케이션 실행

```bash
streamlit run app.py
```

실행 후 자동으로 브라우저가 열립니다.
또는 수동으로 http://localhost:8501 접속

## 5단계: 시스템 사용

### 추천 종목 조회
1. 사이드바에서 시장(KOSPI/KOSDAQ) 선택
2. 업종 선택 (선택사항)
3. "추천 종목 조회" 버튼 클릭
4. AI가 분석한 Top 10 종목 확인

### 개별 종목 분석
1. "종목 분석" 탭 클릭
2. 6자리 종목 코드 입력 (예: 005930)
3. "분석 시작" 버튼 클릭
4. 상세 분석 결과 확인

### 포트폴리오 구성
1. "포트폴리오" 탭 클릭
2. 투자 금액 입력
3. 리스크 레벨 선택
4. "포트폴리오 제안 받기" 버튼 클릭

## 문제 해결

### 패키지 설치 오류
```bash
# pip 업그레이드
pip install --upgrade pip

# 개별 설치 시도
pip install streamlit pandas numpy scikit-learn plotly pykrx
```

### pykrx 설치 오류
```bash
# Windows에서 Visual C++ 빌드 도구 필요할 수 있음
# 또는 다음 명령 시도:
pip install pykrx --no-cache-dir
```

### 실행 오류
```bash
# 캐시 삭제
streamlit cache clear

# 포트 변경
streamlit run app.py --server.port 8502
```

## 주요 종목 코드 예시

- **삼성전자**: 005930
- **SK하이닉스**: 000660
- **NAVER**: 035420
- **카카오**: 035720
- **현대차**: 005380
- **LG화학**: 051910

## 성능 최적화 팁

1. **첫 실행 시**: 데이터 수집 및 모델 초기화로 1-2분 소요
2. **캐시 활용**: .env에서 CACHE_ENABLED=True 설정 (기본값)
3. **동시 요청 제한**: MAX_CONCURRENT_REQUESTS 조정

## 다음 단계

- 📖 [전체 문서](README.md) 읽기
- 🔧 [설정 옵션](.env.example) 확인
- 🧪 [테스트 실행](tests/test_basic.py)

## 지원 및 문의

문제가 발생하면 이슈 페이지에 문의해주세요.

Happy Trading! 📈
