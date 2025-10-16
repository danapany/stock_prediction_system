"""
설정 관리 모듈
.env 파일에서 환경변수를 로드하고 관리합니다.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# 프로젝트 루트 디렉토리
BASE_DIR = Path(__file__).resolve().parent.parent

# .env 파일 로드
env_path = BASE_DIR / '.env'
load_dotenv(dotenv_path=env_path)


class Settings:
    """애플리케이션 설정 클래스"""
    
    # API 설정
    ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY', '')
    FINNHUB_API_KEY = os.getenv('FINNHUB_API_KEY', '')
    
    # 예측 설정
    PREDICTION_DAYS = int(os.getenv('PREDICTION_DAYS', '30'))
    TOP_N_STOCKS = int(os.getenv('TOP_N_STOCKS', '10'))
    MODEL_CONFIDENCE_THRESHOLD = float(os.getenv('MODEL_CONFIDENCE_THRESHOLD', '0.6'))
    
    # 데이터 수집 설정
    DATA_UPDATE_INTERVAL = int(os.getenv('DATA_UPDATE_INTERVAL', '3600'))
    CACHE_ENABLED = os.getenv('CACHE_ENABLED', 'True').lower() == 'true'
    CACHE_TTL = int(os.getenv('CACHE_TTL', '1800'))
    
    # 모델 설정
    USE_ADVANCED_MODEL = os.getenv('USE_ADVANCED_MODEL', 'False').lower() == 'true'
    MODEL_RETRAIN_DAYS = int(os.getenv('MODEL_RETRAIN_DAYS', '7'))
    MIN_TRAINING_SAMPLES = int(os.getenv('MIN_TRAINING_SAMPLES', '100'))
    
    # 로깅 설정
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    ENABLE_DEBUG = os.getenv('ENABLE_DEBUG', 'False').lower() == 'true'
    
    # 시장 설정
    DEFAULT_MARKET = os.getenv('DEFAULT_MARKET', 'KOSPI')
    DEFAULT_SECTOR = os.getenv('DEFAULT_SECTOR', 'ALL')
    
    # UI 설정
    THEME = os.getenv('THEME', 'light')
    CHART_HEIGHT = int(os.getenv('CHART_HEIGHT', '500'))
    SHOW_TECHNICAL_INDICATORS = os.getenv('SHOW_TECHNICAL_INDICATORS', 'True').lower() == 'true'
    
    # 성능 설정
    MAX_CONCURRENT_REQUESTS = int(os.getenv('MAX_CONCURRENT_REQUESTS', '5'))
    REQUEST_TIMEOUT = int(os.getenv('REQUEST_TIMEOUT', '30'))
    
    # 경로 설정
    DATA_DIR = BASE_DIR / 'data'
    MODELS_DIR = BASE_DIR / 'models'
    CACHE_DIR = BASE_DIR / '.cache'
    
    # 시장 카테고리
    MARKET_CATEGORIES = {
        'KOSPI': '코스피',
        'KOSDAQ': '코스닥',
    }
    
    # 업종 카테고리
    SECTOR_CATEGORIES = {
        'ALL': '전체',
        '반도체': '반도체',
        '자동차': '자동차',
        '은행': '은행',
        '보험': '보험',
        '증권': '증권',
        '화학': '화학',
        '철강': '철강',
        '기계': '기계',
        '건설': '건설',
        '운송': '운송',
        '유통': '유통',
        '음식료': '음식료',
        '섬유': '섬유',
        '제약': '제약',
        '통신': '통신',
        '서비스': '서비스',
        'IT': 'IT',
        '엔터테인먼트': '엔터테인먼트',
    }
    
    @classmethod
    def ensure_directories(cls):
        """필요한 디렉토리 생성"""
        cls.DATA_DIR.mkdir(exist_ok=True)
        cls.CACHE_DIR.mkdir(exist_ok=True)
    
    @classmethod
    def get_config_summary(cls):
        """설정 요약 반환"""
        return {
            'prediction_days': cls.PREDICTION_DAYS,
            'top_n_stocks': cls.TOP_N_STOCKS,
            'default_market': cls.DEFAULT_MARKET,
            'cache_enabled': cls.CACHE_ENABLED,
            'advanced_model': cls.USE_ADVANCED_MODEL,
        }


# 싱글톤 인스턴스
settings = Settings()
settings.ensure_directories()
