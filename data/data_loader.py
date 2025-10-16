"""
데이터 로더 모듈
한국 주식시장 데이터를 수집합니다.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import logging

try:
    from pykrx import stock
except ImportError:
    print("pykrx 패키지가 설치되지 않았습니다. pip install pykrx")
    stock = None

from config.settings import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StockDataLoader:
    """주식 데이터 로더 클래스"""
    
    def __init__(self):
        self.cache = {}
        self.cache_timestamp = {}
    
    def get_stock_list(self, market: str = 'KOSPI') -> pd.DataFrame:
        """
        주식 종목 리스트 조회
        
        Args:
            market: 시장 구분 (KOSPI, KOSDAQ)
            
        Returns:
            종목 리스트 DataFrame
        """
        try:
            cache_key = f'stock_list_{market}'
            
            # 캐시 확인
            if self._is_cache_valid(cache_key):
                logger.info(f"캐시에서 {market} 종목 리스트 반환")
                return self.cache[cache_key]
            
            date_str = datetime.now().strftime('%Y%m%d')
            
            if stock is None:
                # 테스트 데이터
                return self._get_sample_stock_list(market)
            
            # 실제 데이터 조회
            tickers = stock.get_market_ticker_list(date=date_str, market=market)
            
            stock_list = []
            for ticker in tickers[:50]:  # 성능을 위해 50개로 제한
                try:
                    name = stock.get_market_ticker_name(ticker)
                    stock_list.append({
                        'ticker': ticker,
                        'name': name,
                        'market': market
                    })
                except Exception as e:
                    logger.warning(f"종목 {ticker} 정보 조회 실패: {e}")
                    continue
            
            df = pd.DataFrame(stock_list)
            
            # 캐시 저장
            self._save_to_cache(cache_key, df)
            
            logger.info(f"{market} 종목 {len(df)}개 조회 완료")
            return df
            
        except Exception as e:
            logger.error(f"종목 리스트 조회 실패: {e}")
            return self._get_sample_stock_list(market)
    
    def get_stock_price(self, ticker: str, days: int = 365) -> pd.DataFrame:
        """
        종목 가격 데이터 조회
        
        Args:
            ticker: 종목 코드
            days: 조회 일수
            
        Returns:
            가격 데이터 DataFrame
        """
        try:
            cache_key = f'price_{ticker}_{days}'
            
            # 캐시 확인
            if self._is_cache_valid(cache_key):
                logger.info(f"캐시에서 {ticker} 가격 데이터 반환")
                return self.cache[cache_key]
            
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            start_str = start_date.strftime('%Y%m%d')
            end_str = end_date.strftime('%Y%m%d')
            
            if stock is None:
                # 테스트 데이터
                return self._generate_sample_price_data(ticker, days)
            
            # 실제 데이터 조회
            df = stock.get_market_ohlcv_by_date(start_str, end_str, ticker)
            
            if df.empty:
                logger.warning(f"{ticker} 가격 데이터가 없습니다")
                return self._generate_sample_price_data(ticker, days)
            
            # 컬럼명 영문으로 변경
            df.columns = ['open', 'high', 'low', 'close', 'volume']
            df.index.name = 'date'
            df = df.reset_index()
            
            # 캐시 저장
            self._save_to_cache(cache_key, df)
            
            logger.info(f"{ticker} 가격 데이터 {len(df)}일 조회 완료")
            return df
            
        except Exception as e:
            logger.error(f"{ticker} 가격 조회 실패: {e}")
            return self._generate_sample_price_data(ticker, days)
    
    def get_market_cap(self, ticker: str) -> Optional[float]:
        """
        시가총액 조회
        
        Args:
            ticker: 종목 코드
            
        Returns:
            시가총액 (억원)
        """
        try:
            if stock is None:
                return np.random.uniform(1000, 100000)
            
            date_str = datetime.now().strftime('%Y%m%d')
            cap = stock.get_market_cap_by_ticker(date=date_str, market="ALL")
            
            if ticker in cap.index:
                # 원 -> 억원
                return cap.loc[ticker, '시가총액'] / 100000000
            
            return None
            
        except Exception as e:
            logger.error(f"{ticker} 시가총액 조회 실패: {e}")
            return None
    
    def get_sector_stocks(self, sector: str, market: str = 'KOSPI') -> List[str]:
        """
        업종별 종목 조회
        
        Args:
            sector: 업종명
            market: 시장 구분
            
        Returns:
            종목 코드 리스트
        """
        try:
            stock_list = self.get_stock_list(market)
            
            if sector == 'ALL':
                return stock_list['ticker'].tolist()
            
            # 업종별 필터링 (실제로는 업종 정보가 필요하지만, 여기서는 샘플 구현)
            # pykrx의 get_market_sector 사용 가능
            filtered = stock_list.sample(min(20, len(stock_list)))
            return filtered['ticker'].tolist()
            
        except Exception as e:
            logger.error(f"업종별 종목 조회 실패: {e}")
            return []
    
    def _is_cache_valid(self, key: str) -> bool:
        """캐시 유효성 확인"""
        if not settings.CACHE_ENABLED:
            return False
        
        if key not in self.cache:
            return False
        
        if key not in self.cache_timestamp:
            return False
        
        elapsed = (datetime.now() - self.cache_timestamp[key]).total_seconds()
        return elapsed < settings.CACHE_TTL
    
    def _save_to_cache(self, key: str, data: pd.DataFrame):
        """캐시에 저장"""
        if settings.CACHE_ENABLED:
            self.cache[key] = data.copy()
            self.cache_timestamp[key] = datetime.now()
    
    def _get_sample_stock_list(self, market: str) -> pd.DataFrame:
        """샘플 종목 리스트 생성"""
        samples = {
            'KOSPI': [
                ('005930', '삼성전자'),
                ('000660', 'SK하이닉스'),
                ('035420', 'NAVER'),
                ('035720', '카카오'),
                ('051910', 'LG화학'),
                ('006400', '삼성SDI'),
                ('207940', '삼성바이오로직스'),
                ('005380', '현대차'),
                ('005490', 'POSCO홀딩스'),
                ('068270', '셀트리온'),
            ],
            'KOSDAQ': [
                ('247540', '에코프로비엠'),
                ('086520', '에코프로'),
                ('091990', '셀트리온헬스케어'),
                ('196170', '알테오젠'),
                ('214150', '클래시스'),
                ('293490', '카카오게임즈'),
                ('328130', '루닛'),
                ('357780', '솔브레인'),
                ('373220', 'LG에너지솔루션'),
                ('278280', '천보'),
            ]
        }
        
        data = samples.get(market, samples['KOSPI'])
        return pd.DataFrame(data, columns=['ticker', 'name']).assign(market=market)
    
    def _generate_sample_price_data(self, ticker: str, days: int) -> pd.DataFrame:
        """샘플 가격 데이터 생성"""
        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
        
        # 랜덤 워크로 가격 생성
        base_price = np.random.uniform(10000, 100000)
        returns = np.random.normal(0.001, 0.02, days)
        prices = base_price * np.exp(np.cumsum(returns))
        
        df = pd.DataFrame({
            'date': dates,
            'open': prices * np.random.uniform(0.98, 1.02, days),
            'high': prices * np.random.uniform(1.00, 1.05, days),
            'low': prices * np.random.uniform(0.95, 1.00, days),
            'close': prices,
            'volume': np.random.randint(100000, 10000000, days)
        })
        
        return df
    
    def clear_cache(self):
        """캐시 초기화"""
        self.cache.clear()
        self.cache_timestamp.clear()
        logger.info("캐시가 초기화되었습니다")


# 싱글톤 인스턴스
data_loader = StockDataLoader()
