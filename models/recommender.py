"""
종목 추천 엔진 모듈
상승 가능성이 높은 종목을 추천합니다.
"""

import pandas as pd
import numpy as np
from typing import List, Dict
import logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from config.settings import settings
from data.data_loader import data_loader
from models.predictor import predictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StockRecommender:
    """종목 추천 클래스"""
    
    def __init__(self):
        self.recommendations_cache = None
        self.cache_timestamp = None
    
    def get_top_recommendations(
        self, 
        market: str = 'KOSPI',
        sector: str = 'ALL',
        top_n: int = None
    ) -> pd.DataFrame:
        """
        Top N 추천 종목 조회
        
        Args:
            market: 시장 구분 (KOSPI, KOSDAQ)
            sector: 업종 구분
            top_n: 추천 개수
            
        Returns:
            추천 종목 DataFrame
        """
        try:
            if top_n is None:
                top_n = settings.TOP_N_STOCKS
            
            logger.info(f"{market} {sector} 종목 추천 시작...")
            
            # 종목 리스트 조회
            if sector == 'ALL':
                stock_list = data_loader.get_stock_list(market)
            else:
                tickers = data_loader.get_sector_stocks(sector, market)
                stock_list = pd.DataFrame({'ticker': tickers})
            
            if stock_list.empty:
                logger.warning("종목 리스트가 비어있습니다")
                return pd.DataFrame()
            
            # 각 종목 분석
            recommendations = []
            
            # 멀티스레딩으로 병렬 처리
            with ThreadPoolExecutor(max_workers=settings.MAX_CONCURRENT_REQUESTS) as executor:
                future_to_ticker = {
                    executor.submit(self._analyze_single_stock, row): row 
                    for _, row in stock_list.iterrows()
                }
                
                for future in as_completed(future_to_ticker):
                    try:
                        result = future.result(timeout=settings.REQUEST_TIMEOUT)
                        if result:
                            recommendations.append(result)
                    except Exception as e:
                        logger.warning(f"종목 분석 중 오류: {e}")
            
            if not recommendations:
                logger.warning("추천 종목이 없습니다")
                return pd.DataFrame()
            
            # DataFrame 생성 및 정렬
            df_recommendations = pd.DataFrame(recommendations)
            
            # 신뢰도 필터링
            df_recommendations = df_recommendations[
                df_recommendations['confidence'] >= settings.MODEL_CONFIDENCE_THRESHOLD
            ]
            
            # 상승 확률로 정렬
            df_recommendations = df_recommendations.sort_values(
                'up_probability', 
                ascending=False
            ).head(top_n)
            
            # 순위 추가
            df_recommendations['rank'] = range(1, len(df_recommendations) + 1)
            
            logger.info(f"추천 종목 {len(df_recommendations)}개 생성 완료")
            
            # 캐시 저장
            self.recommendations_cache = df_recommendations
            self.cache_timestamp = datetime.now()
            
            return df_recommendations
            
        except Exception as e:
            logger.error(f"추천 종목 조회 실패: {e}")
            return pd.DataFrame()
    
    def _analyze_single_stock(self, stock_row: pd.Series) -> Dict:
        """
        단일 종목 분석
        
        Args:
            stock_row: 종목 정보 Series
            
        Returns:
            분석 결과 딕셔너리
        """
        try:
            ticker = stock_row['ticker']
            name = stock_row.get('name', ticker)
            
            # 가격 데이터 조회
            df_price = data_loader.get_stock_price(ticker, days=180)
            
            if df_price.empty or len(df_price) < 60:
                logger.warning(f"{ticker} 데이터 부족")
                return None
            
            # 분석
            analysis = predictor.analyze_stock(df_price)
            
            # 시가총액 조회
            market_cap = data_loader.get_market_cap(ticker)
            
            result = {
                'ticker': ticker,
                'name': name,
                'current_price': analysis['current_price'],
                'up_probability': analysis['prediction']['up_probability'],
                'confidence': analysis['prediction']['confidence'],
                'direction': analysis['prediction']['direction'],
                'rsi': analysis['technical_indicators']['rsi'],
                'trend': analysis['signals']['trend'],
                'overall_signal': analysis['signals']['overall'],
                'market_cap': market_cap,
            }
            
            return result
            
        except Exception as e:
            logger.error(f"종목 분석 실패: {e}")
            return None
    
    def get_recommendations_by_strategy(
        self, 
        strategy: str,
        market: str = 'KOSPI',
        top_n: int = 10
    ) -> pd.DataFrame:
        """
        전략별 추천 종목
        
        Args:
            strategy: 투자 전략 (momentum, value, growth)
            market: 시장 구분
            top_n: 추천 개수
            
        Returns:
            추천 종목 DataFrame
        """
        try:
            # 기본 추천 조회
            recommendations = self.get_top_recommendations(market=market, top_n=top_n * 3)
            
            if recommendations.empty:
                return pd.DataFrame()
            
            # 전략별 필터링 및 정렬
            if strategy == 'momentum':
                # 모멘텀: RSI와 추세 중시
                filtered = recommendations[
                    (recommendations['rsi'] > 50) & 
                    (recommendations['trend'] == 'BULLISH')
                ]
                filtered = filtered.sort_values('up_probability', ascending=False)
                
            elif strategy == 'value':
                # 가치: 저평가 종목 (낮은 RSI)
                filtered = recommendations[
                    (recommendations['rsi'] < 50)
                ]
                filtered = filtered.sort_values(['up_probability', 'rsi'], ascending=[False, True])
                
            elif strategy == 'growth':
                # 성장: 높은 상승 확률과 신뢰도
                filtered = recommendations[
                    recommendations['confidence'] > 0.7
                ]
                filtered = filtered.sort_values('up_probability', ascending=False)
                
            else:
                filtered = recommendations
            
            return filtered.head(top_n)
            
        except Exception as e:
            logger.error(f"전략별 추천 실패: {e}")
            return pd.DataFrame()
    
    def compare_stocks(self, tickers: List[str]) -> pd.DataFrame:
        """
        종목 비교 분석
        
        Args:
            tickers: 비교할 종목 코드 리스트
            
        Returns:
            비교 결과 DataFrame
        """
        try:
            comparisons = []
            
            for ticker in tickers:
                stock_row = pd.Series({'ticker': ticker, 'name': ticker})
                result = self._analyze_single_stock(stock_row)
                
                if result:
                    comparisons.append(result)
            
            if not comparisons:
                return pd.DataFrame()
            
            df_comparison = pd.DataFrame(comparisons)
            return df_comparison
            
        except Exception as e:
            logger.error(f"종목 비교 실패: {e}")
            return pd.DataFrame()
    
    def get_sector_summary(self, market: str = 'KOSPI') -> Dict:
        """
        업종별 요약 통계
        
        Args:
            market: 시장 구분
            
        Returns:
            업종별 통계 딕셔너리
        """
        try:
            summary = {}
            
            for sector in ['ALL', '반도체', '자동차', '은행']:
                recommendations = self.get_top_recommendations(
                    market=market,
                    sector=sector,
                    top_n=5
                )
                
                if not recommendations.empty:
                    summary[sector] = {
                        'count': len(recommendations),
                        'avg_probability': recommendations['up_probability'].mean(),
                        'avg_confidence': recommendations['confidence'].mean(),
                        'top_stock': recommendations.iloc[0]['name'] if len(recommendations) > 0 else None
                    }
            
            return summary
            
        except Exception as e:
            logger.error(f"업종 요약 실패: {e}")
            return {}
    
    def get_portfolio_suggestion(
        self, 
        investment_amount: float,
        risk_level: str = 'medium',
        market: str = 'KOSPI'
    ) -> Dict:
        """
        포트폴리오 제안
        
        Args:
            investment_amount: 투자 금액
            risk_level: 리스크 레벨 (low, medium, high)
            market: 시장 구분
            
        Returns:
            포트폴리오 제안 딕셔너리
        """
        try:
            # 리스크 레벨에 따른 종목 수
            stock_count = {
                'low': 10,
                'medium': 7,
                'high': 5
            }.get(risk_level, 7)
            
            # 추천 종목 조회
            recommendations = self.get_top_recommendations(market=market, top_n=stock_count)
            
            if recommendations.empty:
                return {}
            
            # 신뢰도 기반 가중치 계산
            recommendations['weight'] = recommendations['confidence'] * recommendations['up_probability']
            total_weight = recommendations['weight'].sum()
            recommendations['allocation'] = recommendations['weight'] / total_weight
            
            # 투자 금액 배분
            recommendations['investment'] = recommendations['allocation'] * investment_amount
            recommendations['shares'] = (
                recommendations['investment'] / recommendations['current_price']
            ).astype(int)
            
            portfolio = {
                'total_investment': investment_amount,
                'stock_count': len(recommendations),
                'risk_level': risk_level,
                'stocks': recommendations.to_dict('records'),
                'expected_return': float(recommendations['up_probability'].mean()),
                'diversification_score': float(1 / len(recommendations))
            }
            
            return portfolio
            
        except Exception as e:
            logger.error(f"포트폴리오 제안 실패: {e}")
            return {}


# 싱글톤 인스턴스
recommender = StockRecommender()
