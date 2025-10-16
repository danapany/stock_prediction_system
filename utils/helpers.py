"""
유틸리티 헬퍼 함수 모듈
"""

import pandas as pd
import numpy as np
from typing import Union, List
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def format_currency(value: Union[int, float], currency: str = 'KRW') -> str:
    """
    통화 포맷팅
    
    Args:
        value: 금액
        currency: 통화 단위
        
    Returns:
        포맷된 문자열
    """
    try:
        if pd.isna(value):
            return 'N/A'
        
        if currency == 'KRW':
            if value >= 1_000_000_000_000:  # 조
                return f"{value/1_000_000_000_000:.2f}조원"
            elif value >= 100_000_000:  # 억
                return f"{value/100_000_000:.0f}억원"
            elif value >= 10_000:  # 만
                return f"{value/10_000:.0f}만원"
            else:
                return f"{value:,.0f}원"
        else:
            return f"${value:,.2f}"
            
    except Exception as e:
        logger.error(f"통화 포맷팅 실패: {e}")
        return str(value)


def format_percentage(value: float, decimal_places: int = 2) -> str:
    """
    퍼센트 포맷팅
    
    Args:
        value: 값 (0-1 또는 0-100)
        decimal_places: 소수점 자리수
        
    Returns:
        포맷된 문자열
    """
    try:
        if pd.isna(value):
            return 'N/A'
        
        # 0-1 범위면 100을 곱함
        if 0 <= value <= 1:
            value = value * 100
        
        return f"{value:.{decimal_places}f}%"
        
    except Exception as e:
        logger.error(f"퍼센트 포맷팅 실패: {e}")
        return str(value)


def format_number(value: Union[int, float], decimal_places: int = 2) -> str:
    """
    숫자 포맷팅
    
    Args:
        value: 숫자
        decimal_places: 소수점 자리수
        
    Returns:
        포맷된 문자열
    """
    try:
        if pd.isna(value):
            return 'N/A'
        
        if isinstance(value, int):
            return f"{value:,}"
        else:
            return f"{value:,.{decimal_places}f}"
            
    except Exception as e:
        logger.error(f"숫자 포맷팅 실패: {e}")
        return str(value)


def calculate_returns(prices: pd.Series) -> pd.Series:
    """
    수익률 계산
    
    Args:
        prices: 가격 Series
        
    Returns:
        수익률 Series
    """
    try:
        returns = prices.pct_change()
        return returns
        
    except Exception as e:
        logger.error(f"수익률 계산 실패: {e}")
        return pd.Series()


def calculate_volatility(prices: pd.Series, window: int = 20) -> float:
    """
    변동성 계산
    
    Args:
        prices: 가격 Series
        window: 윈도우 크기
        
    Returns:
        변동성 (연율화)
    """
    try:
        returns = calculate_returns(prices)
        volatility = returns.rolling(window=window).std() * np.sqrt(252)
        return float(volatility.iloc[-1])
        
    except Exception as e:
        logger.error(f"변동성 계산 실패: {e}")
        return 0.0


def calculate_sharpe_ratio(
    prices: pd.Series, 
    risk_free_rate: float = 0.03,
    window: int = 252
) -> float:
    """
    샤프 비율 계산
    
    Args:
        prices: 가격 Series
        risk_free_rate: 무위험 수익률 (연율)
        window: 윈도우 크기
        
    Returns:
        샤프 비율
    """
    try:
        returns = calculate_returns(prices)
        
        # 연율 수익률
        annual_return = returns.mean() * 252
        
        # 연율 변동성
        annual_volatility = returns.std() * np.sqrt(252)
        
        if annual_volatility == 0:
            return 0.0
        
        sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility
        return float(sharpe_ratio)
        
    except Exception as e:
        logger.error(f"샤프 비율 계산 실패: {e}")
        return 0.0


def get_signal_color(signal: str) -> str:
    """
    신호에 따른 색상 반환
    
    Args:
        signal: 신호 문자열
        
    Returns:
        색상 코드
    """
    color_map = {
        'BUY': '🟢',
        'SELL': '🔴',
        'HOLD': '🟡',
        'BULLISH': '🟢',
        'BEARISH': '🔴',
        'NEUTRAL': '⚪',
        'UP': '⬆️',
        'DOWN': '⬇️',
        'HIGH': '🔺',
        'LOW': '🔻',
        'NORMAL': '➖',
        'OVERBOUGHT': '⚠️',
        'OVERSOLD': '⚠️',
    }
    return color_map.get(signal, '⚪')


def get_confidence_emoji(confidence: float) -> str:
    """
    신뢰도에 따른 이모지 반환
    
    Args:
        confidence: 신뢰도 (0-1)
        
    Returns:
        이모지
    """
    if confidence >= 0.8:
        return '🔥'
    elif confidence >= 0.6:
        return '✨'
    elif confidence >= 0.4:
        return '⭐'
    else:
        return '💫'


def get_date_range(days: int) -> tuple:
    """
    날짜 범위 계산
    
    Args:
        days: 일수
        
    Returns:
        (시작일, 종료일) 튜플
    """
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        return start_date, end_date
        
    except Exception as e:
        logger.error(f"날짜 범위 계산 실패: {e}")
        return datetime.now(), datetime.now()


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    안전한 나눗셈
    
    Args:
        numerator: 분자
        denominator: 분모
        default: 기본값
        
    Returns:
        나눗셈 결과
    """
    try:
        if denominator == 0:
            return default
        return numerator / denominator
        
    except Exception as e:
        logger.error(f"나눗셈 실패: {e}")
        return default


def truncate_text(text: str, max_length: int = 50) -> str:
    """
    텍스트 자르기
    
    Args:
        text: 원본 텍스트
        max_length: 최대 길이
        
    Returns:
        잘린 텍스트
    """
    try:
        if len(text) <= max_length:
            return text
        return text[:max_length-3] + '...'
        
    except Exception as e:
        logger.error(f"텍스트 자르기 실패: {e}")
        return text


def get_market_status() -> str:
    """
    시장 상태 확인
    
    Returns:
        시장 상태 문자열
    """
    try:
        now = datetime.now()
        
        # 한국 시장 시간 (09:00 - 15:30)
        if now.weekday() >= 5:  # 주말
            return '휴장'
        
        market_open = now.replace(hour=9, minute=0, second=0)
        market_close = now.replace(hour=15, minute=30, second=0)
        
        if market_open <= now <= market_close:
            return '개장'
        else:
            return '휴장'
            
    except Exception as e:
        logger.error(f"시장 상태 확인 실패: {e}")
        return '알 수 없음'


def filter_dataframe(
    df: pd.DataFrame,
    filters: dict
) -> pd.DataFrame:
    """
    DataFrame 필터링
    
    Args:
        df: 원본 DataFrame
        filters: 필터 조건 딕셔너리
        
    Returns:
        필터링된 DataFrame
    """
    try:
        filtered_df = df.copy()
        
        for column, condition in filters.items():
            if column not in filtered_df.columns:
                continue
            
            if isinstance(condition, dict):
                # 범위 필터
                if 'min' in condition:
                    filtered_df = filtered_df[filtered_df[column] >= condition['min']]
                if 'max' in condition:
                    filtered_df = filtered_df[filtered_df[column] <= condition['max']]
            elif isinstance(condition, (list, tuple)):
                # 리스트 필터
                filtered_df = filtered_df[filtered_df[column].isin(condition)]
            else:
                # 값 필터
                filtered_df = filtered_df[filtered_df[column] == condition]
        
        return filtered_df
        
    except Exception as e:
        logger.error(f"DataFrame 필터링 실패: {e}")
        return df
