"""
데이터 전처리 모듈
주가 데이터를 분석 및 예측에 적합한 형태로 변환합니다.
"""

import pandas as pd
import numpy as np
from typing import Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StockDataPreprocessor:
    """주식 데이터 전처리 클래스"""
    
    def __init__(self):
        pass
    
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        기술적 지표 추가
        
        Args:
            df: 가격 데이터 DataFrame
            
        Returns:
            지표가 추가된 DataFrame
        """
        try:
            df = df.copy()
            
            # 이동평균선
            df['ma5'] = df['close'].rolling(window=5).mean()
            df['ma20'] = df['close'].rolling(window=20).mean()
            df['ma60'] = df['close'].rolling(window=60).mean()
            
            # 볼린저 밴드
            df['bb_middle'] = df['close'].rolling(window=20).mean()
            bb_std = df['close'].rolling(window=20).std()
            df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
            df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
            
            # RSI (Relative Strength Index)
            df['rsi'] = self._calculate_rsi(df['close'])
            
            # MACD
            exp1 = df['close'].ewm(span=12, adjust=False).mean()
            exp2 = df['close'].ewm(span=26, adjust=False).mean()
            df['macd'] = exp1 - exp2
            df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
            df['macd_diff'] = df['macd'] - df['macd_signal']
            
            # 거래량 이동평균
            df['volume_ma5'] = df['volume'].rolling(window=5).mean()
            df['volume_ma20'] = df['volume'].rolling(window=20).mean()
            
            # 가격 변화율
            df['price_change'] = df['close'].pct_change()
            df['price_change_5d'] = df['close'].pct_change(periods=5)
            df['price_change_20d'] = df['close'].pct_change(periods=20)
            
            # 변동성
            df['volatility'] = df['close'].rolling(window=20).std()
            
            logger.info("기술적 지표 추가 완료")
            return df
            
        except Exception as e:
            logger.error(f"기술적 지표 추가 실패: {e}")
            return df
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """RSI 계산"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        머신러닝을 위한 피처 생성
        
        Args:
            df: 지표가 포함된 DataFrame
            
        Returns:
            피처가 추가된 DataFrame
        """
        try:
            df = df.copy()
            
            # 타겟 변수: 다음 날 상승 여부
            df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
            
            # 추가 피처
            df['high_low_ratio'] = df['high'] / df['low']
            df['close_open_ratio'] = df['close'] / df['open']
            
            # 이동평균 교차
            df['ma5_20_cross'] = (df['ma5'] > df['ma20']).astype(int)
            df['ma20_60_cross'] = (df['ma20'] > df['ma60']).astype(int)
            
            # 볼린저 밴드 위치
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            
            # 거래량 변화
            df['volume_ratio'] = df['volume'] / df['volume_ma20']
            
            # 추세 강도
            df['trend_strength'] = abs(df['close'] - df['ma20']) / df['ma20']
            
            logger.info("피처 생성 완료")
            return df
            
        except Exception as e:
            logger.error(f"피처 생성 실패: {e}")
            return df
    
    def prepare_training_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        학습 데이터 준비
        
        Args:
            df: 전처리된 DataFrame
            
        Returns:
            (특성 DataFrame, 타겟 Series)
        """
        try:
            # NaN 제거
            df = df.dropna()
            
            # 피처 컬럼 선택
            feature_columns = [
                'ma5', 'ma20', 'ma60',
                'rsi', 'macd', 'macd_signal', 'macd_diff',
                'volume_ma5', 'volume_ma20',
                'price_change', 'price_change_5d', 'price_change_20d',
                'volatility',
                'high_low_ratio', 'close_open_ratio',
                'ma5_20_cross', 'ma20_60_cross',
                'bb_position', 'volume_ratio', 'trend_strength'
            ]
            
            # 존재하는 컬럼만 선택
            available_features = [col for col in feature_columns if col in df.columns]
            
            X = df[available_features]
            y = df['target']
            
            logger.info(f"학습 데이터 준비 완료: {len(X)}개 샘플, {len(available_features)}개 피처")
            return X, y
            
        except Exception as e:
            logger.error(f"학습 데이터 준비 실패: {e}")
            return pd.DataFrame(), pd.Series()
    
    def normalize_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        피처 정규화
        
        Args:
            X: 특성 DataFrame
            
        Returns:
            정규화된 DataFrame
        """
        try:
            X_normalized = X.copy()
            
            # 각 컬럼을 0-1 범위로 정규화
            for col in X_normalized.columns:
                min_val = X_normalized[col].min()
                max_val = X_normalized[col].max()
                
                if max_val > min_val:
                    X_normalized[col] = (X_normalized[col] - min_val) / (max_val - min_val)
            
            logger.info("피처 정규화 완료")
            return X_normalized
            
        except Exception as e:
            logger.error(f"피처 정규화 실패: {e}")
            return X
    
    def get_latest_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        최신 데이터의 피처 추출 (예측용)
        
        Args:
            df: 전처리된 DataFrame
            
        Returns:
            최신 피처 DataFrame
        """
        try:
            # 마지막 행 선택
            latest = df.iloc[[-1]]
            
            feature_columns = [
                'ma5', 'ma20', 'ma60',
                'rsi', 'macd', 'macd_signal', 'macd_diff',
                'volume_ma5', 'volume_ma20',
                'price_change', 'price_change_5d', 'price_change_20d',
                'volatility',
                'high_low_ratio', 'close_open_ratio',
                'ma5_20_cross', 'ma20_60_cross',
                'bb_position', 'volume_ratio', 'trend_strength'
            ]
            
            # 존재하는 컬럼만 선택
            available_features = [col for col in feature_columns if col in latest.columns]
            
            return latest[available_features]
            
        except Exception as e:
            logger.error(f"최신 피처 추출 실패: {e}")
            return pd.DataFrame()


# 싱글톤 인스턴스
preprocessor = StockDataPreprocessor()
