"""
주가 예측 모델 모듈
머신러닝 기반 주가 상승/하락 예측
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
import logging
from datetime import datetime

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    import joblib
except ImportError:
    RandomForestClassifier = None
    print("scikit-learn이 설치되지 않았습니다. pip install scikit-learn")

from config.settings import settings
from data.preprocessor import preprocessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StockPredictor:
    """주가 예측 클래스"""
    
    def __init__(self):
        self.model = None
        self.model_trained_at = None
        self.feature_importance = None
    
    def train_model(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """
        모델 학습
        
        Args:
            X: 특성 DataFrame
            y: 타겟 Series
            
        Returns:
            학습 결과 딕셔너리
        """
        try:
            if RandomForestClassifier is None:
                logger.error("scikit-learn이 설치되지 않았습니다")
                return {'success': False, 'error': 'scikit-learn not installed'}
            
            # 데이터 분할
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # 모델 초기화
            if settings.USE_ADVANCED_MODEL:
                # 고급 모델 (더 많은 트리, 깊이)
                self.model = RandomForestClassifier(
                    n_estimators=200,
                    max_depth=15,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1
                )
            else:
                # 기본 모델
                self.model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42,
                    n_jobs=-1
                )
            
            # 학습
            logger.info("모델 학습 시작...")
            self.model.fit(X_train, y_train)
            
            # 예측
            y_pred_train = self.model.predict(X_train)
            y_pred_test = self.model.predict(X_test)
            
            # 평가
            train_accuracy = accuracy_score(y_train, y_pred_train)
            test_accuracy = accuracy_score(y_test, y_pred_test)
            test_precision = precision_score(y_test, y_pred_test, zero_division=0)
            test_recall = recall_score(y_test, y_pred_test, zero_division=0)
            test_f1 = f1_score(y_test, y_pred_test, zero_division=0)
            
            # 피처 중요도
            self.feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            self.model_trained_at = datetime.now()
            
            results = {
                'success': True,
                'train_accuracy': train_accuracy,
                'test_accuracy': test_accuracy,
                'precision': test_precision,
                'recall': test_recall,
                'f1_score': test_f1,
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'trained_at': self.model_trained_at
            }
            
            logger.info(f"모델 학습 완료 - 테스트 정확도: {test_accuracy:.4f}")
            return results
            
        except Exception as e:
            logger.error(f"모델 학습 실패: {e}")
            return {'success': False, 'error': str(e)}
    
    def predict_probability(self, X: pd.DataFrame) -> float:
        """
        상승 확률 예측
        
        Args:
            X: 특성 DataFrame
            
        Returns:
            상승 확률 (0-1)
        """
        try:
            if self.model is None:
                logger.warning("학습된 모델이 없습니다")
                return 0.5  # 중립
            
            # 확률 예측
            proba = self.model.predict_proba(X)
            
            # 상승 확률 (클래스 1)
            up_probability = proba[0][1]
            
            return float(up_probability)
            
        except Exception as e:
            logger.error(f"예측 실패: {e}")
            return 0.5
    
    def predict_direction(self, X: pd.DataFrame) -> str:
        """
        상승/하락 방향 예측
        
        Args:
            X: 특성 DataFrame
            
        Returns:
            'UP' 또는 'DOWN'
        """
        proba = self.predict_probability(X)
        return 'UP' if proba >= 0.5 else 'DOWN'
    
    def get_prediction_confidence(self, X: pd.DataFrame) -> float:
        """
        예측 신뢰도
        
        Args:
            X: 특성 DataFrame
            
        Returns:
            신뢰도 (0-1)
        """
        proba = self.predict_probability(X)
        # 0.5에서 멀수록 신뢰도 높음
        confidence = abs(proba - 0.5) * 2
        return confidence
    
    def analyze_stock(self, df: pd.DataFrame) -> Dict:
        """
        종목 분석
        
        Args:
            df: 가격 데이터 DataFrame
            
        Returns:
            분석 결과 딕셔너리
        """
        try:
            # 기술적 지표 추가
            df = preprocessor.add_technical_indicators(df)
            df = preprocessor.create_features(df)
            
            # 최신 피처 추출
            latest_features = preprocessor.get_latest_features(df)
            
            if latest_features.empty:
                logger.warning("피처 추출 실패")
                return self._get_neutral_analysis()
            
            # 예측
            up_probability = self.predict_probability(latest_features)
            direction = self.predict_direction(latest_features)
            confidence = self.get_prediction_confidence(latest_features)
            
            # 최신 가격 정보
            latest_row = df.iloc[-1]
            
            analysis = {
                'prediction': {
                    'direction': direction,
                    'up_probability': up_probability,
                    'confidence': confidence,
                },
                'current_price': float(latest_row['close']),
                'technical_indicators': {
                    'rsi': float(latest_row.get('rsi', 50)),
                    'macd': float(latest_row.get('macd', 0)),
                    'ma5': float(latest_row.get('ma5', 0)),
                    'ma20': float(latest_row.get('ma20', 0)),
                    'ma60': float(latest_row.get('ma60', 0)),
                },
                'signals': self._generate_signals(df)
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"종목 분석 실패: {e}")
            return self._get_neutral_analysis()
    
    def _generate_signals(self, df: pd.DataFrame) -> Dict:
        """매매 신호 생성"""
        try:
            latest = df.iloc[-1]
            
            signals = {
                'trend': 'NEUTRAL',
                'momentum': 'NEUTRAL',
                'volume': 'NORMAL',
                'overall': 'HOLD'
            }
            
            # 추세 신호
            if latest.get('ma5', 0) > latest.get('ma20', 0) > latest.get('ma60', 0):
                signals['trend'] = 'BULLISH'
            elif latest.get('ma5', 0) < latest.get('ma20', 0) < latest.get('ma60', 0):
                signals['trend'] = 'BEARISH'
            
            # 모멘텀 신호
            rsi = latest.get('rsi', 50)
            if rsi > 70:
                signals['momentum'] = 'OVERBOUGHT'
            elif rsi < 30:
                signals['momentum'] = 'OVERSOLD'
            
            # 거래량 신호
            volume_ratio = latest.get('volume_ratio', 1)
            if volume_ratio > 1.5:
                signals['volume'] = 'HIGH'
            elif volume_ratio < 0.5:
                signals['volume'] = 'LOW'
            
            # 종합 신호
            if signals['trend'] == 'BULLISH' and signals['momentum'] != 'OVERBOUGHT':
                signals['overall'] = 'BUY'
            elif signals['trend'] == 'BEARISH' and signals['momentum'] != 'OVERSOLD':
                signals['overall'] = 'SELL'
            
            return signals
            
        except Exception as e:
            logger.error(f"신호 생성 실패: {e}")
            return {
                'trend': 'NEUTRAL',
                'momentum': 'NEUTRAL',
                'volume': 'NORMAL',
                'overall': 'HOLD'
            }
    
    def _get_neutral_analysis(self) -> Dict:
        """중립적 분석 결과 반환"""
        return {
            'prediction': {
                'direction': 'NEUTRAL',
                'up_probability': 0.5,
                'confidence': 0.0,
            },
            'current_price': 0,
            'technical_indicators': {
                'rsi': 50,
                'macd': 0,
                'ma5': 0,
                'ma20': 0,
                'ma60': 0,
            },
            'signals': {
                'trend': 'NEUTRAL',
                'momentum': 'NEUTRAL',
                'volume': 'NORMAL',
                'overall': 'HOLD'
            }
        }
    
    def save_model(self, filepath: str):
        """모델 저장"""
        try:
            if self.model is not None:
                joblib.dump(self.model, filepath)
                logger.info(f"모델 저장 완료: {filepath}")
        except Exception as e:
            logger.error(f"모델 저장 실패: {e}")
    
    def load_model(self, filepath: str):
        """모델 로드"""
        try:
            self.model = joblib.load(filepath)
            logger.info(f"모델 로드 완료: {filepath}")
        except Exception as e:
            logger.error(f"모델 로드 실패: {e}")


# 싱글톤 인스턴스
predictor = StockPredictor()
