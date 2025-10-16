"""
기본 테스트
"""

import sys
import os

# 프로젝트 루트를 경로에 추가
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def test_imports():
    """모듈 임포트 테스트"""
    try:
        from config import settings
        from data import data_loader, preprocessor
        from models import predictor, recommender
        from utils import visualizer
        
        print("✅ 모든 모듈 임포트 성공")
        return True
    except Exception as e:
        print(f"❌ 모듈 임포트 실패: {e}")
        return False


def test_settings():
    """설정 테스트"""
    try:
        from config import settings
        
        assert settings.TOP_N_STOCKS > 0
        assert settings.PREDICTION_DAYS > 0
        assert settings.DEFAULT_MARKET in ['KOSPI', 'KOSDAQ']
        
        print("✅ 설정 테스트 통과")
        return True
    except Exception as e:
        print(f"❌ 설정 테스트 실패: {e}")
        return False


def test_data_loader():
    """데이터 로더 테스트"""
    try:
        from data import data_loader
        
        # 종목 리스트 조회
        stock_list = data_loader.get_stock_list('KOSPI')
        assert not stock_list.empty
        assert 'ticker' in stock_list.columns
        assert 'name' in stock_list.columns
        
        # 가격 데이터 조회 (샘플)
        ticker = stock_list.iloc[0]['ticker']
        price_data = data_loader.get_stock_price(ticker, days=30)
        assert not price_data.empty
        assert 'close' in price_data.columns
        
        print("✅ 데이터 로더 테스트 통과")
        return True
    except Exception as e:
        print(f"❌ 데이터 로더 테스트 실패: {e}")
        return False


def test_preprocessor():
    """전처리 테스트"""
    try:
        from data import data_loader, preprocessor
        
        # 샘플 데이터
        stock_list = data_loader.get_stock_list('KOSPI')
        ticker = stock_list.iloc[0]['ticker']
        df = data_loader.get_stock_price(ticker, days=100)
        
        # 기술적 지표 추가
        df = preprocessor.add_technical_indicators(df)
        assert 'ma5' in df.columns
        assert 'rsi' in df.columns
        assert 'macd' in df.columns
        
        print("✅ 전처리 테스트 통과")
        return True
    except Exception as e:
        print(f"❌ 전처리 테스트 실패: {e}")
        return False


def test_predictor():
    """예측 모델 테스트"""
    try:
        from data import data_loader, preprocessor
        from models import predictor
        
        # 샘플 데이터
        stock_list = data_loader.get_stock_list('KOSPI')
        ticker = stock_list.iloc[0]['ticker']
        df = data_loader.get_stock_price(ticker, days=200)
        
        # 전처리
        df = preprocessor.add_technical_indicators(df)
        df = preprocessor.create_features(df)
        
        # 학습 데이터 준비
        X, y = preprocessor.prepare_training_data(df)
        
        if len(X) >= 100:
            # 모델 학습
            result = predictor.train_model(X, y)
            assert result.get('success', False)
            
            # 예측
            latest_features = preprocessor.get_latest_features(df)
            prob = predictor.predict_probability(latest_features)
            assert 0 <= prob <= 1
            
            print("✅ 예측 모델 테스트 통과")
        else:
            print("⚠️ 예측 모델 테스트 스킵 (데이터 부족)")
        
        return True
    except Exception as e:
        print(f"❌ 예측 모델 테스트 실패: {e}")
        return False


def run_all_tests():
    """모든 테스트 실행"""
    print("=" * 50)
    print("테스트 시작")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_settings,
        test_data_loader,
        test_preprocessor,
        test_predictor,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        print(f"\n실행: {test.__name__}")
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ 테스트 실행 중 오류: {e}")
            failed += 1
    
    print("\n" + "=" * 50)
    print(f"테스트 결과: {passed}개 통과, {failed}개 실패")
    print("=" * 50)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
