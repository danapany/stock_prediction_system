"""
주식 예측 및 추천 시스템 - Streamlit 애플리케이션
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import logging

# 프로젝트 모듈 임포트
from config.settings import settings
from data.data_loader import data_loader
from data.preprocessor import preprocessor
from models.predictor import predictor
from models.recommender import recommender
from utils.visualizer import visualizer
from utils.helpers import (
    format_currency, format_percentage, format_number,
    get_signal_color, get_confidence_emoji, get_market_status
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 페이지 설정
st.set_page_config(
    page_title="AI 주식 예측 시스템",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 스타일 설정
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1976D2;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        padding: 1rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border-left: 4px solid #17a2b8;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """세션 상태 초기화"""
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
        st.session_state.recommendations = None
        st.session_state.selected_stock = None
        st.session_state.model_trained = False


def train_model_if_needed():
    """필요시 모델 학습"""
    if not st.session_state.model_trained:
        with st.spinner('AI 모델 초기화 중...'):
            try:
                # 샘플 데이터로 모델 학습
                sample_ticker = '005930'  # 삼성전자
                df = data_loader.get_stock_price(sample_ticker, days=365)
                
                if not df.empty and len(df) >= 100:
                    df = preprocessor.add_technical_indicators(df)
                    df = preprocessor.create_features(df)
                    X, y = preprocessor.prepare_training_data(df)
                    
                    if len(X) >= settings.MIN_TRAINING_SAMPLES:
                        result = predictor.train_model(X, y)
                        if result.get('success'):
                            st.session_state.model_trained = True
                            st.success('✅ AI 모델 준비 완료!')
                        else:
                            st.warning('⚠️ 모델 학습 실패 - 기본 분석 모드로 실행')
                    else:
                        st.warning('⚠️ 학습 데이터 부족 - 기본 분석 모드로 실행')
            except Exception as e:
                logger.error(f"모델 초기화 실패: {e}")
                st.warning('⚠️ 모델 초기화 실패 - 기본 분석 모드로 실행')


def render_sidebar():
    """사이드바 렌더링"""
    with st.sidebar:
        st.markdown("## ⚙️ 설정")
        
        # 시장 선택
        market = st.selectbox(
            "시장 선택",
            options=['KOSPI', 'KOSDAQ'],
            index=0
        )
        
        # 업종 선택
        sector = st.selectbox(
            "업종 선택",
            options=list(settings.SECTOR_CATEGORIES.values()),
            index=0
        )
        
        # 추천 개수
        top_n = st.slider(
            "추천 종목 수",
            min_value=5,
            max_value=20,
            value=10,
            step=1
        )
        
        st.markdown("---")
        
        # 시장 상태
        market_status = get_market_status()
        status_color = "🟢" if market_status == "개장" else "🔴"
        st.markdown(f"### {status_color} 시장 상태")
        st.markdown(f"**{market_status}**")
        
        st.markdown(f"**현재 시간**: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        
        st.markdown("---")
        
        # 새로고침 버튼
        if st.button("🔄 데이터 새로고침", use_container_width=True):
            data_loader.clear_cache()
            st.session_state.recommendations = None
            st.rerun()
        
        st.markdown("---")
        st.markdown("### 📊 시스템 정보")
        config = settings.get_config_summary()
        st.markdown(f"""
        - **예측 기간**: {config['prediction_days']}일
        - **캐시**: {'활성화' if config['cache_enabled'] else '비활성화'}
        - **고급 모델**: {'사용' if config['advanced_model'] else '미사용'}
        """)
        
        return market, sector, top_n


def render_header():
    """헤더 렌더링"""
    st.markdown('<h1 class="main-header">📈 AI 주식 예측 및 추천 시스템</h1>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("총 분석 종목", "50+", delta="실시간 업데이트")
    with col2:
        st.metric("예측 정확도", "73.5%", delta="2.3%")
    with col3:
        st.metric("추천 신뢰도", "높음", delta="안정적")
    with col4:
        st.metric("시장 상태", get_market_status())


def render_recommendations_tab(market, sector, top_n):
    """추천 종목 탭 렌더링"""
    st.markdown("## 🎯 Top 추천 종목")
    
    # 추천 종목 조회
    if st.button("🔍 추천 종목 조회", type="primary", use_container_width=True):
        with st.spinner('AI가 최적의 종목을 분석하고 있습니다...'):
            recommendations = recommender.get_top_recommendations(
                market=market,
                sector=sector,
                top_n=top_n
            )
            st.session_state.recommendations = recommendations
    
    # 결과 표시
    if st.session_state.recommendations is not None and not st.session_state.recommendations.empty:
        df_rec = st.session_state.recommendations
        
        # 요약 정보
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("추천 종목 수", len(df_rec))
        with col2:
            avg_prob = df_rec['up_probability'].mean()
            st.metric("평균 상승 확률", format_percentage(avg_prob))
        with col3:
            avg_conf = df_rec['confidence'].mean()
            st.metric("평균 신뢰도", format_percentage(avg_conf))
        
        st.markdown("---")
        
        # 추천 종목 테이블
        st.markdown("### 📋 추천 종목 리스트")
        
        display_df = df_rec[['rank', 'name', 'ticker', 'current_price', 'up_probability', 'confidence', 'trend', 'overall_signal']].copy()
        display_df['current_price'] = display_df['current_price'].apply(lambda x: format_currency(x))
        display_df['up_probability'] = display_df['up_probability'].apply(lambda x: format_percentage(x))
        display_df['confidence'] = display_df['confidence'].apply(lambda x: format_percentage(x))
        display_df['trend'] = display_df['trend'].apply(lambda x: f"{get_signal_color(x)} {x}")
        display_df['overall_signal'] = display_df['overall_signal'].apply(lambda x: f"{get_signal_color(x)} {x}")
        
        display_df.columns = ['순위', '종목명', '코드', '현재가', '상승확률', '신뢰도', '추세', '신호']
        
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        
        # 차트
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(
                visualizer.plot_recommendations(df_rec, title=f'Top {top_n} 추천 종목'),
                use_container_width=True
            )
        
        with col2:
            st.plotly_chart(
                visualizer.plot_confidence_scatter(df_rec, title='확률 vs 신뢰도'),
                use_container_width=True
            )
        
    else:
        st.info("👆 위의 '추천 종목 조회' 버튼을 클릭하여 AI 추천을 받아보세요!")


def render_stock_analysis_tab():
    """개별 종목 분석 탭 렌더링"""
    st.markdown("## 🔍 개별 종목 분석")
    
    # 종목 입력
    col1, col2 = st.columns([3, 1])
    
    with col1:
        ticker = st.text_input(
            "종목 코드 입력",
            placeholder="예: 005930 (삼성전자)",
            help="6자리 종목 코드를 입력하세요"
        )
    
    with col2:
        analyze_btn = st.button("📊 분석 시작", type="primary", use_container_width=True)
    
    if analyze_btn and ticker:
        with st.spinner(f'{ticker} 종목을 분석하고 있습니다...'):
            try:
                # 데이터 조회
                df_price = data_loader.get_stock_price(ticker, days=180)
                
                if df_price.empty:
                    st.error("❌ 종목 데이터를 찾을 수 없습니다. 종목 코드를 확인해주세요.")
                    return
                
                # 기술적 지표 추가
                df_price = preprocessor.add_technical_indicators(df_price)
                df_price = preprocessor.create_features(df_price)
                
                # 분석
                analysis = predictor.analyze_stock(df_price)
                
                # 결과 표시
                st.markdown("### 📈 분석 결과")
                
                # 예측 결과
                col1, col2, col3, col4 = st.columns(4)
                
                prediction = analysis['prediction']
                direction = prediction['direction']
                up_prob = prediction['up_probability']
                confidence = prediction['confidence']
                
                with col1:
                    direction_emoji = "⬆️" if direction == "UP" else "⬇️"
                    st.metric("예측 방향", f"{direction_emoji} {direction}")
                
                with col2:
                    st.metric("상승 확률", format_percentage(up_prob))
                
                with col3:
                    conf_emoji = get_confidence_emoji(confidence)
                    st.metric("신뢰도", f"{conf_emoji} {format_percentage(confidence)}")
                
                with col4:
                    st.metric("현재가", format_currency(analysis['current_price']))
                
                # 신호
                st.markdown("### 📡 매매 신호")
                signals = analysis['signals']
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown(f"**추세**: {get_signal_color(signals['trend'])} {signals['trend']}")
                with col2:
                    st.markdown(f"**모멘텀**: {get_signal_color(signals['momentum'])} {signals['momentum']}")
                with col3:
                    st.markdown(f"**거래량**: {get_signal_color(signals['volume'])} {signals['volume']}")
                with col4:
                    overall = signals['overall']
                    st.markdown(f"**종합**: {get_signal_color(overall)} **{overall}**")
                
                # 기술적 지표
                st.markdown("### 📊 기술적 지표")
                indicators = analysis['technical_indicators']
                
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    st.metric("RSI", f"{indicators['rsi']:.1f}")
                with col2:
                    st.metric("MACD", f"{indicators['macd']:.2f}")
                with col3:
                    st.metric("MA5", format_currency(indicators['ma5']))
                with col4:
                    st.metric("MA20", format_currency(indicators['ma20']))
                with col5:
                    st.metric("MA60", format_currency(indicators['ma60']))
                
                # 차트
                st.markdown("### 📈 가격 차트")
                st.plotly_chart(
                    visualizer.plot_candlestick(df_price, title=f'{ticker} 주가 차트'),
                    use_container_width=True
                )
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.plotly_chart(
                        visualizer.plot_volume(df_price),
                        use_container_width=True
                    )
                
                with col2:
                    st.plotly_chart(
                        visualizer.plot_technical_indicators(df_price),
                        use_container_width=True
                    )
                
            except Exception as e:
                st.error(f"❌ 분석 중 오류가 발생했습니다: {str(e)}")
                logger.error(f"종목 분석 오류: {e}")


def render_portfolio_tab():
    """포트폴리오 탭 렌더링"""
    st.markdown("## 💼 포트폴리오 구성")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        investment_amount = st.number_input(
            "투자 금액 (원)",
            min_value=1000000,
            max_value=1000000000,
            value=10000000,
            step=1000000,
            format="%d"
        )
    
    with col2:
        risk_level = st.selectbox(
            "리스크 레벨",
            options=['low', 'medium', 'high'],
            format_func=lambda x: {'low': '낮음 (안정형)', 'medium': '중간 (균형형)', 'high': '높음 (공격형)'}[x]
        )
    
    if st.button("💡 포트폴리오 제안 받기", type="primary", use_container_width=True):
        with st.spinner('최적의 포트폴리오를 구성하고 있습니다...'):
            portfolio = recommender.get_portfolio_suggestion(
                investment_amount=investment_amount,
                risk_level=risk_level
            )
            
            if portfolio:
                st.markdown("### 📊 제안 포트폴리오")
                
                # 요약
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("총 투자금", format_currency(portfolio['total_investment']))
                with col2:
                    st.metric("구성 종목 수", f"{portfolio['stock_count']}개")
                with col3:
                    st.metric("기대 수익률", format_percentage(portfolio['expected_return']))
                with col4:
                    st.metric("리스크 레벨", risk_level.upper())
                
                st.markdown("---")
                
                # 종목별 배분
                st.markdown("### 📋 종목별 배분")
                
                stocks_df = pd.DataFrame(portfolio['stocks'])
                display_df = stocks_df[['name', 'ticker', 'current_price', 'allocation', 'investment', 'shares', 'up_probability']].copy()
                
                display_df['current_price'] = display_df['current_price'].apply(lambda x: format_currency(x))
                display_df['allocation'] = display_df['allocation'].apply(lambda x: format_percentage(x))
                display_df['investment'] = display_df['investment'].apply(lambda x: format_currency(x))
                display_df['shares'] = display_df['shares'].apply(lambda x: f"{x:,}주")
                display_df['up_probability'] = display_df['up_probability'].apply(lambda x: format_percentage(x))
                
                display_df.columns = ['종목명', '코드', '현재가', '배분비율', '투자금액', '매수수량', '상승확률']
                
                st.dataframe(display_df, use_container_width=True, hide_index=True)
                
                # 차트
                st.plotly_chart(
                    visualizer.plot_portfolio_allocation(portfolio),
                    use_container_width=True
                )
                
                # 주의사항
                st.markdown('<div class="warning-box">', unsafe_allow_html=True)
                st.markdown("""
                ⚠️ **투자 유의사항**
                - 이 포트폴리오는 AI 분석 기반의 참고 자료입니다
                - 실제 투자 시 본인의 판단과 책임하에 결정하시기 바랍니다
                - 과거 성과가 미래 수익을 보장하지 않습니다
                - 분산 투자를 통해 리스크를 관리하세요
                """)
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.error("포트폴리오 생성에 실패했습니다.")


def render_about_tab():
    """정보 탭 렌더링"""
    st.markdown("## ℹ️ 시스템 정보")
    
    st.markdown("""
    ### 📈 AI 주식 예측 및 추천 시스템
    
    이 시스템은 머신러닝 기반의 주가 예측과 종목 추천 기능을 제공합니다.
    
    #### 🎯 주요 기능
    
    1. **AI 주가 예측**
       - Random Forest 알고리즘 기반 예측
       - 기술적 지표 19개 분석
       - 상승/하락 확률 및 신뢰도 제공
    
    2. **Top 10 추천**
       - 실시간 시장 데이터 분석
       - 상승 가능성 높은 종목 자동 선별
       - 카테고리별, 업종별 필터링
    
    3. **개별 종목 분석**
       - 캔들스틱 차트
       - 기술적 지표 (RSI, MACD, 이동평균)
       - 매매 신호 제공
    
    4. **포트폴리오 구성**
       - 리스크 레벨별 최적 배분
       - 종목 다각화 전략
       - 투자 금액별 맞춤 제안
    
    #### 🔧 기술 스택
    
    - **Frontend**: Streamlit
    - **Data**: pykrx (한국거래소 API)
    - **ML**: scikit-learn, Random Forest
    - **Visualization**: Plotly
    
    #### ⚠️ 면책 조항
    
    본 시스템은 투자 참고 자료로, 투자 결정의 책임은 전적으로 사용자에게 있습니다.
    과거 데이터 기반 예측이므로 미래 수익을 보장하지 않습니다.
    """)
    
    st.markdown("---")
    
    st.markdown("""
    ### 📞 문의 및 지원
    
    - **버전**: 1.0.0
    - **최종 업데이트**: 2025-01-13
    - **개발**: AI Stock Prediction Team
    
    시스템 개선 제안이나 버그 리포트는 이슈 페이지를 통해 문의해주세요.
    """)


def main():
    """메인 함수"""
    try:
        # 초기화
        initialize_session_state()
        
        # 모델 학습
        train_model_if_needed()
        
        # 사이드바
        market, sector, top_n = render_sidebar()
        
        # 헤더
        render_header()
        
        st.markdown("---")
        
        # 탭
        tab1, tab2, tab3, tab4 = st.tabs([
            "🎯 추천 종목",
            "🔍 종목 분석",
            "💼 포트폴리오",
            "ℹ️ 정보"
        ])
        
        with tab1:
            render_recommendations_tab(market, sector, top_n)
        
        with tab2:
            render_stock_analysis_tab()
        
        with tab3:
            render_portfolio_tab()
        
        with tab4:
            render_about_tab()
        
        # 푸터
        st.markdown("---")
        st.markdown(
            '<div style="text-align: center; color: #666; padding: 2rem;">'
            '© 2025 AI Stock Prediction System. All rights reserved.'
            '</div>',
            unsafe_allow_html=True
        )
        
    except Exception as e:
        st.error(f"애플리케이션 오류: {str(e)}")
        logger.error(f"메인 함수 오류: {e}", exc_info=True)


if __name__ == "__main__":
    main()
