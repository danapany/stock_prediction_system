"""
시각화 모듈
차트 및 그래프 생성
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StockVisualizer:
    """주식 시각화 클래스"""
    
    def __init__(self):
        self.default_height = 500
        self.color_scheme = {
            'up': '#00C853',
            'down': '#D50000',
            'neutral': '#FFA726',
            'primary': '#1976D2',
            'secondary': '#424242'
        }
    
    def plot_candlestick(
        self, 
        df: pd.DataFrame,
        title: str = '주가 차트',
        height: int = None
    ) -> go.Figure:
        """
        캔들스틱 차트 생성
        
        Args:
            df: 가격 데이터 DataFrame
            title: 차트 제목
            height: 차트 높이
            
        Returns:
            Plotly Figure
        """
        try:
            if height is None:
                height = self.default_height
            
            fig = go.Figure(data=[
                go.Candlestick(
                    x=df['date'],
                    open=df['open'],
                    high=df['high'],
                    low=df['low'],
                    close=df['close'],
                    name='주가',
                    increasing_line_color=self.color_scheme['up'],
                    decreasing_line_color=self.color_scheme['down']
                )
            ])
            
            # 이동평균선 추가
            if 'ma5' in df.columns:
                fig.add_trace(go.Scatter(
                    x=df['date'],
                    y=df['ma5'],
                    name='MA5',
                    line=dict(color='orange', width=1)
                ))
            
            if 'ma20' in df.columns:
                fig.add_trace(go.Scatter(
                    x=df['date'],
                    y=df['ma20'],
                    name='MA20',
                    line=dict(color='blue', width=1)
                ))
            
            if 'ma60' in df.columns:
                fig.add_trace(go.Scatter(
                    x=df['date'],
                    y=df['ma60'],
                    name='MA60',
                    line=dict(color='purple', width=1)
                ))
            
            fig.update_layout(
                title=title,
                yaxis_title='주가',
                xaxis_title='날짜',
                height=height,
                xaxis_rangeslider_visible=False,
                hovermode='x unified'
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"캔들스틱 차트 생성 실패: {e}")
            return go.Figure()
    
    def plot_volume(
        self,
        df: pd.DataFrame,
        title: str = '거래량',
        height: int = None
    ) -> go.Figure:
        """
        거래량 차트 생성
        
        Args:
            df: 가격 데이터 DataFrame
            title: 차트 제목
            height: 차트 높이
            
        Returns:
            Plotly Figure
        """
        try:
            if height is None:
                height = 200
            
            # 거래량 색상 (전일 대비 상승/하락)
            colors = [
                self.color_scheme['up'] if df['close'].iloc[i] >= df['close'].iloc[i-1] 
                else self.color_scheme['down']
                for i in range(len(df))
            ]
            colors[0] = self.color_scheme['neutral']  # 첫 날은 중립
            
            fig = go.Figure(data=[
                go.Bar(
                    x=df['date'],
                    y=df['volume'],
                    name='거래량',
                    marker_color=colors
                )
            ])
            
            fig.update_layout(
                title=title,
                yaxis_title='거래량',
                xaxis_title='날짜',
                height=height,
                hovermode='x unified'
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"거래량 차트 생성 실패: {e}")
            return go.Figure()
    
    def plot_technical_indicators(
        self,
        df: pd.DataFrame,
        title: str = '기술적 지표',
        height: int = None
    ) -> go.Figure:
        """
        기술적 지표 차트 생성 (RSI, MACD)
        
        Args:
            df: 가격 데이터 DataFrame
            title: 차트 제목
            height: 차트 높이
            
        Returns:
            Plotly Figure
        """
        try:
            if height is None:
                height = 400
            
            # 서브플롯 생성
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('RSI', 'MACD'),
                vertical_spacing=0.15,
                row_heights=[0.5, 0.5]
            )
            
            # RSI
            if 'rsi' in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df['date'],
                        y=df['rsi'],
                        name='RSI',
                        line=dict(color='purple', width=2)
                    ),
                    row=1, col=1
                )
                
                # 과매수/과매도 구간
                fig.add_hline(y=70, line_dash="dash", line_color="red", row=1, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="green", row=1, col=1)
            
            # MACD
            if 'macd' in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df['date'],
                        y=df['macd'],
                        name='MACD',
                        line=dict(color='blue', width=2)
                    ),
                    row=2, col=1
                )
            
            if 'macd_signal' in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df['date'],
                        y=df['macd_signal'],
                        name='Signal',
                        line=dict(color='orange', width=2)
                    ),
                    row=2, col=1
                )
            
            if 'macd_diff' in df.columns:
                colors = [
                    self.color_scheme['up'] if val >= 0 else self.color_scheme['down']
                    for val in df['macd_diff']
                ]
                fig.add_trace(
                    go.Bar(
                        x=df['date'],
                        y=df['macd_diff'],
                        name='Histogram',
                        marker_color=colors
                    ),
                    row=2, col=1
                )
            
            fig.update_layout(
                title=title,
                height=height,
                hovermode='x unified',
                showlegend=True
            )
            
            fig.update_yaxes(title_text="RSI", row=1, col=1)
            fig.update_yaxes(title_text="MACD", row=2, col=1)
            
            return fig
            
        except Exception as e:
            logger.error(f"기술적 지표 차트 생성 실패: {e}")
            return go.Figure()
    
    def plot_recommendations(
        self,
        df_recommendations: pd.DataFrame,
        title: str = 'Top 추천 종목',
        height: int = None
    ) -> go.Figure:
        """
        추천 종목 차트 생성
        
        Args:
            df_recommendations: 추천 종목 DataFrame
            title: 차트 제목
            height: 차트 높이
            
        Returns:
            Plotly Figure
        """
        try:
            if height is None:
                height = 500
            
            if df_recommendations.empty:
                return go.Figure()
            
            # 상위 10개만 표시
            df_top = df_recommendations.head(10).copy()
            
            fig = go.Figure(data=[
                go.Bar(
                    x=df_top['up_probability'],
                    y=df_top['name'],
                    orientation='h',
                    text=df_top['up_probability'].apply(lambda x: f'{x*100:.1f}%'),
                    textposition='auto',
                    marker=dict(
                        color=df_top['up_probability'],
                        colorscale='RdYlGn',
                        showscale=True,
                        colorbar=dict(title='확률')
                    )
                )
            ])
            
            fig.update_layout(
                title=title,
                xaxis_title='상승 확률',
                yaxis_title='종목명',
                height=height,
                yaxis={'categoryorder': 'total ascending'}
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"추천 종목 차트 생성 실패: {e}")
            return go.Figure()
    
    def plot_confidence_scatter(
        self,
        df_recommendations: pd.DataFrame,
        title: str = '확률 vs 신뢰도',
        height: int = None
    ) -> go.Figure:
        """
        확률-신뢰도 산점도 생성
        
        Args:
            df_recommendations: 추천 종목 DataFrame
            title: 차트 제목
            height: 차트 높이
            
        Returns:
            Plotly Figure
        """
        try:
            if height is None:
                height = 500
            
            if df_recommendations.empty:
                return go.Figure()
            
            fig = px.scatter(
                df_recommendations,
                x='up_probability',
                y='confidence',
                text='name',
                color='up_probability',
                size='market_cap',
                color_continuous_scale='RdYlGn',
                title=title,
                labels={
                    'up_probability': '상승 확률',
                    'confidence': '신뢰도',
                    'market_cap': '시가총액'
                }
            )
            
            fig.update_traces(textposition='top center')
            fig.update_layout(height=height)
            
            return fig
            
        except Exception as e:
            logger.error(f"산점도 생성 실패: {e}")
            return go.Figure()
    
    def plot_sector_performance(
        self,
        sector_summary: dict,
        title: str = '업종별 성과',
        height: int = None
    ) -> go.Figure:
        """
        업종별 성과 차트
        
        Args:
            sector_summary: 업종별 요약 딕셔너리
            title: 차트 제목
            height: 차트 높이
            
        Returns:
            Plotly Figure
        """
        try:
            if height is None:
                height = 400
            
            if not sector_summary:
                return go.Figure()
            
            sectors = list(sector_summary.keys())
            avg_probs = [sector_summary[s]['avg_probability'] for s in sectors]
            
            fig = go.Figure(data=[
                go.Bar(
                    x=sectors,
                    y=avg_probs,
                    text=[f'{p*100:.1f}%' for p in avg_probs],
                    textposition='auto',
                    marker_color=self.color_scheme['primary']
                )
            ])
            
            fig.update_layout(
                title=title,
                xaxis_title='업종',
                yaxis_title='평균 상승 확률',
                height=height
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"업종별 성과 차트 생성 실패: {e}")
            return go.Figure()
    
    def plot_portfolio_allocation(
        self,
        portfolio: dict,
        title: str = '포트폴리오 배분',
        height: int = None
    ) -> go.Figure:
        """
        포트폴리오 배분 차트
        
        Args:
            portfolio: 포트폴리오 딕셔너리
            title: 차트 제목
            height: 차트 높이
            
        Returns:
            Plotly Figure
        """
        try:
            if height is None:
                height = 500
            
            if not portfolio or 'stocks' not in portfolio:
                return go.Figure()
            
            stocks = portfolio['stocks']
            names = [s['name'] for s in stocks]
            allocations = [s['allocation'] * 100 for s in stocks]
            
            fig = go.Figure(data=[
                go.Pie(
                    labels=names,
                    values=allocations,
                    hole=0.4,
                    textinfo='label+percent',
                    textposition='outside'
                )
            ])
            
            fig.update_layout(
                title=title,
                height=height,
                showlegend=True
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"포트폴리오 배분 차트 생성 실패: {e}")
            return go.Figure()


# 싱글톤 인스턴스
visualizer = StockVisualizer()
