import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import json
import pickle

# Page configuration
st.set_page_config(
    page_title="Elon Musk Trillionaire Projection",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .big-font {
        font-size:30px !important;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    merged_data = pd.read_csv('data/merged_data.csv', index_col=0, parse_dates=True)
    forecast_df = pd.read_csv('data/forecast_df.csv', index_col=0, parse_dates=True)
    
    scenarios = {}
    for scenario in ['bear', 'base', 'bull']:
        scenarios[scenario] = pd.read_csv(f'data/scenario_{scenario}.csv', 
                                         index_col=0, parse_dates=True)
    
    monte_carlo = pd.read_csv('data/monte_carlo_percentiles.csv', 
                              index_col=0, parse_dates=True)
    
    with open('data/parameters.json', 'r') as f:
        params = json.load(f)
    
    return merged_data, forecast_df, scenarios, monte_carlo, params

# Load enhanced Monte Carlo data
@st.cache_data
def load_enhanced_monte_carlo():
    """Load enhanced Monte Carlo results with extended projections"""
    with open('data/monte_carlo_all_paths.pkl', 'rb') as f:
        all_paths = pickle.load(f)
    
    with open('data/monte_carlo_sample_paths.pkl', 'rb') as f:
        sample_paths = pickle.load(f)
    
    with open('data/monte_carlo_summary.json', 'r') as f:
        summary = json.load(f)
    
    time_dist = pd.read_csv('data/monte_carlo_time_to_trillion.csv')
    
    return all_paths, sample_paths, summary, time_dist

merged_data, forecast_df, scenarios, monte_carlo, params = load_data()

# Sidebar
st.sidebar.title("🎯 Navigation")
page = st.sidebar.radio("Select Page", [
    "📊 Executive Summary",
    "📈 Historical Analysis", 
    "🎯 Trillionaire Timeline",
])

st.sidebar.markdown("---")
st.sidebar.markdown("### 📋 Quick Stats")
st.sidebar.metric("Current Net Worth", f"${params['current_networth']:.1f}B")
st.sidebar.metric("TSLA Price", f"${params['current_tsla_price']:.2f}")
st.sidebar.metric("Ownership", f"{params['current_ownership_pct']:.2f}%")
st.sidebar.metric("Probability of $1T", f"{params['monte_carlo_probability']:.1f}%")

# ============================================================================
# PAGE 1: EXECUTIVE SUMMARY
# ============================================================================
if page == "📊 Executive Summary":
    st.title("💰 When Will Elon Musk Become a Trillionaire?")
    st.markdown("### Comprehensive Analysis & Projection Model")
    
    st.markdown("---")
    
    # Key findings
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Current Net Worth",
            value=f"${params['current_networth']:.1f}B",
            delta=f"{(params['current_networth']/1000)*100:.1f}% to $1T"
        )
    
    with col2:
        base_final = scenarios['base']['Total_Wealth_B'].iloc[-1]
        st.metric(
            label="Projected (2030)",
            value=f"${base_final:.1f}B",
            delta=f"+{base_final - params['current_networth']:.1f}B"
        )
    
    with col3:
        st.metric(
            label="Monte Carlo Probability",
            value=f"{params['monte_carlo_probability']:.1f}%",
            delta="High Confidence"
        )
    
    with col4:
        base_trillion = scenarios['base'][scenarios['base']['Total_Wealth_B'] >= 1000]
        if len(base_trillion) > 0:
            trillion_date = base_trillion.index[0]
            st.metric(
                label="Expected Date",
                value=trillion_date.strftime('%b %Y'),
                delta="Base Scenario"
            )
        else:
            st.metric(
                label="Expected Date",
                value="Beyond 2030",
                delta="Base Scenario"
            )
    
    st.markdown("---")
    
    # Executive Summary
    st.markdown("## 📋 Executive Summary")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown(f"""
        ### Key Findings
        
        This analysis projects Elon Musk's wealth trajectory through November 2030 using:
        - **Ensemble forecasting models** (ARIMA + Prophet)
        - **Bull/Base/Bear scenario analysis**
        - **Monte Carlo simulation** (10,000 iterations)
        
        #### Methodology
        1. **Historical Data**: 186 months (Jun 2010 - Nov 2025)
        2. **TSLA Stock Forecast**: 5-year ensemble projection
        3. **Other Wealth**: SpaceX, xAI, Twitter/X assets
        4. **Scenarios**: Conservative, Likely, Optimistic cases
        
        #### Key Assumptions
        - TSLA ownership: ~{params['current_ownership_pct']:.1f}%
        - Other wealth growth: 10-35% annually
        - No major stock sales or dilution
        """)
    
    with col2:
        st.markdown("### 🎯 Probability Analysis")
        
        prob = params['monte_carlo_probability']
        
        if prob >= 75:
            color = "green"
            assessment = "HIGH"
        elif prob >= 50:
            color = "orange"
            assessment = "MODERATE"
        else:
            color = "red"
            assessment = "LOW"
        
        st.markdown(f"""
        <div class="metric-card">
            <h2 style="color: {color};">{prob:.1f}%</h2>
            <h4>{assessment} LIKELIHOOD</h4>
            <p>Based on 10,000 Monte Carlo simulations</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 📊 Scenario Outcomes")
        for scenario, label in [('bear', '🐻 Bear'), ('base', '📊 Base'), ('bull', '🐂 Bull')]:
            final_wealth = scenarios[scenario]['Total_Wealth_B'].iloc[-1]
            reaches = "✅" if final_wealth >= 1000 else "❌"
            st.markdown(f"**{label}**: ${final_wealth:.1f}B {reaches}")
    
    st.markdown("---")
    
    # Quick visualization
    st.markdown("## 📈 Wealth Projection Overview")
    
    fig = go.Figure()
    
    # Historical
    fig.add_trace(go.Scatter(
        x=merged_data.index,
        y=merged_data['Net_Worth_Billions'],
        name='Historical',
        line=dict(color='#666666', width=3)
    ))
    
    # Scenarios
    colors_scenario = {'bear': '#cc0000', 'base': '#0066cc', 'bull': '#00cc00'}
    for scenario in ['bear', 'base', 'bull']:
        fig.add_trace(go.Scatter(
            x=scenarios[scenario].index,
            y=scenarios[scenario]['Total_Wealth_B'],
            name=f'{scenario.title()} Scenario',
            line=dict(color=colors_scenario[scenario], width=2, dash='dash')
        ))
    
    fig.add_hline(y=1000, line_dash="dot", line_color="gold", 
                  annotation_text="$1 Trillion Target")
    
    fig.update_layout(
        title="Elon Musk Net Worth: Historical & Projected",
        xaxis_title="Date",
        yaxis_title="Net Worth ($ Billions)",
        hovermode='x unified',
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# PAGE 2: HISTORICAL ANALYSIS
# ============================================================================
elif page == "📈 Historical Analysis":
    st.title("📈 Historical Wealth Analysis")
    st.markdown("### June 2010 - November 2025")
    
    st.markdown("---")
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    start_wealth = merged_data['Net_Worth_Billions'].iloc[0]
    end_wealth = merged_data['Net_Worth_Billions'].iloc[-1]
    total_growth = ((end_wealth / start_wealth) - 1) * 100
    cagr = ((end_wealth / start_wealth) ** (1/15.5) - 1) * 100
    
    with col1:
        st.metric("Starting Wealth (2010)", f"${start_wealth:.1f}B")
    with col2:
        st.metric("Current Wealth (2025)", f"${end_wealth:.1f}B")
    with col3:
        st.metric("Total Growth", f"{total_growth:.0f}%")
    with col4:
        st.metric("CAGR", f"{cagr:.1f}%")
    
    st.markdown("---")
    
    # Time series charts
    tab1, tab2, tab3 = st.tabs(["💰 Total Wealth", "📊 TSLA Stock", "🏢 Wealth Components"])
    
    with tab1:
        st.markdown("### Total Net Worth Over Time")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=merged_data.index,
            y=merged_data['Net_Worth_Billions'],
            fill='tozeroy',
            name='Net Worth',
            line=dict(color='#0066cc', width=2)
        ))
        
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Net Worth ($ Billions)",
            hovermode='x',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        if st.checkbox("Show data table"):
            st.dataframe(merged_data[['Net_Worth_Billions', 'TSLA_Adj_Close', 
                                     'Ownership_Percentage']].tail(20))
    
    with tab2:
        st.markdown("### TSLA Stock Price History")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=merged_data.index,
            y=merged_data['TSLA_Adj_Close'],
            name='TSLA Price',
            line=dict(color='#E82127', width=2)
        ))
        
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="TSLA Price ($)",
            hovermode='x',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.markdown("### Wealth Components: TSLA vs Other Assets")
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=merged_data.index,
            y=merged_data['TSLA_Wealth_Billions'],
            stackgroup='one',
            name='TSLA Wealth',
            fillcolor='rgba(232, 33, 39, 0.7)'
        ))
        
        fig.add_trace(go.Scatter(
            x=merged_data.index,
            y=merged_data['Other_Wealth_Billions'],
            stackgroup='one',
            name='Other Wealth (SpaceX, xAI, etc.)',
            fillcolor='rgba(44, 160, 44, 0.7)'
        ))
        
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Wealth ($ Billions)",
            hovermode='x unified',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            tsla_wealth_current = merged_data['TSLA_Wealth_Billions'].iloc[-1]
            other_wealth_current = merged_data['Other_Wealth_Billions'].iloc[-1]
            
            fig_pie = go.Figure(data=[go.Pie(
                labels=['TSLA Wealth', 'Other Wealth'],
                values=[tsla_wealth_current, other_wealth_current],
                marker_colors=['#E82127', '#2ca02c']
            )])
            
            fig_pie.update_layout(title="Current Wealth Breakdown", height=300)
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            st.markdown("### Current Holdings")
            st.metric("TSLA Wealth", f"${tsla_wealth_current:.1f}B")
            st.metric("Other Wealth", f"${other_wealth_current:.1f}B")
            st.metric("TSLA % of Total", 
                     f"{(tsla_wealth_current/end_wealth)*100:.1f}%")

# ============================================================================
# PAGE 3: FORECAST & SCENARIOS
# ============================================================================
elif page == "🔮 Forecast & Scenarios":
    st.title("🔮 5-Year Forecast & Scenario Analysis")
    st.markdown("### December 2025 - November 2030")
    
    st.markdown("---")
    
    st.markdown("## 📊 Select Scenario")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        show_bear = st.checkbox("🐻 Bear Case", value=True)
    with col2:
        show_base = st.checkbox("📊 Base Case", value=True)
    with col3:
        show_bull = st.checkbox("🐂 Bull Case", value=True)
    
    st.markdown("---")
    
    with st.expander("ℹ️ Scenario Assumptions"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            ### 🐻 Bear Case
            **Conservative/Pessimistic**
            - TSLA: 95% lower confidence bound
            - Other Wealth: 10% annual growth
            - Market challenges, slower growth
            """)
        
        with col2:
            st.markdown("""
            ### 📊 Base Case
            **Most Likely**
            - TSLA: Ensemble forecast
            - Other Wealth: 20% annual growth
            - Current trends continue
            """)
        
        with col3:
            st.markdown("""
            ### 🐂 Bull Case
            **Optimistic**
            - TSLA: 95% upper confidence bound
            - Other Wealth: 35% annual growth
            - FSD success, SpaceX IPO, AI boom
            """)
    
    st.markdown("## 📈 Wealth Projection: All Scenarios")
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=merged_data.index,
        y=merged_data['Net_Worth_Billions'],
        name='Historical',
        line=dict(color='#000000', width=3)
    ))
    
    colors = {'bear': '#cc0000', 'base': '#0066cc', 'bull': '#00cc00'}
    show_flags = {'bear': show_bear, 'base': show_base, 'bull': show_bull}
    
    for scenario in ['bear', 'base', 'bull']:
        if show_flags[scenario]:
            fig.add_trace(go.Scatter(
                x=scenarios[scenario].index,
                y=scenarios[scenario]['Total_Wealth_B'],
                name=f'{scenario.title()} Scenario',
                line=dict(color=colors[scenario], width=3, dash='dash')
            ))
    
    fig.add_hline(y=1000, line_dash="dot", line_color="gold", 
                  annotation_text="$1 Trillion Target", line_width=3)
    
    fig.add_vline(
        x=merged_data.index[-1].timestamp() * 1000,  # Convert to milliseconds
        line_dash="dot", 
        line_color="black",
        annotation_text="Forecast Start"
    )
    
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Net Worth ($ Billions)",
        hovermode='x unified',
        height=500,
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    st.markdown("## 📊 Scenario Comparison (November 2030)")
    
    comparison_data = []
    for scenario in ['bear', 'base', 'bull']:
        df = scenarios[scenario]
        final_wealth = df['Total_Wealth_B'].iloc[-1]
        final_tsla = df['TSLA_Price'].iloc[-1]
        final_tsla_wealth = df['TSLA_Wealth_B'].iloc[-1]
        final_other = df['Other_Wealth_B'].iloc[-1]
        reaches_trillion = "✅" if final_wealth >= 1000 else "❌"
        
        comparison_data.append({
            'Scenario': scenario.title(),
            'Total Wealth': f'${final_wealth:.1f}B',
            'TSLA Price': f'${final_tsla:.2f}',
            'TSLA Wealth': f'${final_tsla_wealth:.1f}B',
            'Other Wealth': f'${final_other:.1f}B',
            'Reaches $1T': reaches_trillion
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📈 TSLA Price Projections")
        
        fig_tsla = go.Figure()
        
        fig_tsla.add_trace(go.Scatter(
            x=merged_data.index,
            y=merged_data['TSLA_Adj_Close'],
            name='Historical',
            line=dict(color='#666666', width=2)
        ))
        
        for scenario in ['bear', 'base', 'bull']:
            if show_flags[scenario]:
                fig_tsla.add_trace(go.Scatter(
                    x=scenarios[scenario].index,
                    y=scenarios[scenario]['TSLA_Price'],
                    name=scenario.title(),
                    line=dict(color=colors[scenario], width=2, dash='dash')
                ))
        
        fig_tsla.update_layout(
            xaxis_title="Date",
            yaxis_title="TSLA Price ($)",
            height=350
        )
        
        st.plotly_chart(fig_tsla, use_container_width=True)
    
    with col2:
        st.markdown("### 🏢 Wealth Components (Base Case)")
        
        base_df = scenarios['base']
        
        fig_components = go.Figure()
        
        fig_components.add_trace(go.Scatter(
            x=base_df.index,
            y=base_df['TSLA_Wealth_B'],
            stackgroup='one',
            name='TSLA Wealth',
            fillcolor='rgba(232, 33, 39, 0.7)'
        ))
        
        fig_components.add_trace(go.Scatter(
            x=base_df.index,
            y=base_df['Other_Wealth_B'],
            stackgroup='one',
            name='Other Wealth',
            fillcolor='rgba(44, 160, 44, 0.7)'
        ))
        
        fig_components.update_layout(
            xaxis_title="Date",
            yaxis_title="Wealth ($ Billions)",
            height=350
        )
        
        st.plotly_chart(fig_components, use_container_width=True)

# ============================================================================
# PAGE 4: TRILLIONAIRE TIMELINE
# ============================================================================
elif page == "🎯 Trillionaire Timeline":
    st.title("🎯 Path to $1 Trillion")
    st.markdown("### When Will Elon Musk Reach Trillionaire Status?")
    
    st.markdown("---")
    
    # Check which scenarios reach $1T
    timeline_data = []
    
    for scenario in ['bear', 'base', 'bull']:
        df = scenarios[scenario]
        trillion_reached = df[df['Total_Wealth_B'] >= 1000]
        
        if len(trillion_reached) > 0:
            date_reached = trillion_reached.index[0]
            months_from_now = (date_reached.year - 2025) * 12 + (date_reached.month - 12)
            tsla_price_at_trillion = trillion_reached['TSLA_Price'].iloc[0]
            
            timeline_data.append({
                'Scenario': scenario.title(),
                'Date': date_reached.strftime('%B %Y'),
                'Months Away': months_from_now,
                'TSLA Price': f'${tsla_price_at_trillion:.2f}',
                'Status': '✅ Reached'
            })
        else:
            final_wealth = df['Total_Wealth_B'].iloc[-1]
            timeline_data.append({
                'Scenario': scenario.title(),
                'Date': 'Beyond 2030',
                'Months Away': '>60',
                'TSLA Price': 'N/A',
                'Status': f'❌ ${final_wealth:.1f}B by 2030'
            })
    
    timeline_df = pd.DataFrame(timeline_data)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("## 📅 Timeline by Scenario")
        st.dataframe(timeline_df, use_container_width=True, hide_index=True)
    
    with col2:
        st.markdown("## 🎲 Monte Carlo Estimate")
        st.markdown(f"""
        <div class="metric-card">
            <h3>Probability of reaching $1T by 2030:</h3>
            <h1 style="color: #0066cc;">{params['monte_carlo_probability']:.1f}%</h1>
            <p>Based on 10,000 simulations</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Visualization: Race to $1T
    st.markdown("## 🏁 Race to $1 Trillion")
    
    fig = go.Figure()
    
    colors = {'bear': '#cc0000', 'base': '#0066cc', 'bull': '#00cc00'}
    
    for scenario in ['bear', 'base', 'bull']:
        df = scenarios[scenario]
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['Total_Wealth_B'],
            name=f'{scenario.title()} Scenario',
            line=dict(color=colors[scenario], width=3)
        ))
    
    fig.add_hline(y=1000, line_dash="solid", line_color="gold", 
                  line_width=4, annotation_text="$1 TRILLION TARGET",
                  annotation_position="right")
    
    fig.add_hrect(y0=900, y1=1100, fillcolor="gold", opacity=0.1,
                  annotation_text="Trillionaire Zone", annotation_position="top left")
    
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Net Worth ($ Billions)",
        hovermode='x unified',
        height=500,
        yaxis_range=[0, max(scenarios['bull']['Total_Wealth_B'].max(), 1200)]
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Milestones
    st.markdown("## 🎯 Wealth Milestones")
    
    milestones = [500, 750, 1000, 1250, 1500]
    
    milestone_data = []
    
    for milestone in milestones:
        row = {'Milestone': f'${milestone}B'}
        
        for scenario in ['bear', 'base', 'bull']:
            df = scenarios[scenario]
            reached = df[df['Total_Wealth_B'] >= milestone]
            
            if len(reached) > 0:
                date = reached.index[0].strftime('%b %Y')
                row[scenario.title()] = date
            else:
                row[scenario.title()] = 'Not reached'
        
        milestone_data.append(row)
    
    milestone_df = pd.DataFrame(milestone_data)
    st.dataframe(milestone_df, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # Key drivers
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🚀 Key Growth Drivers")
        st.markdown("""
        1. **Tesla Stock Performance**
           - FSD (Full Self-Driving) monetization
           - Robotaxi launch and scaling
           - Energy storage growth
           - China market expansion
        
        2. **SpaceX Valuation**
           - Starship development
           - Starlink global rollout
           - Potential IPO of Starlink
        
        3. **xAI Growth**
           - Grok AI development
           - Enterprise AI solutions
           - Competition with OpenAI/Google
        
        4. **X (Twitter) Monetization**
           - Subscription revenue growth
           - AI integration
           - Video/content expansion
        """)
    
    with col2:
        st.markdown("### ⚠️ Key Risks")
        st.markdown("""
        1. **Market Risks**
           - Economic recession
           - Tech sector downturn
           - Rising interest rates
        
        2. **Company-Specific Risks**
           - Tesla competition intensifies
           - FSD/Robotaxi delays
           - SpaceX development setbacks
        
        3. **Regulatory Risks**
           - Antitrust concerns
           - SEC investigations
           - International regulations
        
        4. **Personal Risks**
           - Stock sales for cash needs
           - Margin calls
           - Distraction across ventures
        """)

# ============================================================================
# PAGE 5: MONTE CARLO RESULTS
# ============================================================================
elif page == "🎲 Monte Carlo Results":
    st.title("🎲 Monte Carlo Simulation Results")
    st.markdown("### 10,000 Simulations: Wealth Path Uncertainty")
    
    st.markdown("---")
    
    # Load enhanced Monte Carlo data
    try:
        all_paths, sample_paths, summary, time_dist = load_enhanced_monte_carlo()
        enhanced_available = True
    except:
        enhanced_available = False
        st.warning("⚠️ Enhanced Monte Carlo data not found. Showing basic results.")
    
    # Key metrics
    st.markdown("## 📊 Simulation Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    if enhanced_available:
        with col1:
            st.metric(
                "Probability of $1T",
                f"{summary['probability_reach_trillion']:.1f}%",
                f"{summary['simulations_reach_trillion']} paths"
            )
        
        with col2:
            st.metric(
                "Median Time to $1T",
                f"{summary['median_months_to_trillion']:.0f} months",
                f"~{summary['median_months_to_trillion']/12:.1f} years"
            )
        
        with col3:
            st.metric(
                "Median Final Wealth",
                f"${summary['median_final_wealth']:.1f}B",
                f"by Nov 2030"
            )
        
        with col4:
            st.metric(
                "95th Percentile",
                f"${summary['percentile_95_final']:.1f}B",
                "Best case"
            )
    else:
        # FIX 2: Safely access DataFrame values
        try:
            final_median = float(monte_carlo['P50'].iloc[-1])
            final_95 = float(monte_carlo['P95'].iloc[-1])
            final_5 = float(monte_carlo['P5'].iloc[-1])
            prob = params['monte_carlo_probability']
            
            with col1:
                st.metric("Probability of $1T", f"{prob:.1f}%")
            with col2:
                st.metric("Median Final Wealth", f"${final_median:.1f}B")
            with col3:
                st.metric("95th Percentile", f"${final_95:.1f}B")
            with col4:
                st.metric("5th Percentile", f"${final_5:.1f}B")
        except Exception as e:
            st.error(f"Error displaying metrics: {str(e)}")
    
    st.markdown("---")
    
    # Percentile bands
    st.markdown("## 📈 Monte Carlo Uncertainty Bands")
    
    fig = go.Figure()
    
    # Historical
    fig.add_trace(go.Scatter(
        x=merged_data.index,
        y=merged_data['Net_Worth_Billions'],
        name='Historical',
        line=dict(color='#000000', width=3)
    ))
    
    # 5th-95th percentile band
    fig.add_trace(go.Scatter(
        x=monte_carlo.index,
        y=monte_carlo['P95'],
        name='95th percentile',
        line=dict(color='rgba(0,100,80,0)'),
        showlegend=False
    ))
    
    fig.add_trace(go.Scatter(
        x=monte_carlo.index,
        y=monte_carlo['P5'],
        name='5th-95th percentile',
        fill='tonexty',
        fillcolor='rgba(0,100,80,0.2)',
        line=dict(color='rgba(0,100,80,0)'),
    ))
    
    # 25th-75th percentile band
    fig.add_trace(go.Scatter(
        x=monte_carlo.index,
        y=monte_carlo['P75'],
        name='75th percentile',
        line=dict(color='rgba(0,100,80,0)'),
        showlegend=False
    ))
    
    fig.add_trace(go.Scatter(
        x=monte_carlo.index,
        y=monte_carlo['P25'],
        name='25th-75th percentile',
        fill='tonexty',
        fillcolor='rgba(0,100,80,0.3)',
        line=dict(color='rgba(0,100,80,0)'),
    ))
    
    # Median
    fig.add_trace(go.Scatter(
        x=monte_carlo.index,
        y=monte_carlo['P50'],
        name='Median (50th percentile)',
        line=dict(color='#0066cc', width=3)
    ))
    
    fig.add_hline(y=1000, line_dash="dot", line_color="gold", 
                  annotation_text="$1 Trillion", line_width=2)
    
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Net Worth ($ Billions)",
        hovermode='x unified',
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    if enhanced_available:
        # Sample paths visualization
        st.markdown("## 🎯 Sample Simulation Paths")
        st.markdown("*Showing 100 random paths from 10,000 simulations*")
        
        fig_paths = go.Figure()
        
        # Historical
        fig_paths.add_trace(go.Scatter(
            x=merged_data.index,
            y=merged_data['Net_Worth_Billions'],
            name='Historical',
            line=dict(color='#000000', width=3)
        ))
        
        # Sample paths
        for i, path in enumerate(sample_paths):
            fig_paths.add_trace(go.Scatter(
                x=monte_carlo.index,
                y=path,
                line=dict(color='rgba(100,100,100,0.1)', width=1),
                showlegend=False,
                hoverinfo='skip'
            ))
        
        # Median on top
        fig_paths.add_trace(go.Scatter(
            x=monte_carlo.index,
            y=monte_carlo['P50'],
            name='Median Path',
            line=dict(color='#0066cc', width=3)
        ))
        
        fig_paths.add_hline(y=1000, line_dash="dot", line_color="gold", 
                           annotation_text="$1 Trillion", line_width=2)
        
        fig_paths.update_layout(
            xaxis_title="Date",
            yaxis_title="Net Worth ($ Billions)",
            height=500
        )
        
        st.plotly_chart(fig_paths, use_container_width=True)
        
        st.markdown("---")
        
        # Time to trillion distribution
        st.markdown("## ⏱️ Time to Reach $1 Trillion")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig_hist = go.Figure()
            
            fig_hist.add_trace(go.Histogram(
                x=time_dist['months_to_trillion'],
                nbinsx=30,
                name='Distribution',
                marker_color='#0066cc'
            ))
            
            fig_hist.add_vline(x=summary['median_months_to_trillion'], 
                             line_dash="dash", line_color="red",
                             annotation_text=f"Median: {summary['median_months_to_trillion']:.0f} months")
            
            fig_hist.update_layout(
                xaxis_title="Months from Now",
                yaxis_title="Number of Simulations",
                title="Distribution of Time to Reach $1 Trillion",
                height=400
            )
            
            st.plotly_chart(fig_hist, use_container_width=True)
        
        with col2:
            st.markdown("### 📊 Statistics")
            st.metric("Mean", f"{summary['mean_months_to_trillion']:.1f} months")
            st.metric("Median", f"{summary['median_months_to_trillion']:.1f} months")
            st.metric("Std Dev", f"{summary['std_months_to_trillion']:.1f} months")
            st.metric("Min", f"{summary['min_months_to_trillion']:.0f} months")
            st.metric("Max", f"{summary['max_months_to_trillion']:.0f} months")
            
            st.markdown("---")
            
            earliest_date = (pd.Timestamp('2025-12-01') + 
                           pd.DateOffset(months=int(summary['min_months_to_trillion'])))
            median_date = (pd.Timestamp('2025-12-01') + 
                          pd.DateOffset(months=int(summary['median_months_to_trillion'])))
            
            st.markdown("### 📅 Dates")
            st.info(f"**Earliest**: {earliest_date.strftime('%B %Y')}")
            st.success(f"**Median**: {median_date.strftime('%B %Y')}")
    
    st.markdown("---")
    
    # Distribution at key dates
    st.markdown("## 📊 Wealth Distribution at Key Dates")
    
    key_dates = ['2026-12-01', '2028-01-01', '2030-01-01']
    
    for date_str in key_dates:
        if date_str in monte_carlo.index.astype(str):
            date_idx = monte_carlo.index.get_loc(date_str)
            date = monte_carlo.index[date_idx]
            
            st.markdown(f"### {date.strftime('%B %Y')}")
            
            col1, col2, col3, col4, col5 = st.columns(5)
            
            cols = [col1, col2, col3, col4, col5]
            percentiles = ['P5', 'P25', 'P50', 'P75', 'P95']
            labels = ['5th', '25th', '50th', '75th', '95th']
            
            for col, pct, label in zip(cols, percentiles, labels):
                value = monte_carlo.loc[date, pct]
                col.metric(label, f"${value:.1f}B")
            
            st.markdown("---")
    
    # Final wealth distribution
    st.markdown("## 💰 Final Wealth Distribution (November 2030)")
    
    if enhanced_available:
        final_wealths = all_paths[:, -1]
        
        fig_final = go.Figure()
        
        fig_final.add_trace(go.Histogram(
            x=final_wealths,
            nbinsx=50,
            name='Final Wealth Distribution',
            marker_color='#0066cc'
        ))
        
        fig_final.add_vline(x=1000, line_dash="dash", line_color="gold",
                           annotation_text="$1 Trillion")
        
        fig_final.add_vline(x=summary['median_final_wealth'], line_dash="dash", 
                           line_color="red",
                           annotation_text=f"Median: ${summary['median_final_wealth']:.1f}B")
        
        fig_final.update_layout(
            xaxis_title="Final Net Worth ($ Billions)",
            yaxis_title="Number of Simulations",
            height=400
        )
        
        st.plotly_chart(fig_final, use_container_width=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            below_trillion = (final_wealths < 1000).sum()
            st.metric("Below $1T", f"{below_trillion:,}", 
                     f"{below_trillion/len(final_wealths)*100:.1f}%")
        
        with col2:
            at_trillion = ((final_wealths >= 1000) & (final_wealths < 2000)).sum()
            st.metric("$1T - $2T", f"{at_trillion:,}",
                     f"{at_trillion/len(final_wealths)*100:.1f}%")
        
        with col3:
            above_2trillion = (final_wealths >= 2000).sum()
            st.metric("Above $2T", f"{above_2trillion:,}",
                     f"{above_2trillion/len(final_wealths)*100:.1f}%")

# ============================================================================
# PAGE 6: INTERACTIVE CALCULATOR
# ============================================================================
elif page == "⚙️ Interactive Calculator":
    st.title("⚙️ Interactive Wealth Calculator")
    st.markdown("### Build Your Own Scenario")
    
    st.markdown("---")
    
    st.markdown("""
    Adjust the parameters below to create your own custom wealth projection scenario.
    This calculator lets you experiment with different assumptions about Tesla's stock price,
    ownership changes, and other wealth growth.
    """)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🚗 Tesla Parameters")
        
        custom_tsla_start = st.number_input(
            "Starting TSLA Price ($)",
            min_value=100.0,
            max_value=1000.0,
            value=float(params['current_tsla_price']),
            step=10.0,
            help="Current Tesla stock price"
        )
        
        custom_tsla_growth = st.slider(
            "Annual TSLA Growth Rate (%)",
            min_value=-20.0,
            max_value=50.0,
            value=15.0,
            step=1.0,
            help="Expected annual growth rate for Tesla stock"
        )
        
        custom_ownership = st.slider(
            "Ownership Percentage (%)",
            min_value=10.0,
            max_value=25.0,
            value=float(params['current_ownership_pct']),
            step=0.1,
            help="Elon's ownership stake in Tesla"
        )
        
        custom_volatility = st.slider(
            "Price Volatility (σ)",
            min_value=0.2,
            max_value=0.8,
            value=0.4,
            step=0.05,
            help="Higher = more price swings"
        )
    
    with col2:
        st.markdown("### 🏢 Other Wealth Parameters")
        
        custom_other_start = st.number_input(
            "Starting Other Wealth ($B)",
            min_value=50.0,
            max_value=200.0,
            value=float(merged_data['Other_Wealth_Billions'].iloc[-1]),
            step=5.0,
            help="SpaceX, xAI, X (Twitter), etc."
        )
        
        custom_other_growth = st.slider(
            "Annual Other Wealth Growth (%)",
            min_value=0.0,
            max_value=50.0,
            value=20.0,
            step=1.0,
            help="Growth rate for non-Tesla assets"
        )
        
        st.markdown("---")
        
        projection_years = st.slider(
            "Projection Period (years)",
            min_value=1,
            max_value=10,
            value=5,
            help="How far into the future to project"
        )
    
    st.markdown("---")
    
    if st.button("🚀 Calculate Projection", type="primary"):
        # Generate custom projection
        months = projection_years * 12
        dates = pd.date_range(start='2025-12-01', periods=months, freq='MS')
        
        # TSLA projection with drift and volatility
        np.random.seed(42)
        dt = 1/12  # Monthly time step
        
        tsla_prices = [custom_tsla_start]
        for i in range(months - 1):
            drift = (custom_tsla_growth / 100) * dt
            shock = custom_volatility * np.sqrt(dt) * np.random.randn()
            new_price = tsla_prices[-1] * np.exp(drift + shock)
            tsla_prices.append(new_price)
        
        # Calculate wealth
        shares = (custom_ownership / 100) * 3.2e9  # Approximate total shares
        tsla_wealth = np.array(tsla_prices) * shares / 1e9
        
        # Other wealth grows at constant rate
        other_wealth = [custom_other_start * (1 + custom_other_growth/100) ** (i/12) 
                       for i in range(months)]
        
        total_wealth = tsla_wealth + np.array(other_wealth)
        
        # Create DataFrame
        custom_df = pd.DataFrame({
            'Date': dates,
            'TSLA_Price': tsla_prices,
            'TSLA_Wealth': tsla_wealth,
            'Other_Wealth': other_wealth,
            'Total_Wealth': total_wealth
        })
        custom_df.set_index('Date', inplace=True)
        
        st.success("✅ Projection calculated!")
        
        # Results
        st.markdown("## 📊 Your Custom Projection")
        
        col1, col2, col3, col4 = st.columns(4)
        
        final_wealth = custom_df['Total_Wealth'].iloc[-1]
        final_tsla = custom_df['TSLA_Price'].iloc[-1]
        
        with col1:
            st.metric("Final Net Worth", f"${final_wealth:.1f}B")
        
        with col2:
            st.metric("Final TSLA Price", f"${final_tsla:.2f}")
        
        with col3:
            growth = ((final_wealth / (custom_tsla_start * shares / 1e9 + custom_other_start)) - 1) * 100
            st.metric("Total Growth", f"{growth:.1f}%")
        
        with col4:
            reaches = "✅ YES" if final_wealth >= 1000 else "❌ NO"
            st.metric("Reaches $1T?", reaches)
        
        st.markdown("---")
        
        # Chart
        fig = go.Figure()
        
        # Historical
        fig.add_trace(go.Scatter(
            x=merged_data.index,
            y=merged_data['Net_Worth_Billions'],
            name='Historical',
            line=dict(color='#666666', width=2)
        ))
        
        # Custom projection
        fig.add_trace(go.Scatter(
            x=custom_df.index,
            y=custom_df['Total_Wealth'],
            name='Your Projection',
            line=dict(color='#ff6600', width=3, dash='dash')
        ))
        
        # Add trillion line
        fig.add_hline(y=1000, line_dash="dot", line_color="gold",
                     annotation_text="$1 Trillion", line_width=2)
        
        # Check if/when trillion is reached
        trillion_reached = custom_df[custom_df['Total_Wealth'] >= 1000]
        if len(trillion_reached) > 0:
            trillion_date = trillion_reached.index[0]
            fig.add_vline(x=trillion_date, line_dash="dot", line_color="gold",
                         annotation_text=f"$1T: {trillion_date.strftime('%b %Y')}")
        
        fig.update_layout(
            title="Custom Wealth Projection",
            xaxis_title="Date",
            yaxis_title="Net Worth ($ Billions)",
            hovermode='x unified',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Detailed breakdown
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📈 TSLA Price Path")
            
            fig_tsla = go.Figure()
            
            fig_tsla.add_trace(go.Scatter(
                x=custom_df.index,
                y=custom_df['TSLA_Price'],
                fill='tozeroy',
                line=dict(color='#E82127', width=2)
            ))
            
            fig_tsla.update_layout(
                xaxis_title="Date",
                yaxis_title="TSLA Price ($)",
                height=300
            )
            
            st.plotly_chart(fig_tsla, use_container_width=True)
        
        with col2:
            st.markdown("### 🏢 Wealth Components")
            
            fig_components = go.Figure()
            
            fig_components.add_trace(go.Bar(
                x=['Start', 'End'],
                y=[custom_tsla_start * shares / 1e9, custom_df['TSLA_Wealth'].iloc[-1]],
                name='TSLA Wealth',
                marker_color='#E82127'
            ))
            
            fig_components.add_trace(go.Bar(
                x=['Start', 'End'],
                y=[custom_other_start, custom_df['Other_Wealth'].iloc[-1]],
                name='Other Wealth',
                marker_color='#2ca02c'
            ))
            
            fig_components.update_layout(
                xaxis_title="",
                yaxis_title="Wealth ($ Billions)",
                barmode='stack',
                height=300
            )
            
            st.plotly_chart(fig_components, use_container_width=True)
        
        # Data table
        if st.checkbox("Show projection data table"):
            st.dataframe(custom_df.round(2), use_container_width=True)
        
        # Download button
        csv = custom_df.to_csv()
        st.download_button(
            label="📥 Download Projection CSV",
            data=csv,
            file_name="custom_wealth_projection.csv",
            mime="text/csv"
        )
    
    st.markdown("---")
    
    # Comparison with scenarios
    st.markdown("## 🔄 Compare with Standard Scenarios")
    
    if st.checkbox("Show comparison"):
        st.markdown("""
        Compare your custom parameters with the standard Bear/Base/Bull scenarios
        to see how your assumptions stack up.
        """)
        
        comparison_data = {
            'Parameter': [
                'TSLA Growth Rate',
                'Other Wealth Growth',
                'Volatility'
            ],
            'Your Scenario': [
                f'{custom_tsla_growth:.1f}%',
                f'{custom_other_growth:.1f}%',
                f'{custom_volatility:.2f}'
            ],
            'Bear': ['~5%', '10%', '0.40'],
            'Base': ['~15%', '20%', '0.40'],
            'Bull': ['~25%', '35%', '0.40']
        }
        
        comp_df = pd.DataFrame(comparison_data)
        st.dataframe(comp_df, use_container_width=True, hide_index=True)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666666;'>
        <p>💡 <strong>Disclaimer</strong>: This is a projection model based on historical data and assumptions. 
        Actual results may vary significantly. Not financial advice.</p>
        <p>Data sources: Yahoo Finance, Forbes, Bloomberg | Last updated: November 2025</p>
    </div>
    """, unsafe_allow_html=True) 
