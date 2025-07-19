import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objs as go
from streamlit_plotly_events import plotly_events
from scipy.optimize import minimize

# Dummy list of assets
ASSETS = ["Well A", "Well B", "Well C", "Well D"]


def init_app():
    if 'in_forecast_yrs' in st.session_state:
        st.session_state['forecast_yrs'] = st.session_state['in_forecast_yrs']
        st.session_state.__delitem__('in_forecast_yrs')
    if 'forecast_yrs' not in st.session_state:
        st.session_state['forecast_yrs'] = 10.
    if 'b_slider' in st.session_state:
        st.session_state['b'] = st.session_state['b_slider']
        st.session_state.__delitem__('b_slider')
    if 'b' not in st.session_state:
        st.session_state['b'] = 1.
    if 'd_pct_slider' in st.session_state:
        st.session_state['d_pct'] = st.session_state['d_pct_slider']
        st.session_state.__delitem__('d_pct_slider')
    if 'd_pct' not in st.session_state:
        st.session_state['d_pct'] = 15.

@st.cache_data
def get_asset_timeseries(asset_name, forecast_yrs):
    """
    Dummy function to simulate retrieving timeseries data for a given asset.
    Returns a DataFrame with 'date' and 'value' columns.
    The data loosely follows an exponential decline with added noise.
    """
    np.random.seed(hash(asset_name) % 2**32 + 1)  # Seed for reproducibility per asset
    dates = pd.date_range(datetime.today() - timedelta(days=10*365.25), periods=int(121 + forecast_yrs * 12), freq='ME')
    # Randomize exponential decline parameters
    q0 = np.random.uniform(100., 1200.)
    d_annual = np.random.uniform(0.5, 0.35)  # 15% to 35% annual decline
    b = np.random.uniform(0.0001, 0.95)  # hyperbolic exponent
    d_monthly = d_annual / 12.
    months = np.arange(121)
    # base = q0 * np.exp(-d_monthly * months)
    base = q0 / np.power(1. + b * d_monthly * months, 1. / b)
    ar_coef = 0.8  # Autoregressive coefficient (0 < ar_coef < 1)
    noise_std = base * 0.15
    noise = np.zeros(121)
    noise[0] = np.random.normal(0, noise_std[0])
    for i in range(1, 121):
        noise[i] = ar_coef * noise[i-1] + np.random.normal(0, noise_std[i])
    values = base + noise
    df = pd.DataFrame({"date": dates})
    df["value"] = None
    df.iloc[:len(values), 1] = values
    df['date'] = df['date'].dt.date
    st.session_state['b'] = b
    st.session_state['d_pct'] = d_annual * 100.
    return df, b, d_annual * 100.

def calculate_forward_economics():
    x = pd.date_range(max(asset_df['date']), periods=int(st.session_state['forecast_yrs'] * 12 + 1), freq='ME').date.tolist()[1:]
    oil = generate_model_curve(x, hist=False)
    df = pd.DataFrame({"date": x, "oil": oil})
    # df['gas'] = 0.
    # df['water'] = 0.
    df['revenue'] = price_oil * df['oil'] #+ price_gas * df['gas']
    df['gr_exp'] = var_cost_oil * df['oil'] + fixed_cost #+ var_cost_gas * df['gas'] + var_cost_water * df['water']
    df['net_exp'] = df['gr_exp'] * wi
    df['net_rev'] = df['revenue'] * nri * (1 - sev_tax) - df['net_exp']
    df['net_rev'] = df['net_rev'] * (1 - ad_val)
    df['disc_fact'] = [((1 + disc_rate / 12) ** m) for m in list(range(len(df)))]
    df['net_disc_rev'] = df['net_rev'] * df['disc_fact']
    return max(df['net_disc_rev'].cumsum())

def generate_model_curve(x_dates, hist=True):
    """
    Generate y-values for a model line based on Arps decline using session state parameters.
    Uses q0 as the first value in the data, b and d_pct from session state.
    """
    if hist:
        q0 = 1000
        b = st.session_state.get('b', 1.0)
        d_pct = st.session_state.get('d_pct', 100)
        days = (pd.to_datetime(pd.Series(x_dates)) - pd.to_datetime(pd.Series(x_dates)[0])).dt.days
        d = d_pct / 100  # percent to fraction per year
        d_daily = d / 365.25
        if b == 0:
            y = q0 * np.exp(-d_daily * days)
        else:
            if np.abs(b - 1) < 1e-5:
                b -= 1e-5
            y = q0 / np.power(1 + b * d_daily * days, 1 / b)
        y = y / y.iloc[st.session_state.selected_idx] * asset_df['value'][st.session_state.selected_idx]
        st.session_state['q0'] = q0 / y.iloc[st.session_state.selected_idx] * asset_df['value'][st.session_state.selected_idx]
        st.session_state['tie_date'] = asset_df['date'][st.session_state.selected_idx]
    else:
        q0 = st.session_state.get('q0')
        b = st.session_state.get('b', 1.0)
        d_pct = st.session_state.get('d_pct', 100)
        days = (pd.to_datetime(pd.Series(x_dates)) - pd.to_datetime(st.session_state['tie_date'])).dt.days
        d = d_pct / 100  # percent to fraction per year
        d_daily = d / 365.25
        if b == 0:
            y = q0 * np.exp(-d_daily * days)
        else:
            y = q0 / np.power(1 + b * d_daily * days, 1 / b)
    return y

st.set_page_config(page_title="Asset Line Chart", layout="wide")
st.title("Economics Calculator")

init_app()

# Dropdown menu for asset selection
selected_asset = st.selectbox("Select Asset", ASSETS)

# Retrieve data for the selected asset
asset_df, asset_b, asset_d_pct = get_asset_timeseries(selected_asset, forecast_yrs=st.session_state['forecast_yrs'])

# If the asset has changed, update the session state for b and d_pct
if 'last_asset' not in st.session_state or st.session_state['last_asset'] != selected_asset:
    st.session_state['b'] = asset_b
    st.session_state['d_pct'] = asset_d_pct
    st.session_state['last_asset'] = selected_asset

# Add sliders for parameters above the chart
st.markdown('### Parameters')
col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    st.slider(
        label="b",
        min_value=0.0,
        max_value=2.0,
        value=st.session_state['b'],
        step=0.01,
        key='b_slider',
        help="Hyperbolic exponent. Controls the curve shape."
    )
with col2:
    st.slider(
        label="Di (%/yr)",
        min_value=0.,
        max_value=200.,
        value=st.session_state['d_pct'],
        step=0.5,
        key='d_pct_slider',
        help="Nominal decline rate. Controls how steep the curve is."
    )
with col3:
    autofit = st.toggle('Autofit Curve', value=False, key='autofit_toggle', help="Automatically optimize b and Di parameters, constrained by selected tie-out point.")

# Autofit functionality
if autofit:
    st.markdown(
        """
        <style>
        /* Change the color of the slider handle (thumb) */
        .stSlider [role="slider"] {
            background-color: #888888 !important;  /* Grey handle */
            border: 2px solid #888888 !important;
        }
        /* Change the color of the active track (filled part) */
        .stSlider [data-baseweb="slider"] > div > div:first-child:not([data-testid="stSliderTickBarMin"]) {
            background: #888888 !important;
        }
        /* Change the color of the value label above the slider */
        [data-testid="stSliderThumbValue"] {
            color: #888888 !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    def arps_curve(x_dates, b, d_pct):
        q0 = 1000
        days = (pd.to_datetime(pd.Series(x_dates)) - pd.to_datetime(pd.Series(x_dates)[0])).dt.days
        d = d_pct / 100
        d_daily = d / 365.25
        if b == 0:
            y = q0 * np.exp(-d_daily * days)
        else:
            if np.abs(b - 1) < 1e-5:
                b -= 1e-5
            y = q0 / np.power(1 + b * d_daily * days, 1 / b)
        # Scale to tie point
        y = y / y.iloc[st.session_state.selected_idx] * asset_df['value'][st.session_state.selected_idx]
        return y
    def loss(params):
        b_fit, d_pct_fit = params
        y = asset_df['value'].values
        y_pred = arps_curve(asset_df['date'], b=b_fit, d_pct=d_pct_fit)
        return np.sum((y - y_pred) ** 2)
    res = minimize(loss, [st.session_state['b'], st.session_state['d_pct']], bounds=[(0, 2), (0, 500)])
    b = res.x[0]
    d_pct = res.x[1]
    if (np.abs(b - st.session_state['b']) > 0.001) or (np.abs(d_pct - st.session_state['d_pct']) > 0.001):
        st.session_state['b'] = b
        st.session_state['d_pct'] = d_pct
        st.session_state.__delitem__('b_slider')
        st.session_state.__delitem__('d_pct_slider')
        # st.rerun()


# Initialize or get the selected point index from session state
if 'selected_idx' not in st.session_state or st.session_state.get('last_asset') != selected_asset:
    st.session_state.selected_idx = 120  # Start at the end (last point)
    st.session_state.last_asset = selected_asset

# Generate model curve
model_y = generate_model_curve(asset_df['date'])

# --- Economic Parameters Sidebar ---
st.sidebar.header('Economic Parameters')
price_oil = st.sidebar.number_input('Oil Price ($/BBL)', min_value=0.0, value=70., step=2.5)
# price_gas = st.sidebar.number_input('$/MCF Gas', min_value=0.0, value=3.75, step=0.25)
var_cost_oil = st.sidebar.number_input('Var Exp ($/BBL Oil)', min_value=0.0, value=5.0, step=0.5)
# var_cost_gas = st.sidebar.number_input('$/MCF Gas', min_value=0.0, value=1.0, step=0.1)
# var_cost_water = st.sidebar.number_input('$/BBL Water', min_value=0.0, value=2.0, step=0.1)

# st.sidebar.subheader('Fixed Cost')
fixed_cost = st.sidebar.number_input('Fixed Exp ($/Well/Mo)', min_value=0.0, value=1000.0, step=100.0)

st.sidebar.subheader('Ownership')
wi = st.sidebar.number_input('Working Interest (%)', min_value=0.0, max_value=100.0, value=100., step=5.)/100
nri = st.sidebar.number_input('Net Rev Interest (%)', min_value=0.0, max_value=100.0, value=80., step=5.)/100

st.sidebar.subheader('Other')
disc_rate = st.sidebar.number_input(
    'Discount Rate (%/Yr)', min_value=0.0, max_value=100.0, value=10.0, step=5., format="%.1f")/100
ad_val = st.sidebar.number_input(
    'Ad Valorem (%)', min_value=0.0, max_value=100.0, value=10.0, step=0.5, format="%.1f")/100
sev_tax = st.sidebar.number_input(
    'Severance Tax (%)', min_value=0.0, max_value=100.0, value=5.5, step=0.5, format="%.1f")/100

# 1. Create the Plotly figure with the tie point at the current index
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=asset_df['date'],
    y=asset_df['value'],
    mode='lines+markers',
    name='Oil Rate (BOPD)',
    hovertemplate='<b>Date:</b> %{x}<br><b>Oil Rate:</b> %{y:.2f}<extra></extra>'
))
fig.add_trace(go.Scatter(
    x=[asset_df['date'][st.session_state.selected_idx]],
    y=[asset_df['value'][st.session_state.selected_idx]],
    mode='markers',
    marker=dict(size=12, color='red'),
    name='Tie Point',
))
fig.add_trace(go.Scatter(
    x=asset_df['date'],
    y=model_y,
    mode='lines',
    line=dict(color='green', dash='dash'),
    name='Model Curve',
))
fig.update_layout(
    margin=dict(l=80, r=40, t=60, b=80),
    showlegend=True,
    height=450,
    xaxis_title='Date',
    yaxis_title='Oil Rate (BOPD)',
    yaxis_type='log',
    dragmode=None,
    legend=dict(
        x=0,
        y=0,
        xanchor='left',
        yanchor='bottom',
        bgcolor='rgba(255,255,255,0.5)',  # semi-transparent background
        bordercolor='rgba(0,0,0,0.2)',    # subtle border
        borderwidth=1
    )
)

# 2. Pass the figure to plotly_events
selected_points = plotly_events(
    fig,
    click_event=True,
    select_event=False,
    hover_event=False,
    override_height=450
)

st.number_input(
    'Forecast Life (Yr)', min_value=0.0, value=st.session_state['forecast_yrs'], step=1., key='in_forecast_yrs')

# 3. If the user clicked, update the marker to the nearest data point
if selected_points:
    click_idx = selected_points[0]['pointIndex']
    st.session_state['selected_idx'] = click_idx if click_idx <= 120 else 120
    st.rerun()  # Force rerun to update the marker position

st.markdown(f'#### NPV{int(disc_rate * 100)} (MM$): {round(calculate_forward_economics()/1e6, 3)}')

with st.expander('Instructions', expanded=False):
    st.markdown('''
- Click any data point on the chart to select the Tie-Point.
- Economic parameters are available in sidebar (use >> to expand).
- Parameter adjustments will not work when autofit is active.
- NPV calculation includes a well-level economic limit.
''')
