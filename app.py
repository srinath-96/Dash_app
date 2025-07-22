# main.py
import pandas as pd
import plotly.express as px
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import numpy as np
import statsmodels.api as sm

# --- 1. Data Loading and Preprocessing ---

# Load the datasets for each city

city_a_df = pd.read_excel('dataset.xlsx',sheet_name=0)
city_a_df['city_id'] = 1
city_b_df = pd.read_excel('dataset.xlsx',sheet_name=1)
city_b_df['city_id'] = 2        
city_c_df = pd.read_excel('dataset.xlsx',sheet_name=2)
city_c_df['city_id'] = 3
city_d_df = pd.read_excel('dataset.xlsx',sheet_name=3)
city_d_df['city_id'] = 4
city_e_df = pd.read_excel('dataset.xlsx',sheet_name=4)
city_e_df['city_id'] = 5

city_e_df.rename(columns={'accommodation_type_name': 'accommadation_type_name'},inplace=True)
city_c_df.rename(columns={'accommodation_type_name': 'accommadation_type_name'},inplace=True)
city_d_df.rename(columns={'accommodation_type_name': 'accommadation_type_name'},inplace=True)
# Combine all city data into a single DataFrame

all_cities_df = pd.concat([city_a_df, city_b_df, city_c_df, city_d_df, city_e_df], ignore_index=True)
all_cities_df.drop('days before checkin',axis=1,inplace=True)
all_cities_df['chain_hotel'] = all_cities_df['chain_hotel'].map({'chain': 1, 'non-chain': 0})
# --- Feature Engineering ---
# Convert date columns to datetime objects
all_cities_df['booking_date'] = pd.to_datetime(all_cities_df['booking_date'])
all_cities_df['checkin_date'] = pd.to_datetime(all_cities_df['checkin_date'])

# Calculate 'days_before_checkin'
all_cities_df['days_before_checkin'] = (all_cities_df['checkin_date'] - all_cities_df['booking_date']).dt.days
all_cities_df['checkin_month'] = all_cities_df['checkin_date'].dt.month
all_cities_df['checkin_month_name'] = all_cities_df['checkin_date'].dt.strftime('%B')
all_cities_df.rename(columns={'accommadation_type_name': 'accommodation_type_name'},inplace=True)
# Data Cleaning: Remove rows where 'days_before_checkin' is negative or ADR_USD is non-positive
all_cities_df = all_cities_df[all_cities_df['days_before_checkin'] >= 0]
all_cities_df = all_cities_df[all_cities_df['ADR_USD'] > 0]

all_cities_df['accommodation_type_name'] = all_cities_df['accommodation_type_name'].fillna('Unknown')
all_cities_df['log_ADR_USD'] = np.log(all_cities_df['ADR_USD'])


# --- 2. Dash Application Layout ---

# Initialize the Dash app
app = dash.Dash(__name__)
server = app.server

# Define the layout of the dashboard
app.layout = html.Div(
    
    children=[
        # --- Header ---
        html.H1(
            "Agoda Urgency Message: Price Analysis Dashboard",
            style={'textAlign': 'center', 'color': '#fff'}
        ),
        

        # --- Filters/Controls ---
        html.Div(
            style={'display': 'flex', 'width': '100%', 'marginBottom': '20px', 'flexWrap': 'nowrap'},
            children=[
                html.Div(
                    className="controls-container",
                    style={
                        'backgroundColor': '#222',
                        'padding': '20px',
                        'borderRadius': '8px',
                        'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
                        'display': 'grid',
                        'gridTemplateColumns': 'repeat(auto-fit, minmax(10px, 1fr))',
                        'gap': '10px',
                        'marginBottom': '0',
                        'width': '70%'
                    },
                    children=[
                        # City Filter
                        html.Div([
                            html.Label("City", style={'fontWeight': 'bold', 'marginBottom': '5px', 'color': '#fff'}),
                            dcc.Dropdown(
                                id='city-filter',
                                options=[{'label': str(city), 'value': city} for city in sorted(all_cities_df['city_id'].unique())],
                                value=None, # Default to all cities
                                multi=True,
                                placeholder="Select",
                                style={'backgroundColor': '#e0e0e0', 'color': '#000', 'border': '1px solid #000'}
                            ),
                        ]),

                        # Star Rating Filter
                        html.Div([
                            html.Label("Star Rating", style={'fontWeight': 'bold', 'marginBottom': '5px', 'color': '#fff'}),
                            dcc.Dropdown(
                                id='star-rating-filter',
                                options=[{'label': f"{rating} Stars", 'value': rating} for rating in sorted(all_cities_df['star_rating'].unique())],
                                value=None,
                                multi=True,
                                placeholder="Select",
                                style={'backgroundColor': '#e0e0e0', 'color': '#000', 'border': '1px solid #000 '}
                            ),
                        ]),

                        # Accommodation Type Filter
                        html.Div([
                            html.Label("Accommodation Type", style={'fontWeight': 'bold', 'marginBottom': '5px', 'color': '#fff'}),
                            dcc.Dropdown(
                                id='accommodation-type-filter',
                                options=[
                                    {'label': acc_type, 'value': acc_type}
                                    for acc_type in sorted(
                                        all_cities_df['accommodation_type_name']
                                        .dropna()
                                        .astype(str)
                                        .unique()
                                    )
                                ],
                                value=None,
                                multi=True,
                                placeholder="Select",
                                style={'backgroundColor': '#e0e0e0', 'color': '#000', 'border': '1px solid #000'}
                            ),
                        ]),

                        # Chain Hotel Filter
                        html.Div([
                            html.Label("Chain Hotel", style={'fontWeight': 'bold', 'marginBottom': '5px', 'color': '#fff'}),
                            dcc.Dropdown(
                                id='chain-hotel-filter',
                                options=[
                                    {'label': 'Chain Hotels', 'value': 1},
                                    {'label': 'Independent Hotels', 'value': 0}
                                ],
                                value=None,
                                multi=True,
                                placeholder="Select",
                                style={'backgroundColor': '#e0e0e0', 'color': '#000', 'border': '1px solid #000'}
                            ),
                        ]),
                        # Add a DatePickerRange for check-in and check-out dates after the month filter
                        html.Div([
                            html.Label("Check-in/Check-out Date Range", style={'fontWeight': 'bold', 'marginBottom': '5px', 'color': '#fff'}),
                            dcc.DatePickerRange(
                                id='date-range-filter',
                                min_date_allowed=all_cities_df['checkin_date'].min(),
                                max_date_allowed=all_cities_df['checkout_date'].max(),
                                start_date=all_cities_df['checkin_date'].min(),
                                end_date=all_cities_df['checkout_date'].max(),
                                display_format='YYYY-MM-DD',
                                style={'backgroundColor': '#222', 'color': '#222', 'border': '1px solid #333', 'minWidth': '180px', 'width': '250px'}
                            ),
                        ]),
                        # Add the toggle as the last child in the controls-container grid
                        html.Div([
                            dcc.RadioItems(
                                id='adr-log-toggle',
                                options=[
                                    {'label': 'ADR', 'value': 'ADR_USD'},
                                    {'label': 'log(ADR)', 'value': 'log_ADR_USD'}
                                ],
                                value='ADR_USD',
                                labelStyle={'display': 'inline-block', 'marginRight': '10px', 'color': '#fff'},
                                inputStyle={'marginRight': '5px'}
                            )
                        ], style={'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center'}),
                    ]
                ),
                html.Div(
                    style={'width': '30%', 'backgroundColor': '#222', 'padding': '20px', 'borderRadius': '8px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)', 'marginLeft': '20px'},
                    children=[
                        html.Label("Days Before Check-in Range", style={'fontWeight': 'bold', 'marginBottom': '10px', 'display': 'block', 'color': '#fff'}),
                        dcc.RangeSlider(
                            id='days-before-checkin-slider',
                            min=all_cities_df['days_before_checkin'].min(),
                            max=all_cities_df['days_before_checkin'].max(),
                            value=[all_cities_df['days_before_checkin'].min(), all_cities_df['days_before_checkin'].max()],
                            marks={i: str(i) for i in range(0, all_cities_df['days_before_checkin'].max() + 1, 30)},
                            step=1,
                            tooltip={"placement": "bottom", "always_visible": False}
                        ),
                    ]
                )
            ]
        ),

        # --- Main Graph ---
        html.Div(
            style={'display': 'grid', 'gridTemplateColumns': '1.5fr 2.5fr', 'gap': '0px', 'marginBottom': '0px', 'backgroundColor': '#111'},
            children=[
                html.Div([
                    html.Div(id='kpi-trend', style={'backgroundColor': '#222', 'padding': '16px', 'borderRadius': '8px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.5)', 'color': '#fff', 'fontWeight': 'bold', 'fontSize': '1.3em', 'marginBottom': '12px', 'textAlign': 'center'}),
                    html.Div(id='kpi-base', style={'backgroundColor': '#222', 'padding': '16px', 'borderRadius': '8px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.5)', 'color': '#fff', 'fontWeight': 'bold', 'fontSize': '1.3em', 'marginBottom': '12px', 'textAlign': 'center'}),
                    html.Div(id='kpi-strength', style={'backgroundColor': '#222', 'padding': '16px', 'borderRadius': '8px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.5)', 'color': '#fff', 'fontWeight': 'bold', 'fontSize': '1.3em','marginBottom': '12px', 'textAlign': 'center'})
                ], style={'display': 'flex', 'flexDirection': 'column', 'justifyContent': 'center', 'height': '400px', 'width': '90%', 'minWidth': '0', 'maxWidth': '100%'}),
                dcc.Loading(
                    id="loading-icon",
                    type="circle",
                    children=dcc.Graph(id='price-trend-scatter-plot', style={'height': '400px', 'backgroundColor': '#111'})
                )
            ]
        ),

        # --- Grid of Bar Graphs ---
        html.Div(
            style={
                'display': 'grid',
                'gridTemplateColumns': 'repeat(3, 1fr)',
                'gap': '0px',
                'marginTop': '0px',
                'backgroundColor': '#111'
            },
            children=[
                html.Div([
                    dcc.Graph(id='adr-by-accommodation-bar', style={'height': '450px', 'backgroundColor': '#111'})
                ], style={'backgroundColor': '#222', 'padding': '4px', 'borderRadius': '8px'}),
                html.Div([
                    dcc.Graph(id='adr-by-chain-bar', style={'height': '450px', 'backgroundColor': '#111'})
                ], style={'backgroundColor': '#222', 'padding': '4px', 'borderRadius': '8px'}),
                html.Div([
                    dcc.Graph(id='adr-by-star-bar', style={'height': '450px', 'backgroundColor': '#111'})
                ], style={'backgroundColor': '#222', 'padding': '4px', 'borderRadius': '8px'})
            ]
        ),
    ],
    style={'fontFamily': 'Arial, sans-serif', 'backgroundColor': '#111', 'padding': '20px', 'minHeight': '100vh'}
)


# --- 3. Callback to Update Graph ---

@app.callback(
    [Output('price-trend-scatter-plot', 'figure'),
     Output('kpi-trend', 'children'),
     Output('kpi-base', 'children'),
     Output('kpi-strength', 'children')],
    [
        Input('city-filter', 'value'),
        Input('star-rating-filter', 'value'),
        Input('accommodation-type-filter', 'value'),
        Input('chain-hotel-filter', 'value'),
        Input('days-before-checkin-slider', 'value'),
        Input('date-range-filter', 'start_date'),
        Input('date-range-filter', 'end_date'),
        Input('adr-log-toggle', 'value')
    ]
)
def update_graph(selected_cities, selected_ratings, selected_acc_types, selected_chain_status, days_range, start_date, end_date, y_axis):
    filtered_df = all_cities_df.copy()
    if selected_cities:
        filtered_df = filtered_df[filtered_df['city_id'].isin(selected_cities)]
    if selected_ratings:
        filtered_df = filtered_df[filtered_df['star_rating'].isin(selected_ratings)]
    if selected_acc_types:
        filtered_df = filtered_df[filtered_df['accommodation_type_name'].isin(selected_acc_types)]
    if selected_chain_status:
        filtered_df = filtered_df[filtered_df['chain_hotel'].isin(selected_chain_status)]
    if start_date and end_date:
        filtered_df = filtered_df[(filtered_df['checkin_date'] >= start_date) & (filtered_df['checkout_date'] <= end_date)]
    filtered_df = filtered_df[(filtered_df['days_before_checkin'] >= days_range[0]) & (filtered_df['days_before_checkin'] <= days_range[1])]

    # Use selected y-axis (ADR or log(ADR))
    y_col = y_axis if y_axis in filtered_df.columns else 'ADR_USD'
    if not filtered_df.empty:
        trend_df = filtered_df.groupby('days_before_checkin')[y_col].mean().reset_index()
    else:
        trend_df = pd.DataFrame(columns=['days_before_checkin', y_col])

    # Calculate slope, intercept, and R^2 using OLS regression on grouped data
    if not trend_df.empty:
        X = trend_df[['days_before_checkin']]  # Ensure X is a DataFrame
        y = trend_df[y_col]
        X_ = sm.add_constant(X)
        model = sm.OLS(y, X_).fit()
        slope = model.params['days_before_checkin']
        intercept = model.params['const']
        r2 = model.rsquared
        # Format slope with sign and currency
        slope_sign = '+' if slope >= 0 else '-'
        slope_val = abs(slope)
        if y_col == 'ADR_USD':
            slope_str = f"{slope_sign}${slope_val:.2f} per day"
            intercept_str = f"${intercept:.2f}"
        else:
            slope_str = f"{slope_sign}{slope_val:.2f} per day (log)"
            intercept_str = f"{intercept:.2f} (log)"
        # Trend strength color class
        if r2 >= 0.7:
            trend_class = 'trend-strong'
            trend_strength = 'Strong'
        elif r2 >= 0.4:
            trend_class = 'trend-moderate'
            trend_strength = 'Moderate'
        else:
            trend_class = 'trend-weak'
            trend_strength = 'Weak'
        kpi_text = [
            html.Span([
                html.Span("Trend: ", style={'fontWeight': 'bold', 'fontSize': '1.2em'}),
                html.Span(f"{slope_str}", style={'fontWeight': 'bold', 'fontSize': '1.5em'})
            ], style={'display': 'block', 'textAlign': 'center', 'marginBottom': '0.5em'}),
            html.Span([
                html.Span("Base Price: ", style={'fontWeight': 'bold', 'fontSize': '1.2em'}),
                html.Span(f"{intercept_str}", style={'fontWeight': 'bold', 'fontSize': '1.5em'})
            ], style={'display': 'block', 'textAlign': 'center', 'marginBottom': '0.5em'}),
            html.Span(f"Trend Strength: {r2:.2f} ({trend_strength})", className=trend_class, style={'fontSize': '1.2em', 'display': 'block', 'textAlign': 'center'})
        ]
    else:
        kpi_text = [
            html.Span([
                html.Span("Trend: ", style={'fontWeight': 'bold', 'fontSize': '1.2em'}),
                html.Span("N/A", style={'fontWeight': 'bold', 'fontSize': '1.5em'})
            ], style={'display': 'block', 'textAlign': 'center', 'marginBottom': '0.5em'}),
            html.Span([
                html.Span("Base Price: ", style={'fontWeight': 'bold', 'fontSize': '1.2em'}),
                html.Span("N/A", style={'fontWeight': 'bold', 'fontSize': '1.5em'})
            ], style={'display': 'block', 'textAlign': 'center', 'marginBottom': '0.5em'}),
            html.Span("Trend Strength: N/A", style={'fontSize': '1.2em', 'display': 'block', 'textAlign': 'center'})
        ]

    # Create the scatter plot with a trendline using grouped data
    fig = px.scatter(
        trend_df,
        x='days_before_checkin',
        y=y_col,
        trendline="ols",
        trendline_color_override="red",
        title=f"{'Log ' if y_col == 'log_ADR_USD' else ''}Average Daily Rate (ADR_USD) vs. Days Before Check-in"
    )
    fig.update_layout(
        xaxis_title="Days Before Check-in",
        yaxis_title="Log(ADR_USD)" if y_col == 'log_ADR_USD' else "Average Daily Rate (ADR_USD in USD)",
        plot_bgcolor='#111',
        paper_bgcolor='#111',
        font=dict(color='#fff'),
        title_font_size=20,
        margin=dict(l=40, r=40, t=60, b=40)
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#333', zerolinecolor='#333', linecolor='#fff', tickfont=dict(color='#fff'), title_font=dict(color='#fff'))
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#333', zerolinecolor='#333', linecolor='#fff', tickfont=dict(color='#fff'), title_font=dict(color='#fff'))
    return fig, kpi_text[0], kpi_text[1], kpi_text[2]

# In the callback for 'adr-by-accommodation-bar', replace the bar chart with a treemap of top 5 accommodation types by mean ADR
@app.callback(
    Output('adr-by-accommodation-bar', 'figure'),
    [
        Input('city-filter', 'value'),
        Input('star-rating-filter', 'value'),
        Input('accommodation-type-filter', 'value'),
        Input('chain-hotel-filter', 'value'),
        Input('days-before-checkin-slider', 'value'),
        Input('date-range-filter', 'start_date'),
        Input('date-range-filter', 'end_date')
    ]
)
def update_adr_by_accommodation(selected_cities, selected_ratings, selected_acc_types, selected_chain_status, days_range, start_date, end_date):
    filtered_df = all_cities_df.copy()
    if selected_cities:
        filtered_df = filtered_df[filtered_df['city_id'].isin(selected_cities)]
    if selected_ratings:
        filtered_df = filtered_df[filtered_df['star_rating'].isin(selected_ratings)]
    if selected_acc_types:
        filtered_df = filtered_df[filtered_df['accommodation_type_name'].isin(selected_acc_types)]
    if selected_chain_status:
        filtered_df = filtered_df[filtered_df['chain_hotel'].isin(selected_chain_status)]
    if start_date and end_date:
        filtered_df = filtered_df[(filtered_df['checkin_date'] >= start_date) & (filtered_df['checkout_date'] <= end_date)]
    filtered_df = filtered_df[(filtered_df['days_before_checkin'] >= days_range[0]) & (filtered_df['days_before_checkin'] <= days_range[1])]
    bar_df = filtered_df.groupby('accommodation_type_name')['ADR_USD'].mean().reset_index()
    # Get top 5 accommodation types by mean ADR
    top5 = bar_df.nlargest(5, 'ADR_USD')
    if top5.empty or 'accommodation_type_name' not in top5.columns:
        fig = px.treemap(title='Top 5 Accommodation Types by ADR')
        fig.add_annotation(
            text="No data available for the selected filters.",
            xref="paper", yref="paper", showarrow=False, font=dict(size=16)
        )
        return fig
    fig = px.treemap(
        top5,
        path=['accommodation_type_name'],
        values='ADR_USD',
        color='ADR_USD',
        color_continuous_scale='Blues',
        title='Top 5 Accommodation Types by ADR',
        labels={'ADR_USD': 'Average Daily Rate (USD)', 'accommodation_type_name': 'Accommodation Type'}
    )
    fig.update_layout(
        plot_bgcolor='#111',
        paper_bgcolor='#111',
        font=dict(color='#fff'),
        title_font_size=20,
        margin=dict(l=40, r=40, t=60, b=40)
    )
    return fig

@app.callback(
    Output('adr-by-chain-bar', 'figure'),
    [
        Input('city-filter', 'value'),
        Input('star-rating-filter', 'value'),
        Input('accommodation-type-filter', 'value'),
        Input('chain-hotel-filter', 'value'),
        Input('days-before-checkin-slider', 'value'),
        Input('date-range-filter', 'start_date'),
        Input('date-range-filter', 'end_date')
    ]
)
def update_adr_by_chain(selected_cities, selected_ratings, selected_acc_types, selected_chain_status, days_range, start_date, end_date):
    filtered_df = all_cities_df.copy()
    if selected_cities:
        filtered_df = filtered_df[filtered_df['city_id'].isin(selected_cities)]
    if selected_ratings:
        filtered_df = filtered_df[filtered_df['star_rating'].isin(selected_ratings)]
    if selected_acc_types:
        filtered_df = filtered_df[filtered_df['accommodation_type_name'].isin(selected_acc_types)]
    if selected_chain_status:
        filtered_df = filtered_df[filtered_df['chain_hotel'].isin(selected_chain_status)]
    if start_date and end_date:
        filtered_df = filtered_df[(filtered_df['checkin_date'] >= start_date) & (filtered_df['checkout_date'] <= end_date)]
    filtered_df = filtered_df[(filtered_df['days_before_checkin'] >= days_range[0]) & (filtered_df['days_before_checkin'] <= days_range[1])]
    # Use the filtered_df for box plot, mapping chain_hotel to readable labels
    filtered_df['chain_hotel_label'] = filtered_df['chain_hotel'].map({1: 'Chain Hotel', 0: 'Independent Hotel'})
    fig = px.box(
        filtered_df,
        x='chain_hotel_label',
        y='ADR_USD',
        points='outliers',
        title='ADR Distribution by Chain Hotel Status',
        labels={'ADR_USD': 'Average Daily Rate (USD)', 'chain_hotel_label': 'Chain Hotel Status'}
    )
    fig.update_layout(
        xaxis_title="Chain Hotel Status",
        yaxis_title="Average Daily Rate (USD)",
        plot_bgcolor='#111',
        paper_bgcolor='#111',
        font=dict(color='#fff'),
        title_font_size=20,
        margin=dict(l=40, r=40, t=60, b=40)
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#333', zerolinecolor='#333', linecolor='#fff', tickfont=dict(color='#fff'), title_font=dict(color='#fff'))
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#333', zerolinecolor='#333', linecolor='#fff', tickfont=dict(color='#fff'), title_font=dict(color='#fff'))
    return fig

@app.callback(
    Output('adr-by-star-bar', 'figure'),
    [
        Input('city-filter', 'value'),
        Input('star-rating-filter', 'value'),
        Input('accommodation-type-filter', 'value'),
        Input('chain-hotel-filter', 'value'),
        Input('days-before-checkin-slider', 'value'),
        Input('date-range-filter', 'start_date'),
        Input('date-range-filter', 'end_date')
    ]
)
def update_adr_by_star(selected_cities, selected_ratings, selected_acc_types, selected_chain_status, days_range, start_date, end_date):
    filtered_df = all_cities_df.copy()
    if selected_cities:
        filtered_df = filtered_df[filtered_df['city_id'].isin(selected_cities)]
    if selected_ratings:
        filtered_df = filtered_df[filtered_df['star_rating'].isin(selected_ratings)]
    if selected_acc_types:
        filtered_df = filtered_df[filtered_df['accommodation_type_name'].isin(selected_acc_types)]
    if selected_chain_status:
        filtered_df = filtered_df[filtered_df['chain_hotel'].isin(selected_chain_status)]
    if start_date and end_date:
        filtered_df = filtered_df[(filtered_df['checkin_date'] >= start_date) & (filtered_df['checkout_date'] <= end_date)]
    filtered_df = filtered_df[(filtered_df['days_before_checkin'] >= days_range[0]) & (filtered_df['days_before_checkin'] <= days_range[1])]
    bar_df = filtered_df.groupby('star_rating')['ADR_USD'].mean().reset_index()
    if bar_df.empty or 'star_rating' not in bar_df.columns:
        fig = px.bar(title='ADR by Star Rating')
        fig.add_annotation(
            text="No data available for the selected filters.",
            xref="paper", yref="paper", showarrow=False, font=dict(size=16)
        )
        return fig
    fig = px.bar(
        bar_df,
        x='star_rating',
        y='ADR_USD',
        title='ADR by Star Rating',
        labels={'ADR_USD': 'Average Daily Rate (USD)', 'star_rating': 'Star Rating'}
    )
    fig.update_layout(
        plot_bgcolor='#111',
        paper_bgcolor='#111',
        font=dict(color='#fff'),
        title_font_size=20,
        margin=dict(l=40, r=40, t=60, b=40)
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#333', zerolinecolor='#333', linecolor='#fff', tickfont=dict(color='#fff'), title_font=dict(color='#fff'))
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#333', zerolinecolor='#333', linecolor='#fff', tickfont=dict(color='#fff'), title_font=dict(color='#fff'))
    return fig


# --- 4. Run the Application ---
if __name__ == '__main__':
    app.run(debug=True)
