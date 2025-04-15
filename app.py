import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc
import logging
import ast
from collections import Counter
import re
import io
import base64
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# Setup logging
nltk.download('stopwords')
STOPWORDS = set(stopwords.words('english')) | {'us'}  # Add 'us' to stopwords since it's a country but often a stopword in text
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define countries of interest
COUNTRIES_OF_INTEREST = ['Brazil', 'Jordan', 'Peru', 'Colombia', 'Japan', 'Norway', 'Indonesia', 'Thailand', 'Guyana', 'United States']
COUNTRIES_OF_INTEREST_REDDIT = ['Brazil', 'Jordan', 'Peru', 'Colombia', 'Japan', 'Norway', 'Indonesia', 'Thailand', 'Guyana']
COUNTRY_COLORS = {
    'Brazil': '#1F77B4',         # Blue
    'Jordan': '#FF7F0E',         # Orange
    'Peru': '#2CA02C',           # Green
    'Colombia': '#D62728',       # Red
    'Japan': '#9467BD',          # Purple
    'Norway': '#8C564B',         # Brown
    'Indonesia': '#E377C2',      # Pink
    'Thailand': '#7F7F7F',       # Gray
    'Guyana': '#BCBD22',         # Olive
    'United States': '#17BECF',  # Teal
    'United Kingdom': '#AEC7E8'  # Light Blue 
}
# Define event types
NEGATIVE_EVENTS = ['Social Unrest', 'Economic Crisis', 'Political Instability', 'Conflict', 'Natural Disaster']
POSITIVE_EVENTS = ['Economic Recovery', 'Policy Reform', 'Social Improvement', 'Peace Resolution']
# Define event types (updated to match your new data)
NEGATIVE_EVENTS_REDDIT = ['Social Unrest', 'Economic Crisis', 'Political Instability', 'Conflict', 'Natural Disaster']
POSITIVE_EVENTS_REDDIT = ['Economic Growth', 'Political Stability', 'Peacebuilding', 'Social Progress', 'Technological Development', 'Environmental Action']
# Event severity weights
EVENT_SEVERITY_WEIGHTS = {
    'Social Unrest': 1.0, 'Economic Crisis': 1.0, 'Political Instability': 1.0, 'Conflict': 1.0, 'Natural Disaster': 1.0,
    'Economic Recovery': 1.0, 'Policy Reform': 1.0, 'Social Improvement': 1.0, 'Peace Resolution': 1.0, 'Other': 0.0
}
# Event severity weights (updated to include new event types)
EVENT_SEVERITY_WEIGHTS_REDDIT = {
    'Social Unrest': 1.0, 'Economic Crisis': 1.0, 'Political Instability': 1.0, 'Conflict': 1.0, 'Natural Disaster': 1.0,
    'Economic Growth': 1.0, 'Political Stability': 1.0, 'Peacebuilding': 1.0, 'Social Progress': 1.0,
    'Technological Development': 1.0, 'Environmental Action': 1.0, 'Other': 0.0
}
# Load historical data
try:
    df = pd.read_csv('data/annual_data_v3.1.csv')
    logger.info(f"Loaded historical data with {len(df)} rows.")
except FileNotFoundError as e:
    logger.error(f"Error: Historical data file not found: {e}")
    raise

# CSV file options for news data
CSV_OPTIONS = {
    '29 March': 'data/analysed_articles_29_march_filtered.csv',
    '8 April': 'data/analysed_articles_8_april_filtered.csv'
}

# Function to load news data
def load_news_data(csv_path):
    try:
        news_df = pd.read_csv(csv_path)
        news_df['countries_detected'] = news_df['countries_detected'].apply(ast.literal_eval)
        news_df['published_at'] = pd.to_datetime(news_df['published_at'], errors='coerce')
        logger.info(f"Loaded news data from {csv_path} with {len(news_df)} rows.")
        return news_df
    except FileNotFoundError as e:
        logger.error(f"Error: News data file not found: {e}")
        return pd.DataFrame()

try:
    reddit_df = pd.read_csv('data/gemini_redditv2.csv')  # Update path to your new data file
    # Process migration_intent and country columns
    reddit_df['migration_intent'] = reddit_df['migration_intent'].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )
    reddit_df['country'] = reddit_df['country'].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )
    # Normalize nested dictionaries
    migration_intent_df = pd.json_normalize(reddit_df['migration_intent'])
    country_df = pd.json_normalize(reddit_df['country'])
    reddit_df = pd.concat([reddit_df, migration_intent_df, country_df], axis=1)
    reddit_df = reddit_df.drop(columns=['migration_intent', 'country'])
    # Rename columns for consistency
    reddit_df = reddit_df.rename(columns={
        'present': 'migration_present',
        'confidence': 'migration_confidence',
        'name': 'country',
    })
    # Clean sentiment column (remove % and convert to int)
    reddit_df['sentiment'] = reddit_df['sentiment'].str.replace('%', '').astype(int)
    
    # Handle date column
    logger.info(f"Sample date values before processing: {reddit_df['date'].head().tolist()}")
    reddit_df['date'] = reddit_df['date'].astype(str)
    reddit_df = reddit_df[reddit_df['date'].str.strip() == '2025']
    reddit_df['date'] = pd.to_datetime(reddit_df['date'] + '-01-01', errors='coerce')
    reddit_df['date'] = reddit_df['date'].dt.to_period('W').apply(lambda r: r.start_time if pd.notnull(r) else pd.NaT)
    
    logger.info(f"Loaded Reddit data with {len(reddit_df)} rows for 2025.")
    logger.info(f"Columns after processing: {reddit_df.columns.tolist()}")
    logger.info(f"Sample data: {reddit_df[['sentiment', 'event_type', 'migration_present', 'country']].head().to_dict()}")
except FileNotFoundError as e:
    logger.error(f"Error: Reddit data file not found: {e}")
    reddit_df = pd.DataFrame({
        'sentiment': [50], 'sentiment_reasoning': ['Sample reasoning'], 'event_type': ['Other'],
        'migration_present': ['not present'], 'migration_confidence': [0.0], 'country': ['United States'],
        'ThreadTitle': ['Sample Reddit Post'], 'date': ['2025']
    })
    reddit_df['date'] = pd.to_datetime(reddit_df['date'] + '-01-01', errors='coerce')
    reddit_df['date'] = reddit_df['date'].dt.to_period('W').apply(lambda r: r.start_time)
    logger.info(f"Loaded sample Reddit data with {len(reddit_df)} rows.")

countries_historical = df['country'].unique().tolist()
countries_historical.insert(0, "All Countries")
years = sorted(df['year'].unique().tolist())
reddit_years = ["2025"]
# Define indicators and metrics
all_indicators = [
    'inflation_period_average', 'wb_unemployment_rate', 'min_wage_monthly', 'informal_employment_rate',
    'working_poverty_rate', 'cpi_rank', 'cpi_score', 'education_level', 'employment_rate', 'gdp_growth',
    'political_stability', 'rule_of_law', 'epi_score', 'gpi_rank', 'gpi_score', 'happiness_score',
    'social_support_score', 'ilo_unemployment_rate', 'hdi_score', 'avg_education_length', 'life_expectancy',
    'uk_imports', 'uk_exports', 'real_gdp_growth', 'gdp_current_usd', 'fdi_net_outflows_pct_gdp',
    'fdi_net_inflows_pct_gdp', 'gdp_per_capita_current_usd', 'Inflation_Period_Average'
]

main_metrics = ['MOS', 'MRI', 'MWI', 'asylum_applications', 'net_migration', 'EPS', 'SSS', 'GRS', 'ESS', 'HCM', 'DTR']
METRIC_ORDER = ['MOS', 'MWI_MRI', 'EPS', 'SSS', 'GRS', 'ESS', 'HCM', 'DTR', 'net_migration', 'asylum_applications']

metric_descriptions = {
    'MOS': {
        'name': 'Migration Opportunity Score (MOS)',
        'description': (
            'The Migration Opportunity Score (MOS) is derived from two indices: Migration Risk Index (MRI) and Migration Win Index (MWI). '
            'It balances migration pressures and opportunities using the formula:\n'
            'MOS = α × MRI − β × MWI\n'
            'where α and β are adjustable weightings, currently set to 0.5 and 0.5, calibrated through sensitivity analysis.'
        )
    },
    'MRI': {
        'name': 'Migration Risk Index (MRI)',
        'description': (
            'The Migration Risk Index (MRI) measures factors that increase migration pressures, aggregating three sub-scores:\n'
            'The MRI is calculated as:\n'
            'MRI = (0.3333 × EPS) + (0.3333 × SSS) + (0.3333 × GRS)'
        )
    },
    'EPS':{
        'name': 'Economic Pressure Score (EPS)',
        'description': (
            ' EPS = (0.3333 × Unemployment) + (0.3333 × Working Poverty) + 0.3333 × (1 - GDP Growth)\n')
    },
    'SSS':{
        'name': 'Social Stability Score (SSS)',
        'description': (
            ' SSS = (0.3333 × Happiness) + (0.3333 × Peace) + (0.3333 × HDI Score)\n')
    },
    'GRS':{
        'name': 'Governance Risk Score (GRS):',
        'description': (
            ' GRS = (0.3333 × Corruption) + (0.3333 × Political Stability) + (0.3333 × Rule of Law)\n')
    },
    'MWI': {
        'name': 'Migration Win Index (MWI)',
        'description': (
            'The Migration Win Index (MWI) captures positive migration attributes, such as compliance, economic stability, and beneficial bilateral relations:\n'
            'The MWI is calculated as:\n'
            ' MWI = (0.3333 × Economic Stability) + (0.3333 × Trade Relations) + (0.3333 × Human Capital)'
        )
    },
    'ESS':{
        'name': 'Economic Stability (ESS)',
        'description': (
            ' EPS = ESS = 0.5 × GDP Stability + 0.5 × Employment Rate\n')
    },
    'DTR':{
        'name': 'Diplomatic and Trade Relations (DTR)',
        'description': (
            ' DTR = 0.5 × Trade Volume with UK + 0.5 × FDI Net Inflows (% of GDP)\n')
    },
    'HCM':{
        'name': 'Human Capital Metrics (HCM):',
        'description': (
            ' HCM = 0.5 × Avg. Education Length + 0.5 × English Proficiency Score\n')
    },
    'asylum_applications': {
        'name': 'Asylum Applications',
        'description': 'The number of asylum applications submitted, indicating the level of forced migration due to conflict or persecution.'
    },
    'net_migration': {
        'name': 'Net Migration',
        'description': 'The difference between the number of immigrants and emigrants, showing the overall migration balance.',
        'unit': 'Count',
        'range': '[-∞, +∞]',
        'interpretation': 'Positive values represent immigration, negative values represent emigration',
        'source': 'World Bank'
    }
}

indicator_descriptions = {
    'ilo_unemployment_rate': {
        'name': 'Unemployment Rate (ILO)',
        'unit': 'Percent %',
        'range': '[0, 100]',
        'interpretation': 'Lower is better (less unemployment)',
        'source': 'ILO (International Labour Organisation)'
    },
    'wb_unemployment_rate': {
        'name': 'Unemployment Rate (World Bank)',
        'unit': 'Percent %',
        'range': '[0, 100]',
        'interpretation': 'Lower is better (less unemployment)',
        'source': 'World Bank'
    },
    'employment_rate': {
        'name': 'Employment Rate',
        'unit': 'Percent %',
        'range': '[0, 100]',
        'interpretation': 'Higher is better (more employment)',
        'source': 'World Bank'
    },
    'informal_employment_rate': {
        'name': 'Informal Employment Rate',
        'unit': 'Percent %',
        'range': '[0, 100]',
        'interpretation': 'Lower is better (less informal employment)',
        'source': 'ILO (International Labour Organisation)'
    },
    'min_wage_monthly': {
        'name': 'Minimum Wage (PPP)',
        'unit': 'Purchasing Power Parity (PPP)',
        'range': '[0, +∞]',
        'interpretation': 'Higher is better (greater purchasing power from wage)',
        'source': 'ILO (International Labour Organisation)'
    },
    'working_poverty_rate': {
        'name': 'Working Poverty Rate',
        'unit': 'Percent %',
        'range': '[0, 100]',
        'interpretation': 'Lower is better (less working poverty)',
        'source': 'World Bank'
    },
    'gdp_growth': {
        'name': 'GDP Growth',
        'unit': 'Percent %',
        'range': '[-∞, +∞]',
        'interpretation': 'Higher is better (greater economic growth)',
        'source': 'World Bank'
    },
    'political_stability': {
        'name': 'Political Stability Index',
        'unit': 'Standard Normal Distribution',
        'range': '~[-2.5, 2.5]',
        'interpretation': 'Higher is better (more stability, less violence)',
        'source': 'World Bank'
    },
    'rule_of_law': {
        'name': 'Rule of Law Index',
        'unit': 'Standard Normal Distribution',
        'range': '~[-2.5, 2.5]',
        'interpretation': 'Higher is better (stronger legal systems, less corruption)',
        'source': 'World Bank'
    },
    'education_level': {
        'name': 'Education Level',
        'unit': 'Percent %',
        'range': '[0, 100]',
        'interpretation': 'Higher is better (higher proportion of adults over age 25 with at least short cycle tertiary education or equivalent)',
        'source': 'World Bank'
    },
    'cpi_score': {
        'name': 'Corruption Perceptions Index Score',
        'unit': 'Index',
        'range': '[1, 100]',
        'interpretation': 'Higher is better (less perceived corruption)',
        'source': 'Transparency International'
    },
    'cpi_rank': {
        'name': 'Corruption Perceptions Index Rank',
        'unit': 'Index',
        'range': '[1, 180]',
        'interpretation': 'Lower is better (less perceived corruption)',
        'source': 'Transparency International'
    },
    'gpi_score': {
        'name': 'Global Peace Index Score',
        'unit': 'Index',
        'range': '[1, 5]',
        'interpretation': 'Lower is better (more peaceful)',
        'source': 'Institute for Economics and Peace (IEP)'
    },
    'gpi_rank': {
        'name': 'Global Peace Index Rank',
        'unit': 'Index',
        'range': '[1, 163]',
        'interpretation': 'Lower is better (more peaceful)',
        'source': 'Institute for Economics and Peace (IEP)'
    },
    'happiness_score': {
        'name': 'Happiness Score',
        'unit': 'Index',
        'range': '[1, 10]',
        'interpretation': 'Higher is better (greater life satisfaction)',
        'source': 'World Happiness Report'
    },
    'social_support_score': {
        'name': 'Social Support Score',
        'unit': 'Index',
        'range': '[0, 3]',
        'interpretation': 'Higher is better (stronger social networks)',
        'source': 'World Happiness Report'
    },
    'epi_score': {
        'name': 'English Proficiency Index Score',
        'unit': 'Index',
        'range': '[0, 800]',
        'interpretation': 'Higher score indicates greater English proficiency',
        'source': 'EF English Proficiency Index'
    },
    'net_migration': {
        'name': 'Net Migration',
        'unit': 'Count',
        'range': '[-∞, +∞]',
        'interpretation': 'Positive values represent immigration, negative values represent emigration',
        'source': 'World Bank'
    },
    'inflation_period_average': {
        'name': 'Inflation Period Average',
        'unit': 'Percent %',
        'range': '[-∞, +∞]',
        'interpretation': 'Lower is better (less inflation)',
        'source': 'World Bank'
    },
    'hdi_score': {
        'name': 'Human Development Index (HDI) Score',
        'unit': 'Index',
        'range': '[0, 1]',
        'interpretation': 'Higher is better (better human development)',
        'source': 'United Nations Development Programme (UNDP)'
    },
    'avg_education_length': {
        'name': 'Average Education Length',
        'unit': 'Years',
        'range': '[0, +∞]',
        'interpretation': 'Higher is better (longer education)',
        'source': 'World Bank'
    },
    'life_expectancy': {
        'name': 'Life Expectancy',
        'unit': 'Years',
        'range': '[0, +∞]',
        'interpretation': 'Higher is better (longer life expectancy)',
        'source': 'World Bank'
    },
    'uk_imports': {
        'name': 'UK Imports',
        'unit': 'USD',
        'range': '[0, +∞]',
        'interpretation': 'Higher indicates more trade with the UK',
        'source': 'World Bank'
    },
    'uk_exports': {
        'name': 'UK Exports',
        'unit': 'USD',
        'range': '[0, +∞]',
        'interpretation': 'Higher indicates more trade with the UK',
        'source': 'World Bank'
    },
    'real_gdp_growth': {
        'name': 'Real GDP Growth',
        'unit': 'Percent %',
        'range': '[-∞, +∞]',
        'interpretation': 'Higher is better (greater economic growth)',
        'source': 'World Bank'
    },
    'gdp_current_usd': {
        'name': 'GDP (Current USD)',
        'unit': 'USD',
        'range': '[0, +∞]',
        'interpretation': 'Higher indicates larger economy',
        'source': 'World Bank'
    },
    'fdi_net_outflows_pct_gdp': {
        'name': 'FDI Net Outflows (% of GDP)',
        'unit': 'Percent %',
        'range': '[-∞, +∞]',
        'interpretation': 'Higher indicates more investment abroad',
        'source': 'World Bank'
    },
    'fdi_net_inflows_pct_gdp': {
        'name': 'FDI Net Inflows (% of GDP)',
        'unit': 'Percent %',
        'range': '[-∞, +∞]',
        'interpretation': 'Higher indicates more foreign investment',
        'source': 'World Bank'
    },
    'gdp_per_capita_current_usd': {
        'name': 'GDP Per Capita (Current USD)',
        'unit': 'USD',
        'range': '[0, +∞]',
        'interpretation': 'Higher indicates higher economic output per person',
        'source': 'World Bank'
    },
    'Inflation_Period_Average': {
        'name': 'Inflation Period Average (Duplicate)',
        'unit': 'Percent %',
        'range': '[-∞, +∞]',
        'interpretation': 'Lower is better (less inflation)',
        'source': 'World Bank'
    }
}


# Event detection function
def detect_events(news_df):
    event_criteria = (
        (news_df['event_type'] != 'Other') &
        (news_df['migration_intent_confidence'] > 0.1)
    )
    events_df = news_df[event_criteria].copy()
    events_df = events_df.explode('countries_detected').reset_index(drop=True)
    events_df['event_id'] = events_df.groupby(['countries_detected', 'event_type', 'published_at']).ngroup()
    events_df = events_df.groupby('event_id').first().reset_index()
    logger.info(f"Detected {len(events_df)} significant events")
    return events_df

# Risk and win score calculation
def calculate_migration_risk(events_df):
    country_risk = {country: {
        'total_negative_events': 0, 'total_positive_events': 0,
        'avg_negative_sentiment_score': 0.0, 'avg_positive_sentiment_score': 0.0,
        'avg_negative_migration_intent': 0.0, 'avg_positive_migration_intent': 0.0,
        'total_negative_event_severity': 0.0, 'total_positive_event_severity': 0.0,
        'negative_event_types': {}, 'positive_event_types': {},
        'migration_risk_to_uk': 0.0, 'win_score': 0.0
    } for country in COUNTRIES_OF_INTEREST}
    negative_event_counts = events_df[events_df['event_type'].isin(NEGATIVE_EVENTS)]['countries_detected'].value_counts().to_dict()
    positive_event_counts = events_df[events_df['event_type'].isin(POSITIVE_EVENTS)]['countries_detected'].value_counts().to_dict()
    max_negative_events = max(negative_event_counts.values(), default=1)
    max_positive_events = max(positive_event_counts.values(), default=1)

    for country in COUNTRIES_OF_INTEREST:
        country_events = events_df[events_df['countries_detected'] == country]
        if country_events.empty:
            continue
        negative_events = country_events[country_events['event_type'].isin(NEGATIVE_EVENTS)]
        positive_events = country_events[country_events['event_type'].isin(POSITIVE_EVENTS)]
        total_negative_events = len(negative_events)
        total_positive_events = len(positive_events)
        avg_negative_sentiment_score = negative_events['sentiment_score'].mean() if total_negative_events > 0 else 0.0
        avg_positive_sentiment_score = positive_events['sentiment_score'].mean() if total_positive_events > 0 else 0.0
        avg_negative_migration_intent = negative_events['migration_intent_confidence'].mean() if total_negative_events > 0 else 0.0
        avg_positive_migration_intent = positive_events['migration_intent_confidence'].mean() if total_positive_events > 0 else 0.0
        negative_frequency_weight = min(2.0, 1 + (total_negative_events / max_negative_events)) if max_negative_events > 0 else 1.0
        positive_frequency_weight = min(2.0, 1 + (total_positive_events / max_positive_events)) if max_positive_events > 0 else 1.0
        total_negative_event_severity = sum(
            EVENT_SEVERITY_WEIGHTS.get(event_type, 0.5) * negative_frequency_weight * count
            for event_type, count in negative_events['event_type'].value_counts().items()
        )
        total_positive_event_severity = sum(
            EVENT_SEVERITY_WEIGHTS.get(event_type, 0.5) * positive_frequency_weight * count
            for event_type, count in positive_events['event_type'].value_counts().items()
        )
        country_risk[country].update({
            'total_negative_events': total_negative_events,
            'total_positive_events': total_positive_events,
            'avg_negative_sentiment_score': avg_negative_sentiment_score,
            'avg_positive_sentiment_score': avg_positive_sentiment_score,
            'avg_negative_migration_intent': avg_negative_migration_intent,
            'avg_positive_migration_intent': avg_positive_migration_intent,
            'total_negative_event_severity': total_negative_event_severity,
            'total_positive_event_severity': total_positive_event_severity,
            'negative_event_types': negative_events['event_type'].value_counts().to_dict(),
            'positive_event_types': positive_events['event_type'].value_counts().to_dict()
        })

    max_negative_severity = max([data['total_negative_event_severity'] for data in country_risk.values()], default=1)
    max_positive_severity = max([data['total_positive_event_severity'] for data in country_risk.values()], default=1)

    for country in COUNTRIES_OF_INTEREST:
        data = country_risk[country]
        norm_negative_severity = data['total_negative_event_severity'] / max_negative_severity if max_negative_severity > 0 else 0.0
        norm_positive_severity = data['total_positive_event_severity'] / max_positive_severity if max_positive_severity > 0 else 0.0
        migration_risk = (
            0.3333 * norm_negative_severity +
            0.3333 * (data['avg_negative_sentiment_score'] + 1) / 2 +
            0.3333 * data['avg_negative_migration_intent']
        )
        win_score = (
            0.3333 * norm_positive_severity +
            0.3333 * (data['avg_positive_sentiment_score'] + 1) / 2 +
            0.3333 * data['avg_positive_migration_intent']
        )
        data.update({
            'migration_risk_to_uk': migration_risk,
            'win_score': win_score
        })

    return pd.DataFrame.from_dict(country_risk, orient='index').reset_index().rename(columns={'index': 'country'})

def plot_reddit_graphs(filtered_df, selected_year):
    if filtered_df.empty:
        logger.warning("Filtered Reddit DataFrame is empty.")
        return html.P("No Reddit data available for the selected filters.")
    
    logger.info(f"Filtered Reddit data rows: {len(filtered_df)}")
    logger.info(f"Filtered data sample: {filtered_df[['sentiment', 'event_type', 'migration_present', 'country']].head().to_dict()}")
    
    # Check if this is a United States-only filter
    is_us_only = len(filtered_df['country'].unique()) == 1 and filtered_df['country'].iloc[0] == 'United States'
    
    # Sentiment Distribution
    sentiment_counts = filtered_df['sentiment'].value_counts().sort_index()
    fig_sentiment = go.Figure() if sentiment_counts.empty else px.bar(
        sentiment_counts.reset_index(),
        x='sentiment', y='count', color='sentiment',
        color_continuous_scale='RdYlGn', title="Sentiment Distribution of Threads (2025)"
    ).update_layout(
        xaxis_title="Sentiment Score (0 to 100)", yaxis_title="Number of Threads",
        plot_bgcolor='white', paper_bgcolor='white',
        xaxis=dict(tickmode='linear', dtick=5, range=[0, 100]),
        height=450, width=750 if is_us_only else 750  # Adjust width for US case
    )
    
    # Event Distribution
    event_counts = filtered_df.groupby(['country', 'event_type']).size().reset_index(name='count')
    if event_counts.empty:
        fig_events = go.Figure()
    else:
        if is_us_only:
            us_event_counts = filtered_df['event_type'].value_counts().reset_index()
            fig_events = px.bar(
                us_event_counts, x='event_type', y='count', color='event_type',
                color_discrete_sequence=px.colors.qualitative.Plotly,
                title="Event Type Distribution (United States, 2025)"
            ).update_layout(
                xaxis_title="Event Type", yaxis_title="Number of Threads",
                plot_bgcolor='white', paper_bgcolor='white',
                height=450, width=750, showlegend=False
            )
        else:
            fig_events = px.bar(
                event_counts, x='country', y='count', color='event_type',
                title="Event Type Distribution by Country (2025)", barmode='stack'
            ).update_layout(
                xaxis_title="Country", yaxis_title="Number of Threads",
                plot_bgcolor='white', paper_bgcolor='white',
                height=450, width=750
            )
    
    # Migration Intent Presence
    migration_counts = filtered_df['migration_present'].value_counts()
    fig_migration = go.Figure() if migration_counts.empty else px.bar(
        x=migration_counts.index, y=migration_counts.values,
        color=migration_counts.index,
        color_discrete_map={'not present': 'lightcoral', 'implicit': 'lightblue', 'explicit': 'lightgreen'},
        title="Migration Intent Presence in Threads (2025)"
    ).update_layout(
        xaxis_title="Migration Intent", yaxis_title="Number of Threads",
        plot_bgcolor='white', paper_bgcolor='white',
        height=450, width=750 if is_us_only else 750, showlegend=False
    )
    
    # Negative Events
    negative_events_df = filtered_df[filtered_df['event_type'].isin(NEGATIVE_EVENTS_REDDIT)]
    if negative_events_df.empty:
        fig_negative = go.Figure()
    else:
        if is_us_only:
            us_negative_counts = negative_events_df['event_type'].value_counts().reset_index()
            fig_negative = px.bar(
                us_negative_counts, x='event_type', y='count', color='event_type',
                color_discrete_sequence=px.colors.sequential.Reds,
                title="Negative Events (United States, 2025)"
            ).update_layout(
                xaxis_title="Negative Event Type", yaxis_title="Number of Threads",
                plot_bgcolor='white', paper_bgcolor='white',
                height=450, width=750, showlegend=False
            )
        else:
            negative_counts = negative_events_df.groupby(['country', 'event_type']).size().reset_index(name='count')
            fig_negative = px.bar(
                negative_counts, x='country', y='count', color='event_type',
                title="Negative Events by Country (2025)", barmode='stack',
                color_discrete_sequence=px.colors.sequential.Reds
            ).update_layout(
                xaxis_title="Country", yaxis_title="Number of Threads",
                plot_bgcolor='white', paper_bgcolor='white',
                height=450, width=750
            )
    
    # Positive Events
    positive_events_df = filtered_df[filtered_df['event_type'].isin(POSITIVE_EVENTS_REDDIT)]
    if positive_events_df.empty:
        fig_positive = go.Figure()
    else:
        if is_us_only:
            us_positive_counts = positive_events_df['event_type'].value_counts().reset_index()
            fig_positive = px.bar(
                us_positive_counts, x='event_type', y='count', color='event_type',
                color_discrete_sequence=px.colors.sequential.Greens,
                title="Positive Events (United States, 2025)"
            ).update_layout(
                xaxis_title="Positive Event Type", yaxis_title="Number of Threads",
                plot_bgcolor='white', paper_bgcolor='white',
                height=450, width=750, showlegend=False
            )
        else:
            positive_counts = positive_events_df.groupby(['country', 'event_type']).size().reset_index(name='count')
            fig_positive = px.bar(
                positive_counts, x='country', y='count', color='event_type',
                title="Positive Events by Country (2025)", barmode='stack',
                color_discrete_sequence=px.colors.sequential.Greens
            ).update_layout(
                xaxis_title="Country", yaxis_title="Number of Threads",
                plot_bgcolor='white', paper_bgcolor='white',
                height=450, width=750
            )
    
    # MOS Plot for United States
    fig_mos_us = go.Figure()
    if is_us_only:
        mos_df = calculate_mos_reddit_us(filtered_df, df)
        if not mos_df.empty:
            fig_mos_us = go.Figure(data=[
                go.Bar(name='Old MOS (2024)', x=['Old MOS (2024)'], y=[mos_df['mos_old'].values[0]], marker_color='cornflowerblue'),
                go.Bar(name='Real-Time MOS (2025)', x=['Real-Time MOS (2025)'], y=[mos_df['mos_real_time'].values[0]], marker_color='orange')
            ]).update_layout(
                plot_bgcolor='white', paper_bgcolor='white',
                title="Old MOS (2024) vs Real-Time MOS (2025) - United States",
                yaxis_title="MOS Score", xaxis_title="Migration Opportunity Score",
                barmode='group', height=450, width=750,
                hovermode='x unified',
            )
    
    # Layout
    if is_us_only:
        layout = [
            html.Div([
                html.Div([dcc.Graph(figure=fig_sentiment)], style={'width': '50%', 'display': 'inline-block'}),
                html.Div([dcc.Graph(figure=fig_events)], style={'width': '50%', 'display': 'inline-block'})
            ], style={'display': 'flex', 'flex-wrap': 'wrap', 'width': '100%'}),
            html.Div([
                html.Div([dcc.Graph(figure=fig_migration)], style={'width': '50%', 'display': 'inline-block'}),
                html.Div([dcc.Graph(figure=fig_negative)], style={'width': '50%', 'display': 'inline-block'})
            ], style={'display': 'flex', 'flex-wrap': 'wrap', 'width': '100%'}),
            html.Div([
                html.Div([dcc.Graph(figure=fig_positive)], style={'width': '50%', 'display': 'inline-block'}),
                html.Div([dcc.Graph(figure=fig_mos_us)], style={'width': '50%', 'display': 'inline-block'})
            ], style={'display': 'flex', 'flex-wrap': 'wrap', 'width': '100%'})
        ]
    else:
        layout = [
            html.Div([
                html.Div([dcc.Graph(figure=fig_sentiment)], style={'width': '50%', 'display': 'inline-block'}),
                html.Div([dcc.Graph(figure=fig_events)], style={'width': '50%', 'display': 'inline-block'})
            ], style={'display': 'flex', 'flex-wrap': 'wrap', 'width': '100%'}),
            html.Div([
                html.Div([dcc.Graph(figure=fig_migration)], style={'width': '50%', 'display': 'inline-block'}),
                html.Div([dcc.Graph(figure=fig_negative)], style={'width': '50%', 'display': 'inline-block'})
            ], style={'display': 'flex', 'flex-wrap': 'wrap', 'width': '100%'}),
            html.Div([
                html.Div([dcc.Graph(figure=fig_positive)], style={'width': '750px'})
            ], style={'display': 'flex', 'justify-content': 'center', 'width': '100%'})
        ]
    
    return html.Div(layout)
# MOS calculation function
def calculate_mos(risk_df, csv_path_name, selected_countries):
    mri_mwi_2024 = df[df['year'] == 2024].copy()
    merged_df = mri_mwi_2024.merge(risk_df[['country', 'migration_risk_to_uk', 'win_score']], on='country', how='inner')
    merged_df['real_time_mos'] = (merged_df['migration_risk_to_uk'] * merged_df['MRI']) - (merged_df['win_score'] * merged_df['MWI'])
    
    # Filter to selected countries and COUNTRIES_OF_INTEREST
    mos_df = merged_df[merged_df['country'].isin(selected_countries)][['country', 'MRI', 'MWI', 'migration_risk_to_uk', 'win_score', 'real_time_mos']]
    
    # Create MOS plot
    fig_mos = px.bar(mos_df, x='country', y='real_time_mos', 
                     title=f"Real-Time MOS Score by Country ({csv_path_name}) - Selected Countries",
                     hover_data=['MRI', 'MWI', 'migration_risk_to_uk', 'win_score']) if not mos_df.empty else go.Figure()

    # Comparison plot (old MOS vs real-time MOS)
    comparison_df = merged_df[merged_df['country'].isin(selected_countries)][['country', 'MOS', 'real_time_mos']]
    plot_df = comparison_df.melt(id_vars='country', value_vars=['MOS', 'real_time_mos'], var_name='mos_type', value_name='mos_value')
    fig_comparison = px.bar(plot_df, x='country', y='mos_value', color='mos_type', barmode='group',
                            title=f"Comparison of Old MOS and Real-Time MOS ({csv_path_name}) - Selected Countries",
                            labels={'mos_value': 'MOS Score', 'mos_type': 'MOS Type'}) if not plot_df.empty else go.Figure()
    
    return fig_mos, fig_comparison
def calculate_mos_reddit_us(filtered_df, historical_df):
    """
    Calculate MOS and Real-Time MOS for United States Reddit data.
    Adapted from your provided logic.
    """
    if filtered_df.empty or 'United States' not in filtered_df['country'].unique():
        return pd.DataFrame(columns=['country', 'mos', 'real_time_mos', 'migration_risk_to_uk', 'win_score'])

    # Filter to US data
    us_df = filtered_df[filtered_df['country'] == 'United States'].copy()

    # Filter for negative and positive events
    negative_events = us_df[us_df['event_type'].isin(NEGATIVE_EVENTS_REDDIT)]
    positive_events = us_df[us_df['event_type'].isin(POSITIVE_EVENTS_REDDIT)]

    total_negative_events = len(negative_events)
    total_positive_events = len(positive_events)

    # Calculate log ratio (handle division by zero)
    neg_to_pos_ratio = total_negative_events / total_positive_events if total_positive_events > 0 else float('inf')
    log_ratio = np.log(neg_to_pos_ratio) if neg_to_pos_ratio > 0 else 0  # Default to 0 if no positive events
    scaled_log_ratio = np.clip(log_ratio / np.log(100), 0, 1) if neg_to_pos_ratio > 0 else 0

    # Sentiment scores
    avg_negative_sentiment_score = negative_events['sentiment'].mean() if total_negative_events > 0 else 0.0
    avg_positive_sentiment_score = positive_events['sentiment'].mean() if total_positive_events > 0 else 0.0

    # Migration intent (proportion of 'implicit' or 'explicit')
    avg_negative_migration_intent = (negative_events['migration_present'] != "not present").mean() if total_negative_events > 0 else 0.0
    avg_positive_migration_intent = (positive_events['migration_present'] != "not present").mean() if total_positive_events > 0 else 0.0
    max_negative_events = total_negative_events or 1
    max_positive_events = total_positive_events or 1
    negative_frequency_weight = min(2.0, 1 + (total_negative_events / max_negative_events))
    positive_frequency_weight = min(2.0, 1 + (total_positive_events / max_positive_events))

    total_negative_event_severity = 0.0
    negative_event_types = negative_events['event_type'].value_counts().to_dict()
    for event_type, count in negative_event_types.items():
        base_severity = EVENT_SEVERITY_WEIGHTS.get(event_type, 0.5)
        total_negative_event_severity += base_severity * negative_frequency_weight * count

    total_positive_event_severity = 0.0
    positive_event_types = positive_events['event_type'].value_counts().to_dict()
    for event_type, count in positive_event_types.items():
        base_severity = EVENT_SEVERITY_WEIGHTS.get(event_type, 0.5)
        total_positive_event_severity += base_severity * positive_frequency_weight * count

    # Migration Risk and Win Score
    migration_risk = (
        0.5 * (avg_negative_sentiment_score / 100) +
        0.5 * avg_negative_migration_intent
    ) * (1 + 0.2 * scaled_log_ratio)

    win_score = (
        0.5 * (avg_positive_sentiment_score / 100) +
        0.5 * avg_positive_migration_intent
    ) * (1 - 0.2 * scaled_log_ratio)

    # Get historical MRI and MWI for 2024
    us_2024 = historical_df[(historical_df['country'] == 'United States') & (historical_df['year'] == 2024)]
    if us_2024.empty:
        logger.warning("No 2024 data for United States in annual_data_v3.1.csv; using default MOS.")
        mos_old, mri, mwi = 0.0, 0.5, 0.5  # Defaults if missing
    else:
        mos_old = us_2024['MOS'].values[0]
        mri = us_2024['MRI'].values[0]
        mwi = us_2024['MWI'].values[0]

    # Calculate Real-Time MOS
    mos_real_time = (migration_risk * mri) - (win_score * mwi)

    # Create DataFrame
    mos_df = pd.DataFrame({
        'country': ['United States'],
        'mos_old': [mos_old],  # 2024 MOS from annual_data_v3.1.csv
        'mos_real_time': [mos_real_time],  # 2025 MOS from Reddit
        'migration_risk_to_uk': [migration_risk],
        'win_score': [win_score]
    })
    logger.info(f"Calculated MOS for US: {mos_df.to_dict()}")
    return mos_df
def create_word_cloud_plot(df, title, text_column='title'):
    if df.empty:
        return go.Figure().update_layout(title=title)
    
    # Combine all text into a single string
    text = ' '.join(df[text_column].dropna().astype(str))
    
    # Clean text: lowercase, remove punctuation, split into words
    words = re.findall(r'\b\w+\b', text.lower())
    
    # Remove stopwords
    words = [word for word in words if word not in STOPWORDS]
    
    # Join words back into a single string for word cloud
    cleaned_text = ' '.join(words)
    
    if not cleaned_text.strip():
        return go.Figure().update_layout(title=title)
    
    # Generate word cloud
    wordcloud = WordCloud(width=800, height=400,
                          background_color='white',
                          min_font_size=10,
                          max_words=100).generate(cleaned_text)
    
    # Convert word cloud to image without GUI
    wordcloud_image = wordcloud.to_image()
    
    # Save image to bytes buffer
    buf = io.BytesIO()
    wordcloud_image.save(buf, format='PNG')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    
    # Create Plotly figure with the image
    fig = go.Figure()
    fig.add_layout_image(
        dict(
            source=f'data:image/png;base64,{img_str}',
            xref="paper", yref="paper",
            x=0, y=1,
            sizex=1, sizey=1,
            xanchor="left", yanchor="top",
            layer="below"
        )
    )
    fig.update_layout(
        title=title,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(l=0, r=0, t=30, b=0)
    )
    return fig
# Initialize Dash app
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)
app.title = "VISTA Dashboard"
server = app.server

# Layout
app.layout = html.Div([
    html.H1("VISTA Dashboard", style={'textAlign': 'center', 'color': '#ffffff', 'backgroundColor': '#1f77b4', 'padding': '20px', 'marginBottom': '0px', 'borderRadius': '5px 5px 0 0'}),
    dcc.Tabs(id="tabs", value='historical-trends', children=[
        dcc.Tab(label='Historical Trends', value='historical-trends', style={'fontSize': '18px', 'padding': '10px'}),
        dcc.Tab(label='Real-Time Sentiments', value='real-time-sentiments', style={'fontSize': '18px', 'padding': '10px'}),
    ]),
    html.Div(id='tabs-content', style={'padding': '20px', 'backgroundColor': '#e9ecef', 'minHeight': '100vh'})
])

@app.callback(
    Output('tabs-content', 'children'),
    Input('tabs', 'value')
)
def render_content(tab):
    if tab == 'historical-trends':
        return html.Div([
            html.Div([
                html.Div([
                    html.Label("Select Country(ies):", style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                    dcc.Dropdown(
                        id='country-dropdown-historical',
                        options=[{'label': country, 'value': country} for country in countries_historical],
                        value=["All Countries"],
                        multi=True,
                        style={'width': '100%'}
                    ),
                ], style={'width': '45%', 'padding': '10px'}),
                html.Div([
                    html.Label("Select Year(s):", style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                    dcc.Dropdown(
                        id='year-dropdown',
                        options=[{'label': str(year), 'value': year} for year in years],
                        value=years,
                        multi=True,
                        style={'width': '100%'}
                    ),
                ], style={'width': '45%', 'padding': '10px'}),
            ], style={'display': 'flex', 'justifyContent': 'center', 'width': '100%', 'marginBottom': '20px'}),
            html.Div([
                html.Div([
                    html.Label("Select Indicators to Compare:", style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                    dcc.Dropdown(
                        id='indicator-dropdown',
                        options=[{'label': indicator, 'value': indicator} for indicator in all_indicators],
                        value=['gdp_growth', 'happiness_score'],
                        multi=True,
                        style={'width': '100%'}
                    ),
                ], style={'width': '45%', 'padding': '10px'}),
                html.Div([
                    html.Label("Select Main Metrics to Display:", style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                    dcc.Dropdown(
                        id='main-metrics-dropdown',
                        options=[{'label': metric, 'value': metric} for metric in main_metrics],
                        value=['MOS', 'MRI', 'MWI', 'asylum_applications', 'net_migration'],
                        multi=True,
                        style={'width': '100%'}
                    ),
                ], style={'width': '45%', 'padding': '10px'}),
            ], style={'display': 'flex', 'justifyContent': 'center', 'width': '100%', 'marginBottom': '20px'}),
            html.Div(id='main-metrics-graphs-container', style={'width': '100%'}),
            html.Button("Show Metrics Descriptions", id="toggle-metrics-desc", n_clicks=0, style={'margin': '10px auto', 'display': 'block'}),
            html.Div(id='metrics-descriptions', style={'padding': '20px', 'display': 'none'}),
            html.Div(id='indicators-graphs-container', style={'width': '100%'}),
            html.Button("Show Indicators Descriptions", id="toggle-indicators-desc", n_clicks=0, style={'margin': '10px auto', 'display': 'block'}),
            html.Div(id='indicators-descriptions', style={'padding': '20px', 'display': 'none'})
        ], style={'backgroundColor': '#f8f9fa', 'padding': '20px', 'borderRadius': '5px', 'marginBottom': '20px'})
    elif tab == 'real-time-sentiments':
        return html.Div([
            html.Div([
                html.Div(id='csv-dropdown-container', children=[
                    html.Label("Select News Data File:", style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                    dcc.Dropdown(id='csv-dropdown', options=[{'label': k, 'value': v} for k, v in CSV_OPTIONS.items()],
                                 value=list(CSV_OPTIONS.values())[1], style={'width': '100%'})
                ], style={'width': '30%', 'padding': '5px'}),
                html.Div([
                    html.Label("Select Data Source:", style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                    dcc.Dropdown(id='data-source-dropdown', options=[{'label': 'News', 'value': 'news'}, {'label': 'Reddit', 'value': 'reddit'}],
                                 value='news', style={'width': '100%'})
                ], style={'width': '30%', 'padding': '5px'}),
                html.Div([
                    html.Label("Select Country(ies):", style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                    dcc.Dropdown(id='country-dropdown-sentiments', multi=True, value=COUNTRIES_OF_INTEREST_REDDIT, style={'width': '100%'})
                ], style={'width': '30%', 'padding': '5px'}),
            ], style={'display': 'flex', 'justifyContent': 'space-between', 'marginBottom': '10px'}),
            html.Div(id='reddit-year-container', children=[
                html.Label("Select Year (Reddit Only):", style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                dcc.Dropdown(id='reddit-year-dropdown', options=[{'label': year, 'value': year} for year in reddit_years],
                             value="2025", style={'width': '100%'})
            ], style={'width': '30%', 'padding': '5px', 'margin': '0 auto', 'display': 'none'}),
            html.Div(id='sentiment-graphs-container')
        ])

# Callbacks for Historical Trends
@app.callback(
    [Output('main-metrics-graphs-container', 'children'),
     Output('metrics-descriptions', 'children'),
     Output('indicators-graphs-container', 'children'),
     Output('indicators-descriptions', 'children')],
    [Input('country-dropdown-historical', 'value'),
     Input('year-dropdown', 'value'),
     Input('indicator-dropdown', 'value'),
     Input('main-metrics-dropdown', 'value')]
)
def update_historical_graphs(selected_countries, selected_years, selected_indicators, selected_main_metrics):
    try:
        if "All Countries" in selected_countries:
            filtered_df = df[df['year'].isin(selected_years)]
        else:
            filtered_df = df[
                (df['country'].isin(selected_countries)) &
                (df['year'].isin(selected_years))
            ]

        # Log the unique countries in filtered_df to verify UK presence
        logger.info(f"Unique countries in filtered_df: {filtered_df['country'].unique().tolist()}")

        required_columns = set(selected_main_metrics + selected_indicators)
        missing_columns = required_columns - set(filtered_df.columns)
        if missing_columns:
            error_msg = f"Missing columns in data: {missing_columns}"
            logger.error(error_msg)
            return (
                html.P(error_msg),
                html.P("Descriptions unavailable due to missing data."),
                html.P("Indicators unavailable due to missing data."),
                html.P("Indicator descriptions unavailable due to missing data.")
            )

        single_year = len(selected_years) == 1

        main_metrics_graphs = []
        metric_groups = []
        mwi_mri_metrics = [metric for metric in ['MWI', 'MRI'] if metric in selected_main_metrics]
        if mwi_mri_metrics:
            metric_groups.append(('MWI_MRI', mwi_mri_metrics))
        other_metrics = [metric for metric in selected_main_metrics if metric not in ['MWI', 'MRI']]
        for metric in other_metrics:
            metric_groups.append((metric, [metric]))

        sorted_metric_groups = []
        for metric_name in METRIC_ORDER:
            for group_name, metrics in metric_groups:
                if group_name == metric_name or (metric_name in metrics and group_name in metrics):
                    sorted_metric_groups.append((group_name, metrics))
                    break

        graph_elements = []
        for group_name, metrics in sorted_metric_groups:
            fig = go.Figure()
            if single_year:
                for metric in metrics:
                    for country in filtered_df['country'].unique():
                        country_df = filtered_df[filtered_df['country'] == country]
                        color = COUNTRY_COLORS.get(country, '#1F77B4')  # Default to blue if country not in mapping
                        fig.add_trace(go.Bar(
                            x=[country],
                            y=country_df[metric],
                            name=f"{country} - {metric}",
                            legendgroup=country,
                            showlegend=True,
                            marker_color=color
                        ))
                fig.update_layout(
                    barmode='group',
                    title="MWI and MRI" if group_name == 'MWI_MRI' else f"{metrics[0]}",
                    xaxis_title="Country",
                    yaxis_title="Value",
                    legend_title="Country / Metric",
                    title_font_size=14,
                    legend_font_size=10,
                    margin=dict(l=40, r=40, t=40, b=40),
                    height=450,
                    width=750,
                    plot_bgcolor='#ffffff'
                )
            else:
                for metric in metrics:
                    for country in filtered_df['country'].unique():
                        country_df = filtered_df[filtered_df['country'] == country]
                        line_style = 'dot' if metric == 'MRI' else 'solid'
                        color = COUNTRY_COLORS.get(country, '#1F77B4')  # Default to blue if country not in mapping
                        fig.add_trace(go.Scatter(
                            x=country_df['year'],
                            y=country_df[metric],
                            mode='lines+markers',
                            name=f"{country} - {metric}",
                            line=dict(dash=line_style, color=color),  # Use COUNTRY_COLORS
                            legendgroup=country,
                            showlegend=True
                        ))
                fig.update_layout(
                    title="MWI and MRI (MWI: Solid, MRI: Dotted)" if group_name == 'MWI_MRI' else f"{metrics[0]}",
                    xaxis_title="Year",
                    yaxis_title="Value",
                    legend_title="Country / Metric",
                    xaxis=dict(tickmode='linear'),
                    title_font_size=14,
                    legend_font_size=10,
                    margin=dict(l=40, r=40, t=40, b=40),
                    height=450,
                    width=750,
                    plot_bgcolor='#ffffff'
                )
            graph_elements.append(html.Div(dcc.Graph(figure=fig), style={'width': '50%', 'padding': '5px'}))

        if not selected_main_metrics:
            main_metrics_graphs = [html.Div(html.P("Select main metrics to display graphs."), style={'textAlign': 'center'})]
        else:
            rows = []
            for i in range(0, len(graph_elements), 2):
                row_graphs = graph_elements[i:i+2]
                row = html.Div(row_graphs, style={'display': 'flex', 'flexDirection': 'row', 'width': '100%', 'justifyContent': 'center', 'gap': '10px'})
                rows.append(row)
            main_metrics_graphs = rows

        metrics_descriptions_content = []
        if selected_main_metrics:
            metrics_descriptions_content.append(html.H4("Selected Metrics Descriptions"))
            for metric in selected_main_metrics:
                if metric in metric_descriptions:
                    desc = metric_descriptions[metric]
                    description = desc.get('description', 'No description available.')
                    description_lines = description.split('\n')
                    description_children = []
                    for i, line in enumerate(description_lines):
                        description_children.append(line)
                        if i < len(description_lines) - 1:
                            description_children.append(html.Br())
                    content = [
                        html.Strong(f"{desc.get('name', metric)}:"),
                        html.P(description_children)
                    ]
                    for field in ['unit', 'range', 'interpretation', 'source']:
                        if field in desc:
                            content.append(html.P(f"{field.capitalize()}: {desc[field]}"))
                    metrics_descriptions_content.append(html.Div(content))

        indicators_graphs = []
        if not selected_indicators:
            indicators_graphs = [html.Div(html.P("Select indicators to display graphs."), style={'textAlign': 'center'})]
        else:
            graph_elements = []
            for indicator in selected_indicators:
                fig = go.Figure()
                if single_year:
                    for country in filtered_df['country'].unique():
                        country_df = filtered_df[filtered_df['country'] == country]
                        color = COUNTRY_COLORS.get(country, '#1F77B4')  # Default to blue if country not in mapping
                        fig.add_trace(go.Bar(
                            x=[country],
                            y=country_df[indicator],
                            name=country,
                            legendgroup=country,
                            showlegend=True,
                            marker_color=color  # Use COUNTRY_COLORS
                        ))
                    fig.update_layout(
                        barmode='group',
                        title=f"{indicator}",
                        xaxis_title="Country",
                        yaxis_title="Value",
                        legend_title="Country",
                        title_font_size=14,
                        legend_font_size=10,
                        margin=dict(l=40, r=40, t=40, b=40),
                        height=450,
                        width=750,
                        plot_bgcolor='#ffffff'
                    )
                else:
                    for country in filtered_df['country'].unique():
                        country_df = filtered_df[filtered_df['country'] == country]
                        color = COUNTRY_COLORS.get(country, '#1F77B4')  # Default to blue if country not in mapping
                        fig.add_trace(go.Scatter(
                            x=country_df['year'],
                            y=country_df[indicator],
                            mode='lines+markers',
                            name=country,
                            line=dict(color=color),  # Use COUNTRY_COLORS
                            legendgroup=country,
                            showlegend=True
                        ))
                    fig.update_layout(
                        title=f"{indicator} Over Time",
                        xaxis_title="Year",
                        yaxis_title="Value",
                        legend_title="Country",
                        xaxis=dict(tickmode='linear'),
                        title_font_size=14,
                        legend_font_size=10,
                        margin=dict(l=40, r=40, t=40, b=40),
                        height=450,
                        width=750,
                        plot_bgcolor='#ffffff'
                    )
                graph_elements.append(html.Div(dcc.Graph(figure=fig), style={'width': '50%', 'padding': '5px'}))
            rows = []
            for i in range(0, len(graph_elements), 2):
                row_graphs = graph_elements[i:i+2]
                row = html.Div(row_graphs, style={'display': 'flex', 'flexDirection': 'row', 'width': '100%', 'justifyContent': 'center', 'gap': '10px'})
                rows.append(row)
            indicators_graphs = rows

        indicators_descriptions_content = []
        if selected_indicators:
            indicators_descriptions_content.append(html.H4("Selected Indicators Descriptions"))
            for indicator in selected_indicators:
                if indicator in indicator_descriptions:
                    desc = indicator_descriptions[indicator]
                    content = [
                        html.Strong(f"{desc['name']}:"),
                        html.P(f"Unit: {desc['unit']}"),
                        html.P(f"Range: {desc['range']}"),
                        html.P(f"Interpretation: {desc['interpretation']}"),
                        html.P(f"Source: {desc['source']}")
                    ]
                    indicators_descriptions_content.append(html.Div(content))

        return main_metrics_graphs, metrics_descriptions_content, indicators_graphs, indicators_descriptions_content
    except Exception as e:
        logger.error(f"Error in update_historical_graphs: {str(e)}")
        return (
            html.P(f"Error generating graphs: {str(e)}"),
            html.P("Descriptions unavailable due to error."),
            html.P("Indicators unavailable due to error."),
            html.P("Indicator descriptions unavailable due to error.")
        )
# Toggle Metrics Descriptions visibility
@app.callback(
    Output('metrics-descriptions', 'style'),
    Input('toggle-metrics-desc', 'n_clicks'),
    prevent_initial_call=True
)
def toggle_metrics_descriptions(n_clicks):
    if n_clicks is None:
        return {'padding': '20px', 'display': 'none'}
    if n_clicks % 2 == 1:
        return {'padding': '20px', 'display': 'block'}
    else:
        return {'padding': '20px', 'display': 'none'}

# Toggle Indicators Descriptions visibility
@app.callback(
    Output('indicators-descriptions', 'style'),
    Input('toggle-indicators-desc', 'n_clicks'),
    prevent_initial_call=True
)
def toggle_indicators_descriptions(n_clicks):
    if n_clicks is None:
        return {'padding': '20px', 'display': 'none'}
    if n_clicks % 2 == 1:
        return {'padding': '20px', 'display': 'block'}
    else:
        return {'padding': '20px', 'display': 'none'}
# Toggle CSV dropdown visibility
@app.callback(
    Output('csv-dropdown-container', 'style'),
    Input('data-source-dropdown', 'value')
)
def toggle_csv_dropdown(data_source):
    base_style = {'width': '30%', 'padding': '5px'}
    return {**base_style, 'display': 'block' if data_source == 'news' else 'none'}

# Toggle Reddit year dropdown visibility
@app.callback(
    Output('reddit-year-container', 'style'),
    Input('data-source-dropdown', 'value')
)
def toggle_reddit_year_dropdown(data_source):
    base_style = {'width': '30%', 'padding': '5px', 'margin': '0 auto'}
    return {**base_style, 'display': 'block' if data_source == 'reddit' else 'none'}

@app.callback(
    [Output('country-dropdown-sentiments', 'options'),
     Output('country-dropdown-sentiments', 'value')],
    [Input('data-source-dropdown', 'value'),
     Input('csv-dropdown', 'value')]
)
def update_country_options(data_source, csv_path):
    if data_source == 'news':
        news_df = load_news_data(csv_path)
        if news_df.empty:
            logger.warning(f"No data loaded from {csv_path}, using COUNTRIES_OF_INTEREST only")
            all_countries = COUNTRIES_OF_INTEREST.copy()
        else:
            # Handle potential missing or malformed 'countries_detected' column
            if 'countries_detected' not in news_df.columns:
                logger.warning(f"'countries_detected' column missing in {csv_path}, using COUNTRIES_OF_INTEREST")
                all_countries = COUNTRIES_OF_INTEREST.copy()
            else:
                # Ensure 'countries_detected' is list-like; handle exceptions
                try:
                    csv_countries = news_df['countries_detected'].dropna().explode().dropna().unique().tolist()
                    # Remove any None or empty string entries
                    csv_countries = [c for c in csv_countries if c and isinstance(c, str)]
                    all_countries = list(set(csv_countries + COUNTRIES_OF_INTEREST))
                except Exception as e:
                    logger.error(f"Error processing 'countries_detected': {str(e)}, falling back to COUNTRIES_OF_INTEREST")
                    all_countries = COUNTRIES_OF_INTEREST.copy()
        
        all_countries.sort()  # Sort for consistency
        return [{'label': c, 'value': c} for c in COUNTRIES_OF_INTEREST], COUNTRIES_OF_INTEREST
    else:
        country_options = ['All Countries', 'Countries of Interest', 'United States']
        return [{'label': c, 'value': c} for c in country_options], ['All Countries']

# Update sentiment graphs
@app.callback(
    Output('sentiment-graphs-container', 'children'),
    [Input('data-source-dropdown', 'value'),
     Input('country-dropdown-sentiments', 'value'),
     Input('csv-dropdown', 'value'),
     Input('reddit-year-dropdown', 'value')]
)
def update_sentiment_graphs(data_source, selected_countries, csv_path, reddit_year):
    if data_source == 'news':
        news_df = load_news_data(csv_path)
        csv_path_name = [k for k, v in CSV_OPTIONS.items() if v == csv_path][0]
        if not selected_countries:
            return html.P("No countries selected.")
        filtered_df = news_df[news_df['countries_detected'].apply(lambda x: any(country in x for country in selected_countries))]
        if filtered_df.empty:
            return html.P("No data available for selected countries.")

        events_df = detect_events(filtered_df)
        events_df_filtered = events_df[events_df['countries_detected'].isin(selected_countries)]
        negative_events_df = events_df_filtered[events_df_filtered['event_type'].isin(NEGATIVE_EVENTS)]
        positive_events_df = events_df_filtered[events_df_filtered['event_type'].isin(POSITIVE_EVENTS)]

        # Negative Events Plot
        fig_negative_events = go.Figure() if negative_events_df.empty else px.bar(
            negative_events_df.groupby(['countries_detected', 'event_type']).size().reset_index(name='count'),
            x='countries_detected', y='count', color='event_type', title="Negative Events by Country"
        ).update_layout(
            plot_bgcolor='white', paper_bgcolor='white',
            xaxis_title="Country", yaxis_title="Count",
            height=450, width=750
        )

        # Positive Events Plot
        fig_positive_events = go.Figure() if positive_events_df.empty else px.bar(
            positive_events_df.groupby(['countries_detected', 'event_type']).size().reset_index(name='count'),
            x='countries_detected', y='count', color='event_type', title="Positive Events by Country"
        ).update_layout(
            plot_bgcolor='white', paper_bgcolor='white',
            xaxis_title="Country", yaxis_title="Count",
            height=450, width=750
        )

        # Sentiment Distribution
        positive_sentiment_df = positive_events_df.groupby(['countries_detected', 'sentiment']).size().reset_index(name='count')
        negative_sentiment_df = negative_events_df.groupby(['countries_detected', 'sentiment']).size().reset_index(name='count')
        fig_positive_event_sentiment = go.Figure() if positive_sentiment_df.empty else px.bar(
            positive_sentiment_df, x='countries_detected', y='count', color='sentiment',
            title="Sentiment Distribution for Positive Events", barmode='stack'
        ).update_layout(
            plot_bgcolor='white', paper_bgcolor='white',
            xaxis_title="Country", yaxis_title="Count",
            height=450, width=750
        )
        fig_negative_event_sentiment = go.Figure() if negative_sentiment_df.empty else px.bar(
            negative_sentiment_df, x='countries_detected', y='count', color='sentiment',
            title="Sentiment Distribution for Negative Events", barmode='stack'
        ).update_layout(
            plot_bgcolor='white', paper_bgcolor='white',
            xaxis_title="Country", yaxis_title="Count",
            height=450, width=750
        )

        # Risk and Win Scores
        risk_df = calculate_migration_risk(events_df)
        risk_df_filtered = risk_df[risk_df['country'].isin(selected_countries)]
        fig_risk = go.Figure() if risk_df_filtered.empty else px.bar(
            risk_df_filtered, x='country', y='migration_risk_to_uk', title="Migration Risk Score by Country"
        ).update_layout(
            plot_bgcolor='white', paper_bgcolor='white',
            xaxis_title="Country", yaxis_title="Migration Risk Score",
            height=450, width=750
        )
        fig_win = go.Figure() if risk_df_filtered.empty else px.bar(
            risk_df_filtered, x='country', y='win_score', title="Win Score by Country"
        ).update_layout(
            plot_bgcolor='white', paper_bgcolor='white',
            xaxis_title="Country", yaxis_title="Win Score",
            height=450, width=750
        )

        # MOS Plots
        fig_mos, fig_comparison = calculate_mos(risk_df, csv_path_name, selected_countries)
        fig_mos.update_layout(
            plot_bgcolor='white', paper_bgcolor='white',
            xaxis_title="Country", yaxis_title="MOS Score",
            height=450, width=750
        )
        fig_comparison.update_layout(
            plot_bgcolor='white', paper_bgcolor='white',
            xaxis_title="Country", yaxis_title="MOS Score",
            height=450, width=750
        )

        # Word Cloud
        high_intent_df = filtered_df[filtered_df['migration_intent_confidence'] > 0.2]
        fig_word_cloud = create_word_cloud_plot(high_intent_df, f"Word Cloud - {csv_path_name}")
        # Word cloud already uses white background via WordCloud settings

        return html.Div([
            html.Div([dcc.Graph(figure=fig_negative_events)], style={'width': '50%', 'display': 'inline-block'}),
            html.Div([dcc.Graph(figure=fig_positive_events)], style={'width': '50%', 'display': 'inline-block'}),
            html.Div([dcc.Graph(figure=fig_negative_event_sentiment)], style={'width': '50%', 'display': 'inline-block'}),
            html.Div([dcc.Graph(figure=fig_positive_event_sentiment)], style={'width': '50%', 'display': 'inline-block'}),
            # html.Div([dcc.Graph(figure=fig_risk)], style={'width': '50%', 'display': 'inline-block'}),
            # html.Div([dcc.Graph(figure=fig_win)], style={'width': '50%', 'display': 'inline-block'}),
            html.Div([dcc.Graph(figure=fig_mos)], style={'width': '50%', 'display': 'inline-block'}),
            html.Div([dcc.Graph(figure=fig_comparison)], style={'width': '50%', 'display': 'inline-block'}),
            html.Div([
                html.Div([dcc.Graph(figure=fig_word_cloud)], style={'width': '800px'})
            ], style={'display': 'flex', 'justify-content': 'center', 'width': '100%'})
        ], style={'display': 'flex', 'flexWrap': 'wrap'})
    else:  # Reddit
        # Filter based on country selection
        if not selected_countries:
            return html.P("No countries selected.")
        if "All Countries" in selected_countries:
            filtered_df = reddit_df
        elif "Countries of Interest" in selected_countries:
            filtered_df = reddit_df[reddit_df['country'].isin(COUNTRIES_OF_INTEREST_REDDIT)]
        elif "United States" in selected_countries:
            filtered_df = reddit_df[reddit_df['country'] == 'United States']
        else:
            filtered_df = reddit_df[reddit_df['country'].isin(selected_countries)]
        return plot_reddit_graphs(filtered_df, reddit_year)

if __name__ == '__main__':
    # app.run(debug=True)
    app.run(host='0.0.0.0', port=8050)
