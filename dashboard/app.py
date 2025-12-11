import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
import os
from itertools import product
from dotenv import load_dotenv

# --- Configuration and Setup ---

# Load .env
load_dotenv()
API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise ValueError("API_KEY not found in .env")

API_BASE = "https://api.census.gov/data/timeseries/healthins/sahie"

st.set_page_config(layout="wide", page_title="SAHIE Demographics Explorer")

st.title("SAHIE — Small Area Health Insurance Estimates by Demographic Groups")
st.markdown("Interactive time series exploration by combining multiple demographic characteristics. Source: U.S. Census SAHIE API.")

# --- Helper Functions and Mappings ---

@st.cache_data
def load_state_fips():
    """Loads a reference table for State FIPS codes."""
    url = "https://api.census.gov/data/2020/dec/pl?get=NAME&for=state:*"
    r = requests.get(url)
    df = pd.DataFrame(r.json()[1:], columns=r.json()[0])
    return df.rename(columns={"NAME": "state_name", "state": "state_fips"})


DEMOGRAPHIC_MAP = {
    "Race": ("RACECAT", "RACE_DESC"),
    "Age": ("AGECAT", "AGE_DESC"),
    "Sex": ("SEXCAT", "SEX_DESC")
}
DEMOGRAPHIC_DESCS = {k: v[1] for k, v in DEMOGRAPHIC_MAP.items()}

# --- SAHIE API Fetch ---

# Combined list of needed demographic description and category variables
ALL_CAT_VARS = ",".join([v[1] + "," + v[0] for v in DEMOGRAPHIC_MAP.values()])

# Base variables to fetch, including both PCT Uninsured and PCT Insured with their 90% CIs
GET_VARS = f"YEAR,NAME,STATE,PCTUI_PT,PCTUI_LB90,PCTUI_UB90,PCTIC_PT,PCTIC_LB90,PCTIC_UB90,{ALL_CAT_VARS}"


@st.cache_data(show_spinner=True)
def fetch_for_year(year, for_clause):
    """Fetches a single year/geo combination from the SAHIE API."""
    params = {
        "get": GET_VARS,
        "time": str(year),
        "key": API_KEY
    }
    params.update(for_clause)

    r = requests.get(API_BASE, params=params)
    r.raise_for_status()
    
    # Handle the JSON decoding (re-raise specific error if non-JSON is returned)
    try:
        j = r.json()
    except requests.exceptions.JSONDecodeError as e:
        st.error(f"API Error for year {year}. Check console for details. (Did you select an unavailable year?)")
        raise e
        
    df = pd.DataFrame(j[1:], columns=j[0])
    
    # Convert types
    df["YEAR"] = pd.to_datetime(df["YEAR"], format="%Y")
    numeric_cols = ["PCTUI_PT", "PCTUI_LB90", "PCTUI_UB90", "PCTIC_PT", "PCTIC_LB90", "PCTIC_UB90"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            
    return df


@st.cache_data
def fetch_range(years, for_clause):
    """Fetches data across the selected year range."""
    frames = [fetch_for_year(y, for_clause) for y in years]
    return pd.concat(frames, ignore_index=True)


# --- Sidebar: Global Controls ---

st.sidebar.markdown("### Geography")
geo_level = st.sidebar.selectbox("Geography level", ["US", "State"])

if geo_level == "US":
    for_clause = {"for": "us:1"}
    geo_label = "United States"
else:
    state_df = load_state_fips()
    state_name = st.sidebar.selectbox("Select a state", state_df["state_name"].sort_values())
    state_fips = state_df[state_df.state_name == state_name].state_fips.values[0]
    for_clause = {"for": f"state:{state_fips}"}
    geo_label = state_name

st.sidebar.markdown("### Time Period")
year_min = st.sidebar.number_input("Start year", 2005, 2023, 2010)
year_max = st.sidebar.number_input("End year", 2005, 2023, 2022) # Safe default year
years = list(range(year_min, year_max + 1))

if year_min > year_max:
    st.error("Start year must be less than or equal to end year.")
    st.stop()


# --- Fetch Data (with Data Cleaning to remove overall totals) ---

with st.spinner(f"Fetching and cleaning SAHIE data for {geo_label} ({year_min}-{year_max})…"):
    df = fetch_range(years, for_clause)

if df.empty:
    st.error("No data returned for that geography/time selection. Try different years or geography.")
    st.stop()

# --- Filter out the overall total rows that are not broken down by any demographic. ---
all_group_values = ['All Ages', 'Both Sexes', 'All Races/Ethnicities'] 

# Remove rows where ALL demographics are set to the 'All Groups' equivalent.
clean_mask = ~(
    (df['RACE_DESC'].isin(all_group_values)) & 
    (df['AGE_DESC'].isin(all_group_values)) & 
    (df['SEX_DESC'].isin(all_group_values))
)

df = df[clean_mask].copy()

if df.empty:
    st.error("The selected range only contains overall total data, which has been filtered out for clean plotting. Please select demographic filters.")
    st.stop()
# ------------------------------------


# --- Sidebar: Metric and Display ---
st.sidebar.markdown("### Metric and Display")

metric_choice = st.sidebar.selectbox(
    "Select Metric",
    ["Percent Uninsured", "Percent Insured"],
    key='metric_choice' # Added key for dynamic retrieval
)
show_ci = st.sidebar.checkbox("Show 90% Confidence Interval", value=True)

if metric_choice == "Percent Uninsured":
    metric_col = "PCTUI_PT"
    lb_col = "PCTUI_LB90"
    ub_col = "PCTUI_UB90"
else: # Percent Insured
    metric_col = "PCTIC_PT"
    lb_col = "PCTIC_LB90"
    ub_col = "PCTIC_UB90"


# --- Chart-Specific Filters (Moved to Sidebar) ---
st.sidebar.markdown("---")
st.sidebar.subheader("Chart Specific Filters")

# Prepare options for filters
filter_options = {}
for demo_type, desc_col in DEMOGRAPHIC_DESCS.items():
    filter_options[demo_type] = sorted(df[desc_col].dropna().unique())

# 1. Race-Only Filter
with st.sidebar.expander("Race Chart Filters", expanded=True):
    race_filters = st.multiselect(
        "Select Race groups:",
        filter_options["Race"],
        key="race_sel",
        default=[],
    )

# 2. Sex-Only Filter
with st.sidebar.expander("Sex Chart Filters", expanded=True):
    sex_filters = st.multiselect(
        "Select Sex groups:",
        filter_options["Sex"],
        key="sex_sel",
        default=[],
    )

# 3. Age-Only Filter
with st.sidebar.expander("Age Chart Filters", expanded=True):
    age_filters = st.multiselect(
        "Select Age groups:",
        filter_options["Age"],
        key="age_sel",
        default=[], 
    )

# 4. Combined Chart Filters
with st.sidebar.expander("Combined Chart Filters", expanded=False):
    combined_race_filters = st.multiselect(
        "Race:",
        filter_options["Race"],
        key="comb_race_sel",
        default=[]
    )
    combined_age_filters = st.multiselect(
        "Age:",
        filter_options["Age"],
        key="comb_age_sel",
        default=[]
    )
    combined_sex_filters = st.multiselect(
        "Sex:",
        filter_options["Sex"],
        key="comb_sex_sel",
        default=[]
    )
# --- End of Filter Controls ---


# --- Plotting Function ---

def create_time_series_plot(data, plot_type, filters, metric_col, lb_col, ub_col, show_ci, geo_label):
    """
    Creates a Plotly figure based on the selected filters and plot type.
    Handles aggregation for single-variable plots (Race, Age, Sex).
    """
    fig = go.Figure()
    
    if plot_type in ["Race", "Sex", "Age"]:
        # --- Single Variable Plotting (Requires Aggregation) ---
        
        # Determine the column to plot against
        plot_col = DEMOGRAPHIC_DESCS[plot_type]
        
        for group_value in filters:
            # Mask for the specific group (e.g., 'White' for Race chart)
            mask = data[plot_col] == group_value
            sub = data[mask].copy()

            # Aggregate across all other demographic dimensions (Age, Sex, or Race/Sex)
            # Grouping by YEAR ensures a single line per group per year
            grouped_sub = sub.groupby('YEAR').agg({
                metric_col: 'mean',
                lb_col: 'min', 
                ub_col: 'max'
            }).reset_index().sort_values("YEAR")

            if grouped_sub.empty:
                continue

            # 1. Plot CI
            if show_ci:
                fig.add_trace(go.Scatter(
                    x=pd.concat([grouped_sub["YEAR"], grouped_sub["YEAR"].iloc[::-1]]),
                    y=pd.concat([grouped_sub[ub_col], grouped_sub[lb_col].iloc[::-1]]),
                    fill='toself',
                    fillcolor='rgba(0,100,80,0.1)',
                    line=dict(width=0),
                    hoverinfo="skip",
                    showlegend=False
                ))
            
            # 2. Plot Point Estimate
            fig.add_trace(go.Scatter(
                x=grouped_sub["YEAR"],
                y=grouped_sub[metric_col],
                mode="lines+markers",
                name=group_value,
                line=dict(width=2)
            ))
            
    elif plot_type == "Combined":
        # --- Combined Cross-Tab Plotting ---
        # Strategy: For any dimension left blank, filter to the "All" category.
        # This prevents "jagged lines" caused by plotting multiple sub-groups under one label.
        
        defaults = {
            'RACE_DESC': 'All Races/Ethnicities',
            'AGE_DESC': 'All Ages',
            'SEX_DESC': 'Both Sexes'
        }
        
        # Use selected filters if they exist, otherwise use the default 'All' value
        r_opts = combined_race_filters if combined_race_filters else [defaults['RACE_DESC']]
        a_opts = combined_age_filters if combined_age_filters else [defaults['AGE_DESC']]
        s_opts = combined_sex_filters if combined_sex_filters else [defaults['SEX_DESC']]
        
        # Generate all combinations of the selected (or default) options
        active_combos = list(product(r_opts, a_opts, s_opts))
        
        has_plotted = False
        
        for r, a, s in active_combos:
            # Skip the "Total Total" if it was filtered out in data cleaning
            # (The clean_mask removed rows where ALL THREE are default)
            if r == defaults['RACE_DESC'] and a == defaults['AGE_DESC'] and s == defaults['SEX_DESC']:
                continue
                
            # Filter the dataframe
            mask = (
                (data['RACE_DESC'] == r) & 
                (data['AGE_DESC'] == a) & 
                (data['SEX_DESC'] == s)
            )
            
            sub = data[mask].sort_values("YEAR")
            
            if sub.empty:
                continue
            
            has_plotted = True
            
            # Construct a label based only on the specific (non-default) selections
            label_parts = []
            if r != defaults['RACE_DESC']: label_parts.append(r)
            if a != defaults['AGE_DESC']: label_parts.append(a)
            if s != defaults['SEX_DESC']: label_parts.append(s)
            
            label = " | ".join(label_parts)
            if not label: label = "Total" # Fallback

            # 1. Plot CI
            if show_ci and lb_col in sub.columns and ub_col in sub.columns:
                fig.add_trace(go.Scatter(
                    x=pd.concat([sub["YEAR"], sub["YEAR"].iloc[::-1]]),
                    y=pd.concat([sub[ub_col], sub[lb_col].iloc[::-1]]),
                    fill='toself',
                    fillcolor='rgba(0,100,80,0.1)',
                    line=dict(width=0),
                    hoverinfo="skip",
                    showlegend=False
                ))
            
            # 2. Plot Point Estimate
            fig.add_trace(go.Scatter(
                x=sub["YEAR"],
                y=sub[metric_col],
                mode="lines+markers",
                name=label,
                line=dict(width=2)
            ))
            
        if not has_plotted:
            return fig, False
            
    else:
        return fig, False # Should not happen

    # --- Layout Update (Shared) ---
    fig.update_layout(
        title=f"{plot_type} Breakdowns ({geo_label})",
        xaxis_title="Year",
        yaxis_title=f"{metric_choice} (%)",
        template="plotly_white",
        height=350,
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        )
    )
    return fig, True


# --- Data Retrieval Function for Download ---

def get_plot_data(data, plot_type, filters, metric_col, lb_col, ub_col):
    """Retrieves the aggregated/filtered DataFrame used for a specific chart.
    
    For plot_type in ["Race", "Sex", "Age"], filters is a list of selected groups.
    For plot_type == "Combined", filters is a dictionary: 
        {'RACE_DESC': race_list, 'AGE_DESC': age_list, 'SEX_DESC': sex_list}
    """
    
    if plot_type in ["Race", "Sex", "Age"]:
        plot_col = DEMOGRAPHIC_DESCS[plot_type]
        
        if not filters:
            return pd.DataFrame()
        
        # 1. Filter down to the selected groups
        mask = data[plot_col].isin(filters)
        sub = data[mask].copy()

        if sub.empty:
            return pd.DataFrame()

        # 2. Aggregate across all other demographic dimensions, keeping the group column
        # This mirrors the chart's logic, ensuring smooth lines.
        grouping_cols = ['YEAR', plot_col]
        
        grouped_sub = sub.groupby(grouping_cols).agg(
            {
                metric_col: 'mean',
                lb_col: 'min', 
                ub_col: 'max'
            }
        ).reset_index().sort_values("YEAR")
        
        # Rename columns for clarity in download
        grouped_sub = grouped_sub.rename(columns={
            metric_col: f"{plot_type}_Aggregated_Metric",
            lb_col: f"{plot_type}_Aggregated_CI_LB90",
            ub_col: f"{plot_type}_Aggregated_CI_UB90",
        })
        
        return grouped_sub
            
    elif plot_type == "Combined":
        defaults = {
            'RACE_DESC': 'All Races/Ethnicities',
            'AGE_DESC': 'All Ages',
            'SEX_DESC': 'Both Sexes'
        }
        
        # Extract lists from filters dictionary
        r_list = filters.get('RACE_DESC', [])
        a_list = filters.get('AGE_DESC', [])
        s_list = filters.get('SEX_DESC', [])
        
        r_opts = r_list if r_list else [defaults['RACE_DESC']]
        a_opts = a_list if a_list else [defaults['AGE_DESC']]
        s_opts = s_list if s_list else [defaults['SEX_DESC']]
        
        final_mask = pd.Series([False] * len(data))
        
        active_combos = list(product(r_opts, a_opts, s_opts))
        
        for r, a, s in active_combos:
            if r == defaults['RACE_DESC'] and a == defaults['AGE_DESC'] and s == defaults['SEX_DESC']:
                continue
                
            combo_mask = (
                (data['RACE_DESC'] == r) & 
                (data['AGE_DESC'] == a) & 
                (data['SEX_DESC'] == s)
            )
            final_mask |= combo_mask
            
        return data[final_mask].sort_values(["YEAR", "RACE_DESC", "AGE_DESC", "SEX_DESC"])
    
    return pd.DataFrame()


# --- Main Application Layout: Four Charts Vertically ---

st.header(f"Health Insurance Estimates ({geo_label})")
st.subheader(f"Metric: {metric_choice}")

# Prepare options for filters (Global across charts)
filter_options = {}
for demo_type, desc_col in DEMOGRAPHIC_DESCS.items():
    filter_options[demo_type] = sorted(df[desc_col].dropna().unique())


# --- CHART 1: Race Only ---
st.markdown("### 1. Race Breakdowns")
if race_filters:
    fig1, success1 = create_time_series_plot(df, "Race", race_filters, metric_col, lb_col, ub_col, show_ci, geo_label)
    if success1:
        st.plotly_chart(fig1, use_container_width=True)
else:
    st.info("Select Race groups in the sidebar to view this chart.")

st.markdown("---")

# --- CHART 2: Sex Only ---
st.markdown("### 2. Sex Breakdowns")
if sex_filters:
    fig2, success2 = create_time_series_plot(df, "Sex", sex_filters, metric_col, lb_col, ub_col, show_ci, geo_label)
    if success2:
        st.plotly_chart(fig2, use_container_width=True)
else:
    st.info("Select Sex groups in the sidebar to view this chart.")

st.markdown("---")

# --- CHART 3: Age Only ---
st.markdown("### 3. Age Breakdowns")
if age_filters:
    fig3, success3 = create_time_series_plot(df, "Age", age_filters, metric_col, lb_col, ub_col, show_ci, geo_label)
    if success3:
        st.plotly_chart(fig3, use_container_width=True)
else:
    st.info("Select Age groups in the sidebar to view this chart.")

st.markdown("---")

# --- CHART 4: Combined Filters ---
st.markdown("### 4. Combined Demographic Breakdown")
if combined_race_filters or combined_age_filters or combined_sex_filters:
    fig4, success4 = create_time_series_plot(df, "Combined", None, metric_col, lb_col, ub_col, show_ci, geo_label)
    if success4:
        st.plotly_chart(fig4, use_container_width=True)
else:
    st.info("Select groups in the Combined Chart Filters (sidebar) to view this chart.")


# --- Data Download Options ---

st.markdown("---")
st.subheader("Data Download Options")

# Map of display name to a function/data call
download_options_map = {
    "All Raw Data (Filtered for Totals)": lambda: df,
    "Race Chart Data (Aggregated)": lambda: get_plot_data(df, "Race", race_filters, metric_col, lb_col, ub_col),
    "Sex Chart Data (Aggregated)": lambda: get_plot_data(df, "Sex", sex_filters, metric_col, lb_col, ub_col),
    "Age Chart Data (Aggregated)": lambda: get_plot_data(df, "Age", age_filters, metric_col, lb_col, ub_col),
    # Fixed combined chart download to pass the explicit filter dictionary
    "Combined Chart Data (Granular)": lambda: get_plot_data(df, "Combined", 
        {'RACE_DESC': combined_race_filters, 'AGE_DESC': combined_age_filters, 'SEX_DESC': combined_sex_filters}, 
        metric_col, lb_col, ub_col)
}

download_selection = st.selectbox(
    "Select the dataset you wish to download:",
    list(download_options_map.keys())
)

# Fetch the selected data and convert to CSV
selected_df = download_options_map[download_selection]()
selected_csv = selected_df.to_csv(index=False)
file_name = f"{download_selection.lower().replace(' ', '_').replace('(', '').replace(')', '')}_sahie.csv"

st.download_button(
    f"Download '{download_selection}'",
    selected_csv,
    file_name
)