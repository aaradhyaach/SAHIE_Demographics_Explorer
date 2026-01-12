The live version of the interactive Streamlit dashboard can be found at the following link: 
https://aaradhyaach-sahie-explorer.streamlit.app/

1. Introduction
Health insurance coverage remains one of the most important determinants of access to care and long-term health outcomes in the United States. Over the past fifteen years, major federal and state policy changes (including the Affordable Care Act (ACA), Medicaid expansion, and temporary coverage protections introduced during the COVID-19 public health emergency) have reshaped the insurance landscape. These shifts have not affected all demographic groups. Persistent disparities continue to influence who receives coverage and who remains uninsured, particularly among racial and ethnic minority groups, younger adults, and residents of non-expansion states (Keisler-Starkey & Bunch, 2022).
To study these changes systematically, this project develops the Small Area Health Insurance Estimates (SAHIE) Demographics Explorer, an interactive web-based tool designed to visualize annual insured and uninsured rates across states, stratified by Race, Age, and Sex. The tool is built upon the Census Bureau’s SAHIE Time Series API, which provides single-year modeled estimates with corresponding 90% confidence intervals (Bauder et al., 2024). The goal is to give researchers and policymakers a clearer view of how coverage has changed over time (particularly during the COVID-19 period0 and to highlight demographic groups that may require additional policy attention.


2. Background and Motivation
The SAHIE program is the most comprehensive federal resource for modeled annual health insurance estimates at the national, state, and county levels. Unlike broad surveys such as the American Community Survey (ACS), SAHIE integrates multiple administrative and survey inputs to produce statistically stabilized estimates designed for small-area analysis (U.S. Census Bureau, 2023). This makes SAHIE particularly useful for studying trends across multiple demographic categories and over time.
Several features of the SAHIE dataset make it well-suited for temporal analysis:
It provides annual, consistently structured estimates from 2010 onward.
It includes demographic breakouts—Race, Age, and Sex—that are central to health equity research.
It releases confidence bounds that allow users to evaluate precision.
The motivation for developing the Explorer stems from the absence of publicly accessible tools that combine all these features in an interactive, user-driven environment. Existing dashboards rarely allow multi-dimensional filtering, simultaneous comparison of demographic subgroups, or direct visualization of uncertainty. This project fills that gap by integrating SAHIE’s data with modern visualization and analytic tools.


3. Goals and Research Questions
The overarching aim is to create a platform that allows users to explore patterns in health insurance coverage with clarity and flexibility. The project focuses on the following questions:
How have insured and uninsured percentages changed over time across Race, Age, and Sex categories? This includes identifying both gradual changes and sharper shifts during the COVID-19 period.
How does the level of uncertainty vary by demographic subgroup and state?
 SAHIE’s 90% confidence intervals help illustrate where estimates are more or less precise.
Are there identifiable inflection points or anomalies that align with major policy changes or economic disruptions?  Examples include the ACA, Medicaid expansion, and COVID-19–related continuous enrollment protections.
These questions tie directly to core themes in temporal data science, including the identification of trends, breakpoints, and disparities in high-dimensional time series datasets.

4. Dataset Description
The project uses the Census Small Area Health Insurance Estimates Time Series API, which returns annual modeled health insurance estimates for U.S. states and counties.
Data Source: Census Bureau Small Area Health Insurance Estimates (Timeseries API)
Data Scale
For a typical query spanning 2019–2022 at the state level with Race, Age, and Sex breakdowns, the combined dataset consists of several thousand records.
Key Variables
Geography: State FIPS and name
Temporal Component: Year (2019–2022 emphasized)
Demographics: Race category, age group, sex
Coverage Measures: Percent insured (PCTIC) and percent uninsured (PCTUI)
Uncertainty Bounds: Lower and upper 90% confidence intervals
SAHIE’s methodology relies on hierarchical modeling and multi-source inputs, resulting in more stable small-area estimates than survey data alone (Bauder et al., 2024). At the same time, the modeled nature of the data underscores the importance of communicating uncertainty clearly.

5. Clients and Use Cases
The Explorer is built with several audiences in mind:
Public health researchers, who examine disparities in access to coverage.
State health agencies, which track population-level insurance trends and design outreach efforts.
Policy analysts, who evaluate the effects of federal and state policies, particularly Medicaid expansion and ACA subsidy changes.
The tool’s multi-filter interface allows users to compare demographic subgroups and explore how different states experienced the COVID-19 period.

6. Methods and System Design
6.1 Technology Stack
The project uses:
Python for data processing.
Pandas for cleaning, merging, and preparing the dataset.
SAHIE Time Series API for data retrieval.
Streamlit for building the interactive dashboard.
Plotly for dynamic visualizations, including confidence-interval ribbons.
Environment variables for securely storing API keys.


6.2 Data Processing Pipeline
API Querying:  The application sends year-specific requests to the SAHIE API and returns JSON responses.
Preprocessing:  Steps include renaming variables, converting data types, mapping coded demographics to readable labels, and filtering out incomplete or overly aggregated observations.
Storage and Structuring: Cleaned data is stored in DataFrames grouped by demographic category.
Visualization: Plotly generates interactive line charts with confidence-interval shading.
Dashboard Rendering:  Streamlit renders visualizations and allows for interactive filtering and data downloads.
This pipeline enables a fully reproducible workflow for future updates as new SAHIE data releases become available.

7. Visualization 
The Explorer includes several visualization modes, each accessible through the dashboard sidebar. The visualization tool can be accessed in the following link: 
https://aaradhyaach-sahie-explorer.streamlit.app/


Temporal Trends
State- and national-level time series illustrate how insured percentages change year to year, which is particularly important for detecting COVID-era shifts.
Confidence Intervals
Shaded 90% confidence-interval ribbons highlight the precision of modeled estimates. Groups with smaller populations, such as American Indian/Alaska Native, typically have wider intervals.
Demographic Comparisons
Users can compare:
Race groups (e.g., Asian, Hispanic, Black, White)
Age groups (children, younger adults, older adults)
Sex (male/female)
Geographic Visualization
Selecting a state allows comparison between state-level and national trends.
Anomaly Detection
Sudden changes, such as the increases in insured rates following COVID-19 relief policies, can be identified visually.

8. Discussion
The Explorer demonstrates how combining modeled estimates, uncertainty visualization, and interactive filtering can support nuanced analysis of health insurance coverage trends. The tool captures the substantial rise in insurance coverage during the public health emergency, driven largely by continuous Medicaid enrollment requirements enacted under the Families First Coronavirus Response Act (ASPE, 2022). These gains were especially notable among historically underserved groups such as Hispanic and Black populations.
At the same time, several limitations must be acknowledged. SAHIE data are modeled rather than directly observed, and precision varies significantly by subgroup and geography. Additionally, while the Explorer provides visual evidence of COVID-era shifts, deeper statistical modeling (such as interrupted time-series analysis or difference-in-differences comparisons between Medicaid expansion and non-expansion states) would strengthen causal interpretation (Dague et al., 2023).

9. Analysis
The analysis centers on trends in Percent Insured between 2019 and 2022—a period that captures pre-pandemic conditions, the onset of COVID-19, and the continuation of emergency protections.


9.1 National Trends in Percent Insured (2019–2022)

Analysis of health insurance coverage from January 2019 to January 2022 shows the national insured rate increased slightly (from ~90% to ~91%), but significant racial and ethnic disparities persisted.


Key Findings:
Highest Coverage: The Asian (~93%) and White, non-Hispanic/Latino groups maintained the highest, most stable coverage.
Lowest Coverage & Largest Disparity: The Hispanic or Latino group consistently had the lowest rates (~77% to 80%), indicating an enduring coverage gap of 13–15 percentage points.
Most Notable Growth: The most significant percentage point increases were observed among the most underinsured groups, especially American Indian/Alaska Native and Hispanic or Latino, both showing accelerated growth after January 2021. This growth likely reflects the expanded eligibility and subsidies enacted through pandemic-era policies and the Affordable Care Act (ACA).
Anomalous Trend: The Native Hawaiian/Other Pacific Islander group registered a slight decline over the period, suggesting differential policy impact.
9.2 Racial and Ethnic Disparities During COVID
Analysis of Race-specific time series reveals persistent disparities, although all groups experienced increased coverage during COVID.
Hispanic and Black populations, historically more likely to be uninsured, showed the largest relative improvements between 2019 and 2021. This aligns with Census findings that pandemic-era Medicaid expansions disproportionately benefited communities of color (Keisler-Starkey & Bunch, 2022).
White and Asian groups maintained higher baseline insured percentages, with smaller but still positive increases.
The Explorer’s CI ribbons show that racial minority groups often have wider confidence intervals, reflecting SAHIE’s greater modeling uncertainty for smaller or harder-to-measure subpopulations.

9.3 Sex-Based Trends

Analysis by Sex reveals modest differences:
Females consistently exhibit slightly higher insured percentages compared to males, a pattern consistent with long-standing differences in employer-sponsored coverage and eligibility pathways (KFF, 2023).
COVID-era increases benefited both sexes, with males showing slightly larger absolute gains from 2019 to 2021.
9.4 State-Level Variation
State-specific analyses highlight substantial geographic heterogeneity, particularly between Medicaid expansion and non-expansion states. For example:
Expansion states such as Minnesota and New York show higher and more stable insured percentages throughout 2019–2022.
Non-expansion states (e.g., Texas, Florida) show lower baseline insured rates with smaller COVID-era improvements.
This reflects findings from NBER researchers documenting the significant role of Medicaid policy in shaping pandemic-era enrollment (Dague et al., 2023).
9.5 Confidence Interval Behavior and Model Precision
An important feature of the Explorer is the visualization of 90% confidence intervals, which reveal:
CI widths increase in smaller population subgroups (e.g., Native American, small states).
CI widths narrow during periods when administrative data become more informative, such as during increased Medicaid enrollment in 2020–2021.
This aligns with SAHIE technical reports noting that model precision fluctuates with administrative records availability and small-area poststratification (Bauder et al., 2024).

10. Conclusion
The SAHIE Demographics Explorer provides an accessible and analytically robust platform for examining changes in health insurance coverage across the United States. By integrating the SAHIE Time Series API with modern visualization tools, the project supports detailed exploration of demographic disparities, evaluates COVID-era policy impacts, and highlights areas where uncertainty is greatest. The tool is well positioned for continued refinement and expansion as new data becomes available.

