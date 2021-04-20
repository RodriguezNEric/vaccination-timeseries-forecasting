# Machine Learning Models to Predict COVID-19 Vaccinations in Massachusetts
- [Problem Statement](#Problem-Statement)
- [Executive Summary](#Executive-Summary)
- [project-5-part1.ipynb contents](#project-5-part1.ipynb-contents)
- [External Research](#External-Research)
- [Summary & Outlook](#Conclusions-&-Future Steps)
- [Data Sources](#Data-Sources)


## Problem Statement
COVID-19 is an emergent disease that forced societies all over the world to shut down, virtualy in every context. With the creation of vaccines to combat the virus, we've moved into the next change where societies will be allowed to reopen. The ability to predict the progress of COVID-19 vaccination campagins is crucial to governments in order to inform decision making regarding policy that aims to repoen society.<br>
<br>
The goals for this project was divided in two parts:<br>
(1) Build various time series forecasting models to predict the number of vaccines that will be administered over in a two week time period. Models will be evaluated using accuracy and MSE, with the best model being used for part two.<br>
(2) Develop a pipeline to extract, transform, and load raw vaccination data into the production model, and to then visualize the predictions in a streamlit application. Predictions are to be visualized at a county level.<br>

Citations
<br>
1: https://github.com/CSSEGISandData/COVID-19

---

## Data Dictionary
|Feature|Type|Description|
|---|---|---|
|**date**|datetime|date of the observation| 
|**total_vaccinations**|float|total number of doses administered. This is counted as a single dose, and may not equal the total number of people vaccinated, depending on the specific dose regime (e.g. people receive multiple doses). If a person receives one dose of the vaccine, this metric goes up by 1. If they receive a second dose, it goes up by 1 again| 
|**total_vaccinations_per_hundred**|float|total_vaccinations per 100 people in the total population of the state| 
|**daily_vaccinations_raw**|float|daily change in the total number of doses administered. It is only calculated for consecutive days. This is a raw measure provided for data checks and transparency, but we strongly recommend that any analysis on daily vaccination rates be conducted using daily_vaccinations instead| 
|**daily_vaccinations**|float|new doses administered per day (7-day smoothed). For countries that don't report data on a daily basis, we assume that doses changed equally on a daily basis over any periods in which no data was reported. This produces a complete series of daily figures, which is then averaged over a rolling 7-day window| 
|**daily_vaccinations_per_million**|integer|daily_vaccinations per 1,000,000 people in the total population of the state| 
|**people_vaccinated**|float|total number of people who received at least one vaccine dose. If a person receives the first dose of a 2-dose vaccine, this metric goes up by 1. If they receive the second dose, the metric stays the same| 
|**people_vaccinated_per_hundred**|integer|people_vaccinated per 100 people in the total population of the state| 
|**people_fully_vaccinated**|integer|total number of people who received all doses prescribed by the vaccination protocol. If a person receives the first dose of a 2-dose vaccine, this metric stays the same. If they receive the second dose, the metric goes up by 1| 
|**people_fully_vaccinated_per_hundred**|integer|people_fully_vaccinated per 100 people in the total population of the state| 
|**total_distributed**|integer|cumulative counts of COVID-19 vaccine doses recorded as shipped in CDC's Vaccine Tracking System| 
|**total_distributed_per_hundred**|integer|cumulative counts of COVID-19 vaccine doses recorded as shipped in CDC's Vaccine Tracking System per 100 people in the total population of the state| 
|**share_doses_used**|integer|share of vaccination doses administered among those recorded as shipped in CDC's Vaccine Tracking System| 

---

## EXECUTIVE SUMMARY
**Introduction**

**Methodology**

**Significant Findings**

**Conclusions and Recommendations**

A **data science workflow** was implemented to conduct this analysis. Firstly, the **problem statement** was defined—the JHU data needed to determine how to predict COVID-19 deaths. Next, **data scraping** was performed by locating credible sources that housed the appropriate datasets. Before beginning any analysis of the data, each individual dataset was imported to a **Pandas DataFrame**. Next, **data cleaning** was conducted to ensure that all datatypes were accurate and any other errors were fixed. Using all data from a six month period, **exploratory data analysis** was conducted to determine any parameters. Since the COVID-19 datasets contain data from all states, we narrowed our focus and selected Texas data for our analysis and used our findings to perform **data visualization**. Once all data was visualized and all statistical summaries were conveyed, **predictive statistical analysis** was conducted to describe what the distributions were and if any trends appeared in the data.  To confirm and support the observations made, **external research** about the COVID-19 and any other relevant data was conducted. Finally, well-informed **data-driven recommendations** for the COVID-19 were compiled. The most significant finding from our regression model was that confirmed cases and active cases have the largest impact on predicting COVID-19 related deaths.

**Work Flow Elements:**
- Imports and reading Data
- Exploratory data analysis
- Models testing: Null model, linear regression
- PCA
- Ridge modeling
- Lasso modelling
- Time series modeling
- Heat map
- Histograms
- Boxplots
- Scatterplots
- Software requirements: Pandas, Missingno, Sklearn, Matplotlib

### Code Contents:
- Description of the COVID-19 death data columns
- Data Import & Cleaning
- Exploratory Data Analysis
- Data Visualization
- Predictive Statistical analysis
- External Research
- Summary and Future outlook

---

## Summary & Outlook:
We found that the COVID-19 data is indeed important to predict a good score. we have build a regression model to predict the number of deaths caused by COVID-19 over a six-month timeframe. We have identified the variables that contribute the most predictive capabilities in determining COVID-19 deaths. We have used R-squared and RMSE to evaluate our production model.


**Future Outlook**

In order to achieve more reliable data-driven predictions, we will focus on the following in the future: multivariate time series analysis that incorporates the important variables identified in the regression model.

Citations
<br>
1: https://people.duke.edu/~rnau/411arim.htm
<br>
2: https://www.quantstart.com/articles/Autoregressive-Integrated-Moving-Average-ARIMA-p-d-q-Models-for-Time-Series-Analysis/

---
