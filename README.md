# Machine Learning Models to Predict COVID-19 Vaccinations in Massachusetts
- [Problem Statement](#Problem-Statement)
- [Data Dictionary](#Data-Dictionary)
- [Executive Summary](#Executive-Summary)
- [Future Outlook](#Future-Outlook)
- [Citations](#Citations)
- [Data Sources](#Data-Sources)

---

## Problem Statement
COVID-19 is an emergent disease that forced societies all over the world to shut down, in virtually every context<sup>1</sup>. With the creation of vaccines to combat the virus, we've moved into the next stage where societies will be allowed to reopen. The ability to predict the progress of COVID-19 vaccination campagins is crucial to governments in order to inform decision making regarding policy that aims to repoen society.<br>
<br>
The goals for this project was divided in two parts:<br>
(1) Utilize recurrent neural networks to build various time series forecasting models that predict the number of vaccines that will be administered over a two week time period in Massachusetts, USA. Models will be evaluated using accuracy and MSE, with the best model being used for part two.<br>
(2) Develop a pipeline to extract, transform, and load raw vaccination data into the production model, and to then visualize the predictions in a streamlit application. Predictions are to be visualized at a county level.<br>

---

## Data Dictionary
|Feature|Type|Description|
|---|---|---|
|**date**|datetime|date of the observation| 
|**total_vaccinations**|float|total number of doses administered. This is counted as a single dose, and may not equal the total number of people vaccinated, depending on the specific dose regime (e.g. people receive multiple doses). If a person receives one dose of the vaccine, this metric goes up by 1. If they receive a second dose, it goes up by 1 again| 
|**total_vaccinations_per_hundred**|float|total_vaccinations per 100 people in the total population of the state| 
|**daily_vaccinations_raw**|float|daily change in the total number of doses administered. It is only calculated for consecutive days. This is a raw measure provided for data checks and transparency, but we strongly recommend that any analysis on daily vaccination rates be conducted using daily_vaccinations instead| 
|**daily_vaccinations**|float|new doses administered per day (7-day smoothed). For countries that don't report data on a daily basis, we assume that doses changed equally on a daily basis over any periods in which no data was reported. This produces a complete series of daily figures, which is then averaged over a rolling 7-day window| 
|**daily_vaccinations_per_million**|float|daily_vaccinations per 1,000,000 people in the total population of the state| 
|**people_vaccinated**|float|total number of people who received at least one vaccine dose. If a person receives the first dose of a 2-dose vaccine, this metric goes up by 1. If they receive the second dose, the metric stays the same| 
|**people_vaccinated_per_hundred**|float|people_vaccinated per 100 people in the total population of the state| 
|**people_fully_vaccinated**|float|total number of people who received all doses prescribed by the vaccination protocol. If a person receives the first dose of a 2-dose vaccine, this metric stays the same. If they receive the second dose, the metric goes up by 1| 
|**people_fully_vaccinated_per_hundred**|float|people_fully_vaccinated per 100 people in the total population of the state| 
|**total_distributed**|float|cumulative counts of COVID-19 vaccine doses recorded as shipped in CDC's Vaccine Tracking System| 
|**total_distributed_per_hundred**|float|cumulative counts of COVID-19 vaccine doses recorded as shipped in CDC's Vaccine Tracking System per 100 people in the total population of the state| 
|**share_doses_used**|float|share of vaccination doses administered among those recorded as shipped in CDC's Vaccine Tracking System| 

---

## EXECUTIVE SUMMARY
**Introduction** <br>
The COVID-19 pandemic prevails as an ultimatum to the global economic growth and the wellbeing of society. The global spread of COVID-19 is increasing day by day, creating a larger risk of disease or death as well as a strain on the economy<sup>1</sup>. As of April 18, 2021, there have been 31.7 million cases of COVID-19 in the united states, and 567,000 deaths<sup>1</sup>. As of April 18, 2021, three vaccines had received emergency authorization from the FDA to be used at a population wide level to prevent the contraction and spread of COVID-19 (although the Johnson & Johnson vaccine has been halted in order to investigate severe adverse outcomes)<sup>2</sup>. As of April 18, 2021, 25% of US adults are fully vaccinated. This is significant because the key to returning back to “normal” life is ensuring that the vast majority of the population is vaccinated and has resistance to the virus<sup>2</sup>.<br>
As of April 18, 2021, about 30% of US adults in Massachussets are fully vaccinated<sup>3</sup>. The state of Massachusetts is in Phase 4 of their reopening plan, which was only reached once there was a vaccine for COVID-19. Although this stage is called the "new normal", there are still a multitude of restrictions on businesses and individuals in the state that prevent life from feeling normal, and there has been little guidance from the state as to when they believe industries will reopen. The better the state of Massachusetts is able to forecast vacinations rates, the better they will be able to inform businesses and individuals when the state will continue to reopen, and in turn, the aforementioned parties will be able to plan accordingly. 

**Methodology**<br>
A data science workflow was implemented to conduct this analysis. Firstly, the problem statement was defined. Next, data scraping was performed by locating a credible dataset. Before beginning any analysis of the data, the  dataset was imported to a Pandas DataFrame. Next, data cleaning was conducted to ensure that all datatypes were accurate and to address any missing data. Using data that only pertained to the state of Massachussets, exploratory data analysis was conducted to evaluate trend, seasonality, and autocorrelation. Once all data was visualized and all statistical summaries were conveyed, recurrent neural networks were built using SimpleRNN, GRU, and LSTM architecture. Once the best models were found with each respective archictecture, the target variable was predicted and compared with the actual dataset. Models were evaluated on prediction accuracy and mean-squared-error (MSE). Finally, well-informed data-driven recommendations for the COVID-19 vaccine predictions were compiled. 

**Significant Findings**<br>
- There 6 rows of missing data in the first month of the dataset
- Daily vaccination data is non-stationary, but is stationary when differenced once.
- Auto-correlation plot confirms positive trend for daily vaccinations feature
- RNNs on differenced data returned higher MSE scores
- RNN that returned the lowest MSE was the SimpleRNN

**Conclusions and Recommendations**<br>
The SimpleRNN architechture yielded the lowest MSE. All models largely overfit the data. When prediction accuracy was visualized, no model returned an accuracy higher than 7%. This is either due to an error during pre-processing, or because there isn't enough data. The models need to be finetuned before they can be used to accurately predict two-weeks worth of COVID-19 vaccination data in Massachusetts.

Note - when notebook was re-run, all the values stayed the same besdies the SimpleRNN model. Data will need to be re-evaluated to ensure the SimpleRNN is truly the best model.

---

## Future Outlook
<br>
In order to achieve an RNN model that is able to predict COVID-19 vaccination data in Massachussets with over 80% accuracy, more data is needed. The GRU and LSTM architecture hyperparameters will be continued to be tuned in order to yield data that is less over-fit with high predicition accuracy. Once a model can accurately predict COVID-19 vaccinations, an application will be built that visualizes these predictions at the county level in Massachusetts.

---

## Citations
<br>
1: https://covid.cdc.gov/covid-data-tracker/#datatracker-home
<br>
2: https://www.fda.gov/vaccines-blood-biologics/vaccines/emergency-use-authorization-vaccines-explained
<br>
3: https://www.mass.gov/info-details/reopening-massachusetts
<br>

---

## Data Sources
Data was taken from the "Our World in Data" (OWID) repository in GitHub. OWID pulls data from the CDC and from Johns Hopkins University.
- https://github.com/owid/covid-19-data