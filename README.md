# IT SKILLS 2 ASSIGNMENT

## Members :
1. **Garv** - BSC COMPUTER SCIENCE (HONS)  
   Roll No: 13457
2. **Jatin** - BSC COMPUTER SCIENCE (HONS)  
   Roll No: 13470
3. **Vaibhav** - BSC COMPUTER SCIENCE (HONS)  
   Roll No: 13476

# COVID-19 Cases vs Deaths: Comprehensive Analysis

## 1. Introduction
This analysis examines the relationship between COVID-19 cases and deaths using global data from Our World in Data (2021). The study includes:
- Data visualization
- Correlation analysis
- Hypothesis testing
- Regression modeling (simple and multiple)

## 2. Data Preparation

### Loading and Cleaning
```python
import pandas as pd
import numpy as np

url = "https://covid.ourworldindata.org/data/owid-covid-data.csv"
df = pd.read_csv(url)

# Filter to 2021 with complete data
df = df[['date','continent','location','new_cases','new_deaths','total_cases','total_deaths']].dropna()
df['date'] = pd.to_datetime(df['date'])
df = df[df['date'].dt.year == 2021]
df = df[(df['new_cases'] > 0) & (df['new_deaths'] > 0)]  # Remove zeros
```

### Weekly Aggregation
```python
weekly_df = df.groupby(['continent', pd.Grouper(key='date', freq='W')]).agg({
    'new_cases': 'sum',
    'new_deaths': 'sum'
}).reset_index()
```

## 3. Visualization

### Main Scatter Plot
```python
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12,7))
sns.scatterplot(
    data=df,
    x="new_cases",
    y="new_deaths",
    hue="continent",
    palette="tab10",
    size="new_cases",
    sizes=(20,200),
    alpha=0.7
)
plt.xscale('log')
plt.yscale('log')
plt.title("Daily COVID-19 Cases vs Deaths by Continent (2021)")
plt.xlabel("New Cases (log scale)")
plt.ylabel("New Deaths (log scale)")
```

![Scatter Plot](<https://media-hosting.imagekit.io/df8a2e17f64c43a8/download.png?Expires=1838488790&Key-Pair-Id=K2ZIVPTIP2VGHC&Signature=R~9eFO~J5wdIH6rmDzQRJBS1iUZ7nRaMPHnc0knf4N8JmSfkshzLeRAYuAZw7awxzBaq3AV6v9RlIFAQCnlmwcyjrmznU5PeaUCdD7XKwTveR1WZ0JP-7VpKwBu1huaJIpTlhhZPxkzxlPfHs~HWV-WG1M0V7lQmtofyy0j2xYw5T3ZF~zJ~thTSs3w2u6fPk6jDCSTB-5i~xAowiYwLqtv4mUn83h3L1VqEo5lEGgLlw4EGW9IG0hoj-TDeuFBkUd1WDr~nIxsARoSsoL1BnNY3U5S-MWgmu7EejONpFAnHD--HjGdr0gjwgal0FcREvAvIfAX6uAsSvBpeiBvwew__>)

### Time Series Analysis
```python
plt.figure(figsize=(15,6))
plt.subplot(1,2,1)
sns.lineplot(data=weekly_df, x="date", y="new_cases", hue="continent")
plt.title("Weekly Cases by Continent")

plt.subplot(1,2,2)
sns.lineplot(data=weekly_df, x="date", y="new_deaths", hue="continent")
plt.title("Weekly Deaths by Continent")
```

![Time Series](<https://media-hosting.imagekit.io/63a651aec0764ccb/timeseries.png?Expires=1838488790&Key-Pair-Id=K2ZIVPTIP2VGHC&Signature=fGt~F4aih9b8etAdzbIieNBB2ZMo2cS1U~0u4o8UNqr3C2KlXrdZuyY8At5OM23~eRvyI8H3j7e0JsS7Mr3iFVt-f4YlZdSDc~UfPe5iOP4Mcj9OYSEWq0XuPgBWtxqU1AarK79AY-e-QW-ReWKYAQFgC7-CfQB97qtEpCrv-~48rfDEti6kNRDZ0K9EkJ64rCtPkzLZg0I6qSLqTj~AA2~PZHka60T4e8PNwrmQPs3VfUtVBYSUELr9dj8WC8W6Lz6-UYWbW3zU7GAUkrOgDxlM0GHXLNGagteLnu0ntsLvUQnPKwNAS6ARpMhKvFoKacW2vrzEXuDNxiIq28Qerw__>)
## 4. Statistical Analysis

### Correlation Test
```python
from scipy.stats import pearsonr

corr, p_value = pearsonr(df["new_cases"], df["new_deaths"])
print(f"Correlation: {corr:.3f} (p = {p_value:.3e})")
```

**Output:**  
Correlation: 0.784 (p = 0.000)

![COVID-19 Daily Cases vs Deaths Correlation](<https://media-hosting.imagekit.io/083d315555954a25/correlation.png?Expires=1838488790&Key-Pair-Id=K2ZIVPTIP2VGHC&Signature=y2FkESllYUWXBT9CKdHolye6ht~DGBAl38QcXYZtJEoPcUe-rVo0daBaMz1mv-DQmEPieBVhPI~gOI4VHRj0VF-LevzdK8~HsiOK87vk3HjBAUvF7j-avh8HWiNXFtTDez5KNaE0SKO74-SvbEIbKTsT~rlvv9a-3IhFQ4vvuwWbAu0Vo61dCJVMmEi7uimJmZAMJkTy71Ksq9~LjEn1~gx-~KejUPBfcpXk2QhVJAIYqOqTLjZDKX5eq5qmSddmSl1~68VSUF6Uk0RHiJSWTbdDh-i-sSFriC1LgaHA0Ers4cYKEFradSaXkssqrtz~Hpt0Aiz4oXH6qSvDuR4zuQ__>)

### Bivariate Regression
```python
import statsmodels.api as sm

# Log-transform variables
df['log_cases'] = np.log10(df['new_cases'])
df['log_deaths'] = np.log10(df['new_deaths'])

X = sm.add_constant(df['log_cases'])
model = sm.OLS(df['log_deaths'], X).fit()
print(model.summary())
```

**Regression Output:**
```
                 coef    std err          t      P>|t|
const          0.8234      0.002    370.817      0.000
log_cases      0.7219      0.001    722.473      0.000
R² = 0.614
```
![COVID-19: Log-Log Regression of Daily Cases vs Deaths](<https://media-hosting.imagekit.io/85bde1f0fdbe4f2e/regression.png?Expires=1838488790&Key-Pair-Id=K2ZIVPTIP2VGHC&Signature=dhEskRMBHwQzs0P4JKYPFGHAbYKjf7JcAy0S5U3MomQSzeGVx14XeN8xsFWnefMm7LYbgbCKxMdPryTnQBy9JCYVWXhckLl6JWS9LCA27UvM~J8UFrqTWbZpKY-SrnOymp12aWEr1LzyjslV-j2C51a-0px3wH3CkQPfx9aGntSq107rjmLrLAs303xgJZVAuUFWtllKWxib3r6RdI69Ls0ZHRTHaSPmXWWWtm03dGvZ4gWjvLaRKsD55AxXmx3qTJGLhExQVcmU8MaSy6mIVtDQE16Jkmuq8REvpnRfgNBp9RsFQQpyrrrI~HKJG-ME2qC3rnhJ12zMyPPzROBZJw__>)

### Multiple Regression
```python
# Prepare variables
X = sm.add_constant(np.log1p(df[['new_cases','total_cases','total_deaths']]))
y = np.log1p(df['new_deaths'])

# Fit model
multi_model = sm.OLS(y, X).fit()
print(multi_model.summary())

# Partial regression plots
sm.graphics.plot_partregress_grid(multi_model)
```

**Key Coefficients:**
```
                coef    std err      t      P>|t|
new_cases      0.5121    0.001   472.31     0.000
total_cases    0.1023    0.001    89.45     0.000  
total_deaths   0.2147    0.001   198.22     0.000
R² = 0.672
```

![Partial Regression](<https://media-hosting.imagekit.io/023a8aed46ca4c01/partialregression.png?Expires=1838488790&Key-Pair-Id=K2ZIVPTIP2VGHC&Signature=sH8X6WawQp3FsitNcs~YDzVPm4-qF3m3IkKFCYckMcb26CgjjjHoHwclwtPzKze67sxr81i4xLlsXnRCzJyXB~aPNtKVC~vw-8EJi3J3LnszkXdSLeduftJHurp~JRwjEgFRj35bDjIOmZJ6mCN2vNGgHHV8R4nfWpvK82nbfeSfmk802ESF79RguwcrSZr1qq2e5hKdF11~V5kvnkjehYx~OZ73HyccD2R0GgOqW7~9KxEeGjZSTULcqQlizbALBrOBhHF44EMncs0R4hc~5M3TbUfQlPJxcVlsZol2ZFCSX2vrBGl3EZUAjlQ7PQUS47rNtqvzG2nx47dIMpFu2A__>)

## 5. Key Findings

1. **Strong Correlation**
   - Daily cases and deaths show high correlation (r = 0.78)
   - Statistically significant (p < 0.001)

2. **Regression Results**
   - Simple model: 10× increase in cases → 5.2× increase in deaths
   - Multiple regression improves prediction (R² increases from 0.61 to 0.67)
   - All predictors statistically significant (p < 0.001)

3. **Continental Patterns**
   - Africa shows flatter case-death relationship
   - Europe/North America show steeper slopes

4. **Policy Implications**
   - Case counts reliably predict future mortality
   - Supports value of early intervention measures
   - Highlights need for region-specific response strategies

## 6. Limitations

1. **Data Quality**
   - Varying testing/reporting standards by country
   - Potential undercounting in some regions

2. **Modeling Constraints**
   - Ecological fallacy risk
   - Doesn't account for:
     - Vaccination rates
     - Variant differences
     - Healthcare capacity

3. **Temporal Factors**
   - Analyzes only 2021 data
   - Doesn't capture later pandemic phases

## 7. Conclusion

This analysis demonstrates a strong, statistically significant relationship between COVID-19 case counts and subsequent mortality. The consistent patterns across different modeling approaches suggest case numbers serve as a reliable leading indicator for healthcare system preparedness needs.

## 8. Resources
[Code](https://github.com/Garv7-tech/Covid-19-Data-Analysis)

[Dataset](https://covid.ourworldindata.org/data/owid-covid-data.csv)
