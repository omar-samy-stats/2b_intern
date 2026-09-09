# Exploratory Data Analysis (EDA) — Task 1 Part 2 — Notebook Summary

## Note on dataset used
This notebook switches to a different dataset than the student-records cleaning exercise: `consumer_electronics_sales_data.csv`. This is the dataset chosen specifically because it fits the objectives of a company selling consumer electronics, and because it was already clean, it was the right dataset to carry forward into the machine learning and EDA stages of the project (data-cleaning skills were instead demonstrated separately using the deliberately messy student dataset).

## What the dataset contains
9,000 rows, 9 columns, with no missing values and no duplicate rows anywhere in the dataset:
- **ProductID** — a unique identifier per record, not used in the analysis (dropped from the analysis dataframe early on)
- **ProductCategory** — categorical: Smartphones, Smart Watches, Tablets, Sony, Headphones, Laptops (categories represented in the data)
- **ProductBrand** — categorical: Apple, Samsung, Sony, HP, Other Brands
- **ProductPrice** — continuous numeric, ranges roughly from about 100 to about 3000
- **CustomerAge** — numeric, ranges from 18 to 69
- **CustomerGender** — encoded numerically as 0 and 1 (binary)
- **PurchaseFrequency** — numeric, ranges from 1 to 19
- **CustomerSatisfaction** — numeric, ranges from 1 to 5
- **PurchaseIntent** — the target variable, binary (0 or 1)

Summary statistics (mean, std, quartiles) were pulled for all numeric columns, and the count row was reformatted to display as clean whole numbers instead of showing decimal points, purely for a more readable summary table.

## Phase 1: Data Quality Check
A explicit data-quality phase confirmed: zero null values across every column, zero duplicated rows, and CustomerGender confirmed to only ever take the values 0 or 1 (no unexpected categories). This phase exists to establish confidently that no further cleaning was needed before moving into analysis — a deliberate contrast with the messy student dataset used elsewhere in the project.

## Phase 2: Univariate Analysis

### Outlier check on numeric columns (boxplot)
A boxplot was drawn across all numeric columns (excluding ProductID and, separately, excluding ProductPrice for a cleaner scale) to visually scan for outliers. The chart shows CustomerAge with the widest spread (median around 43, box roughly spanning 30 to 56, whiskers reaching from about 18 to about 69, no points beyond the whiskers), PurchaseFrequency with a similarly proportioned spread (median 10, box from about 5 to 15), and CustomerGender, CustomerSatisfaction, and PurchaseIntent all compressed near the bottom of the chart since they occupy much smaller numeric ranges (0-1 or 1-5). Visually, none of the columns showed points sitting outside their whiskers, meaning no obvious extreme outliers by eye.

### Numeric outlier confirmation for ProductPrice (IQR method)
Since ProductPrice's scale made it hard to judge visually on the same chart as the other variables, the interquartile range (IQR) method was applied specifically to it, calculating the lower and upper bounds from Q1, Q3, and 1.5×IQR. This confirmed numerically, not just visually: zero outliers in ProductPrice. This mattered specifically because ProductPrice was flagged as a variable that could distort a linear model if it had extreme values — confirming there were none meant it was safe to keep this feature without needing to cap or transform it before modeling.

### Histograms and skewness
Skewness was calculated for all numeric columns, and all of them came out very close to zero (ProductPrice ≈ 0.029, CustomerAge ≈ 0.004, CustomerGender ≈ -0.036, PurchaseFrequency ≈ -0.001, CustomerSatisfaction ≈ 0.005, PurchaseIntent ≈ -0.268), indicating no meaningful skew in any of the core numeric features.

Histograms (with a KDE curve overlaid) were plotted individually for ProductPrice, CustomerAge, and PurchaseFrequency:
- **ProductPrice distribution:** essentially flat/uniform across its whole range (roughly 100 to 3000), with bar heights staying fairly consistent (mostly between 350-450 counts per bin) rather than peaking anywhere — no single price range dominates.
- **CustomerAge distribution:** also close to uniform across the 18-69 range, with bar heights fluctuating mildly (roughly 330-540 counts) but no strong concentration at any particular age.
- **PurchaseFrequency distribution:** likewise close to uniform across its 1-19 range, with counts per bin staying in a similar band (roughly 440-500) — the notebook's inline note describing this column as "binary just 0 and 1" does not match what the chart and the data range actually show (the column's real range is 1 to 19, not binary); the visual and computed skew confirm a flat, non-skewed distribution rather than a binary one.

## Target Variable Analysis (PurchaseIntent)
Before deeper analysis, a concern was raised about whether encoding PurchaseIntent as 1 (intent) vs. 0 (no intent) could bias a model toward favoring the "1" class. Checking the class balance showed PurchaseIntent = 1 at about 56.6% and PurchaseIntent = 0 at about 43.4% of rows — close enough to balanced that this was concluded not to be a meaningful problem for the machine learning model.

## Bivariate Analysis — the core question
The guiding question for this whole section was framed explicitly: does a feature's value genuinely differ between customers who show purchase intent and those who don't — this is what determines which columns are good candidate predictors for the machine learning stage that follows.

### Correlation heatmap
A correlation matrix (excluding ProductID) was computed and visualized as a heatmap. Key readings: CustomerAge correlates with PurchaseIntent at about 0.29, CustomerGender correlates with PurchaseIntent at about 0.50 (the strongest relationship in the whole matrix), and CustomerSatisfaction correlates with PurchaseIntent at about 0.39. ProductPrice and PurchaseFrequency both show negligible correlation with PurchaseIntent (near zero, around -0.018 and -0.001 respectively). Critically, none of the predictor variables show strong correlation with each other — the heatmap was specifically checked for this reason, to confirm the predictor set does not suffer from multicollinearity, which is an assumption that matters for certain ML models.

### Pair plot
A full pairplot (scatter plots for every pair of numeric variables, with histograms on the diagonal) was generated across all numeric columns. It visually reinforced the heatmap's findings — no visible linear relationship between Purchase Frequency and Product Price, or between Product Price and Customer Age; the scatter clouds for these pairs look like unstructured, evenly spread point clouds rather than showing any trend.

### T-tests on numeric columns vs. PurchaseIntent
An independent-samples t-test was run for each numeric column, comparing the group with PurchaseIntent = 0 against the group with PurchaseIntent = 1:
- **ProductPrice:** t = 1.661, p = 0.0967 — not statistically significant (no meaningful difference between the two groups)
- **CustomerAge:** t = -28.722, p ≈ 0.0000 — strongly significant
- **PurchaseFrequency:** t = 0.139, p = 0.8895 — not statistically significant
- **CustomerSatisfaction:** t = -40.314, p ≈ 0.0000 — strongly significant

Conclusion drawn: CustomerAge and CustomerSatisfaction are meaningfully different between customers with and without purchase intent, making them strong candidate predictors, while PurchaseFrequency and ProductPrice show no significant difference between the groups, making them weak candidates.

### Chi-square tests on categorical columns vs. PurchaseIntent
A chi-square test of independence was run for ProductCategory, ProductBrand, and CustomerGender against PurchaseIntent:
- **ProductCategory:** chi2 = 6.932, p = 0.1395 — not significant
- **ProductBrand:** chi2 = 7.259, p = 0.1228 — not significant
- **CustomerGender:** chi2 = 2287.851, p ≈ 0.0000 — extremely significant

Conclusion: CustomerGender has a strong, statistically significant relationship with PurchaseIntent (consistent with its high 0.50 correlation from the heatmap), making it a strong categorical predictor, while ProductCategory and ProductBrand show no significant relationship with purchase intent on their own.

### Categorical relationships — brand vs. category crosstab
A crosstab of ProductCategory against ProductBrand was produced to see how purchase counts are distributed. Counts were fairly evenly spread across all brand/category combinations (roughly 336-391 per cell), without dramatic dominance by any single combination.

### Top purchased category per brand (bar chart)
For each brand, the single most-purchased category was identified and charted. Results: Apple's top category is Tablets, HP's top category is Laptops, Other Brands' top category is Laptops, Samsung's top category is Laptops, and Sony's top category is Smartphones. So Laptops is the top category for three of the five brands (HP, Other Brands, Samsung), Apple stands out with Tablets as its top seller, and Sony stands out with Smartphones as its top seller — worth noting precisely since the inline note in the notebook describes Laptops as dominant "in most brands" with Apple as the exception, which is accurate for HP/Other Brands/Samsung/Apple, but Sony's top category (Smartphones) is a second exception not mentioned in that note.

### Customer satisfaction by brand
Records were sorted by CustomerSatisfaction (highest first) and the brand counts among top-satisfaction entries were tallied: Samsung led with 1,854, followed by HP (1,820), Sony (1,790), Other Brands (1,776), and Apple (1,760). This was interpreted as a signal that Samsung products draw the strongest customer satisfaction, offered as a soft business insight that the company could potentially lean into Samsung-related offerings to help drive sales.

### Average customer age by product category (line chart)
Average CustomerAge was computed per ProductCategory and sorted ascending, then plotted as a line chart. The order from youngest to oldest average customer age: Laptops (≈42.6), Headphones (≈43.2), Smartphones (≈43.3), Tablets (≈43.5), Smart Watches (≈44.0). The chart shows a gently rising line with no dramatic jumps — all five categories cluster tightly in the low-to-mid 40s average age range. The insight drawn: since most purchasers across all categories fall around their early 40s, marketing/advertising efforts could reasonably be tailored toward that age group rather than needing wildly different targeting per product category.

### Category-level averages summary table
A final grouped summary table was produced showing the mean of every numeric column broken down by ProductCategory (average price, average age, average gender encoding, average purchase frequency, average satisfaction, average purchase intent per category). This table shows the differences between categories are all fairly subtle — e.g., average PurchaseIntent ranges narrowly from about 0.547 (Laptops) to about 0.583 (Smartphones) across categories, reinforcing that ProductCategory itself isn't a strong differentiator of purchase behavior on its own, consistent with its non-significant chi-square result above.

## Overall conclusions carried forward from this notebook
This EDA stage's central output is a shortlist of which features are meaningful predictors of PurchaseIntent for the machine learning stage that follows: **CustomerAge, CustomerSatisfaction, and CustomerGender** emerged as statistically meaningful (via t-tests, chi-square, and correlation), while **ProductPrice, PurchaseFrequency, ProductCategory, and ProductBrand** did not show significant relationships with PurchaseIntent and were treated as weaker candidates. The heatmap and pairplot together also confirmed no problematic multicollinearity among the numeric predictors, which was an important check before moving into modeling.
