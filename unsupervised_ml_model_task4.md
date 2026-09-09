# Unsupervised ML Model — Task 4 — Market Segmentation — Notebook Summary

## Purpose of this notebook
This notebook builds a customer market segmentation model using unsupervised clustering on the same consumer electronics sales dataset used in the supervised modeling stage. Unlike the earlier supervised task (which predicted PurchaseIntent using labeled data), this stage has no target label — the goal instead is to discover natural groupings of customers based on their behavioral characteristics, which could later inform marketing or business strategy (e.g., targeting different customer segments differently).

## Preprocessing

### Feature selection for clustering
Only four continuous numeric columns were selected for clustering: `CustomerAge`, `ProductPrice`, `PurchaseFrequency`, and `CustomerSatisfaction`. The reasoning documented directly in the notebook: clustering algorithms like K-Means rely on measuring distances between data points, which only makes sense for continuous variables where numeric distance is meaningful. Categorical columns (like ProductCategory or ProductBrand) were deliberately excluded from the clustering itself, because including them would distort the distance calculations the algorithm depends on. CustomerGender (binary-encoded) and PurchaseIntent were also excluded from the clustering features — PurchaseIntent because it's the target variable from the supervised stage and not meant to be an input here, and CustomerGender because it's not a continuous distance-based variable in the same sense as the other four.

### Standardization
All four selected features were scaled using StandardScaler (mean 0, standard deviation 1) before clustering. This step matters more for clustering than it did for the supervised models, because K-Means measures raw distances between points — without scaling, a feature with a naturally larger numeric range (like ProductPrice, ranging up to ~3000) would dominate the distance calculation and drown out the influence of smaller-range features (like CustomerSatisfaction, ranging only 1-5).

## Choosing the number of clusters (K) for K-Means

### Elbow Method
Inertia (within-cluster sum of squared distances — a measure of how tightly points cluster around their centroid) was calculated for K ranging from 1 to 14, and plotted. The elbow chart shows inertia dropping steeply from about 36,000 at K=1 down to around 19,000 by K=5, then continuing to decrease more gradually and smoothly through the remaining K values down to around 9,700 at K=14, without a single sharp, obvious "elbow" bend — the curve bends and flattens gradually rather than showing one clear corner. This means the elbow method alone did not give a strongly decisive answer for the best K in this case.

### Silhouette Score
Because the elbow method wasn't decisive, silhouette scores (a metric that measures how well-separated and internally cohesive clusters are, ranging from -1 to 1, higher is better) were calculated for K from 2 to 14 as a second, complementary check. The silhouette chart shows a low point around K=3 (roughly 0.177), a generally rising trend through K=4 to K=8, peaking clearly at **K=8 with a silhouette score of about 0.225** — the highest point on the chart — before dipping slightly and fluctuating in a lower, flatter band (roughly 0.215-0.221) for K=9 through K=14. Because K=8 produced the clear maximum, it was chosen as the final number of clusters for K-Means.

### Final K-Means silhouette score
With K=8 finalized, the model was re-fit and the final silhouette score was confirmed at **≈0.225**. The notebook's own interpretation of this number, written directly as a markdown note: a silhouette score of 0.22 indicates relatively weak cluster separation, meaning the eight identified customer segments overlap considerably and are not highly distinct from one another — an honest acknowledgment that while K=8 was the best available choice among the tested options, the resulting clusters are only moderately well-separated in an absolute sense, not sharply distinct.

## K-Means cluster profiles (K=8)
The mean value of each of the four clustering features was computed per cluster (clusters renumbered 1-8 for readability). The eight resulting profiles:

| Cluster | Avg Age | Avg Price | Avg Purchase Frequency | Avg Satisfaction |
|---|---|---|---|---|
| 1 | ≈32.9 | ≈803 | ≈5.6 | ≈1.93 |
| 2 | ≈30.3 | ≈948 | ≈14.3 | ≈4.10 |
| 3 | ≈54.5 | ≈2200 | ≈14.6 | ≈4.11 |
| 4 | ≈54.7 | ≈861 | ≈6.6 | ≈4.27 |
| 5 | ≈31.1 | ≈2218 | ≈5.9 | ≈3.97 |
| 6 | ≈55.0 | ≈930 | ≈14.2 | ≈1.95 |
| 7 | ≈56.0 | ≈2142 | ≈5.4 | ≈2.03 |
| 8 | ≈31.6 | ≈2168 | ≈14.1 | ≈1.77 |

Reading across these profiles, the eight clusters roughly correspond to combinations of "younger vs older customer," "cheaper vs pricier products," "low vs high purchase frequency," and "low vs high satisfaction" — for example, Cluster 1 represents younger customers buying cheaper products infrequently with low satisfaction, while Cluster 3 represents older customers buying expensive products frequently with high satisfaction. The clusters effectively span most combinations of these four dimensions, which is consistent with the moderate (not strong) silhouette score — the underlying customer base doesn't split into a small number of sharply distinct groups, but rather varies fairly continuously across these dimensions, and K=8 approximates that continuous variation with eight representative segments.

## Agglomerative (Hierarchical) Clustering — second algorithm for comparison
Agglomerative clustering was also run, using the same K=8 (chosen to match K-Means for a fair, direct comparison), on the same scaled feature set.

### Agglomerative silhouette score across K values
A silhouette score sweep (K=2 to 14) was run for Agglomerative clustering as well. The chart shows a very different shape from K-Means' silhouette chart: it starts around 0.146 at K=2, drops to a clear low point around K=3 (roughly 0.123), then rises with some fluctuation, reaching its own peak around K=8 to K=10 (values roughly 0.147-0.150), before gradually declining again toward K=14 (down to about 0.136). Similarly to K-Means, K=8 lands close to the peak region for Agglomerative clustering too, though the peak here is comparatively flatter/broader than K-Means' more distinct single peak.

### Final Agglomerative silhouette score
At K=8, Agglomerative clustering's silhouette score came out to **≈0.150** — noticeably lower than K-Means' ≈0.225 at the same K. This is a meaningful comparison point: **K-Means produced better-separated, more cohesive clusters than Agglomerative clustering did on this dataset**, at least at the K=8 setting used for both.

### Agglomerative cluster profiles
The mean feature values per Agglomerative cluster (8 clusters, numbered 0-7) showed a broadly similar spread of customer types as K-Means (combinations of younger/older, cheaper/pricier, low/high frequency, low/high satisfaction), but with different specific groupings and boundaries — e.g., Agglomerative's cluster 4 (young, mid-price ≈1413, high frequency ≈16.3, high satisfaction ≈4.31) doesn't map one-to-one onto any single K-Means cluster, reflecting that the two algorithms drew the segment boundaries somewhat differently even though both were given the same K and the same input features.

## PCA visualization of the clusters
A 2-dimensional PCA projection (reducing the four clustering features down to two principal components, PC1 and PC2) was used to visualize the K-Means clusters on a scatter plot, with each of the 9,000 points colored by its assigned cluster (viridis color scale, purple through yellow). 

**What the chart shows:** the overall point cloud forms one large, roughly circular/oval mass without clearly separated "islands" of color — the different cluster colors blend into each other at their boundaries rather than forming visually distinct, isolated groups. That said, there is a visible general structure: points colored toward the purple/dark end concentrate more on the left side of the plot (negative PC1), points colored toward the blue end concentrate more on the right side (positive PC1), and green/yellow-colored points sit more toward the lower-middle area — meaning PC1 (the horizontal axis) captures the primary direction along which the clusters differ from each other, while separation along PC2 (vertical) is much less pronounced. This visual pattern — real but overlapping structure, with no sharply isolated cluster islands — is consistent with, and visually confirms, the moderate silhouette score (≈0.225): the clusters are genuinely there and organized along a real underlying gradient in the data, but they blend into their neighbors rather than forming cleanly separated groups. Note: the code cell that generated this plot was later commented out in the notebook (left as reference/inactive code), but its output was preserved from an earlier run and is described here based on that preserved chart.

## Additional exploration noted but not completed in this notebook
A commented-out section outlines a planned next step: profiling each K-Means cluster using the categorical columns that were excluded from the clustering itself (CustomerGender, ProductBrand, ProductCategory) — checking the gender balance per cluster and the distribution of brands/categories per cluster purely for interpretation purposes (understanding who is in each segment), not for the clustering algorithm itself. This step was planned but not executed/run in this version of the notebook.

## Preparation for DBSCAN (K-Distance Graph)
The notebook's final step computes and plots a K-distance graph in preparation for running DBSCAN as a third clustering algorithm (DBSCAN requires an `eps` parameter — the maximum distance for points to be considered neighbors — and the K-distance graph is the standard way to estimate a reasonable value for it). For each point, the distance to its 5th nearest neighbor was calculated, then all these distances were sorted in ascending order and plotted.

**What the chart shows:** the 5th-nearest-neighbor distance rises slowly and steadily from about 0.1 up to about 0.3-0.35 across the bulk of the data points (roughly the first 8,000 of 9,000 points), before curving upward more sharply in the final stretch, reaching about 0.58 at the very last points. The "elbow" in this kind of chart is normally used to pick DBSCAN's `eps` value — the point where the curve begins bending upward more steeply typically marks a reasonable cutoff. In this chart, that upward bend starts becoming more pronounced somewhere around the 8,000-8,500 point mark (roughly where distance crosses ≈0.35-0.4), suggesting an eps value in roughly that range could be a reasonable starting point, though the curve's gradual, gentle slope for most of its length (rather than one sharp corner) means this choice involves some judgment rather than being perfectly clear-cut, similar to the ambiguity noted in the earlier K-Means elbow chart.

**Status:** DBSCAN was subsequently run using this `eps` estimate — results below.

## DBSCAN — third clustering algorithm, run and evaluated
DBSCAN was run using `eps=0.35` (based on the K-distance graph estimate above) and `min_samples=5` (matching the 5-neighbor count used to build that graph), on the same scaled four-feature set (CustomerAge, ProductPrice, PurchaseFrequency, CustomerSatisfaction) used for K-Means and Agglomerative clustering.

**Results:**
- **Number of clusters found: 12**
- **Number of noise points: 177** (points DBSCAN could not assign to any cluster, labeled -1)
- **Silhouette Score: ≈-0.215** (negative)

**What a negative silhouette score means:** unlike K-Means' ≈0.225 and Agglomerative's ≈0.150 (both positive, meaning points are on average closer to their own cluster than to neighboring clusters), a negative score means the opposite — on average, points ended up closer to a *different* cluster than the one they were actually assigned to. This is a clear, meaningful sign that DBSCAN's clustering structure does not fit this data well at the settings used.

**Cluster profiles (mean feature values per DBSCAN cluster, including noise as cluster -1):**

| Cluster | Avg Age | Avg Price | Avg Purchase Frequency | Avg Satisfaction |
|---|---|---|---|---|
| -1 (noise) | 44.99 | 1595.09 | 10.39 | 2.87 |
| 0 | 42.95 | 1506.18 | 9.98 | 1.00 |
| 1 | 43.04 | 1508.04 | 9.89 | 2.00 |
| 2 | 43.94 | 1560.85 | 9.92 | 3.00 |
| 3 | 43.45 | 1516.39 | 10.25 | 5.00 |
| 4 | 43.08 | 1534.14 | 10.20 | 4.00 |
| 5 | 25.00 | 304.36 | 17.78 | 3.00 |
| 6 | 44.25 | 1168.24 | 19.00 | 2.00 |
| 7 | 32.50 | 888.10 | 1.00 | 1.00 |
| 8 | 65.60 | 2838.84 | 1.00 | 5.00 |
| 9 | 43.40 | 2843.54 | 18.60 | 1.00 |
| 10 | 65.29 | 2889.35 | 9.71 | 2.00 |
| 11 | 67.20 | 1162.46 | 1.40 | 1.00 |

A notable pattern: clusters 0 through 4 are almost identical to each other in Age, Price, and PurchaseFrequency, differing mainly by CustomerSatisfaction level (landing almost exactly on 1.0, 2.0, 3.0, 4.0, or 5.0 per cluster) — suggesting DBSCAN is largely just separating points by their exact integer Satisfaction value in the densest region of the data, rather than finding meaningfully distinct customer segments across all four features the way K-Means and Agglomerative did. The remaining clusters (5 through 11) are smaller, scattered groups sitting further out in feature space — closer to what DBSCAN is actually designed to detect (dense pockets separated by sparser regions) — but there are many of them (12 total) and they are inconsistent in size and character, rather than forming a small number of clean, business-interpretable segments.

## Final comparison: K-Means vs. Agglomerative vs. DBSCAN — which is the best model

| Algorithm | Clusters | Silhouette Score | Verdict |
|---|---|---|---|
| **K-Means** | 8 | **≈0.225** | **Best — highest score, positive and clearly the strongest separation of the three** |
| Agglomerative Clustering | 8 | ≈0.150 | Middle — positive but weaker than K-Means |
| DBSCAN | 12 (+177 noise) | ≈-0.215 | Worst — negative score, indicates poor cluster fit |

**K-Means is the best-performing clustering algorithm for this dataset**, by a clear margin once all three are compared directly:

1. **Silhouette score is the deciding metric**, applied consistently across all three algorithms. K-Means' ≈0.225 is not just the highest of the three — it's the only score in what would normally be considered even a "weak-to-moderate" acceptable range; Agglomerative's ≈0.150 is weaker still, and DBSCAN's negative score indicates its clustering is actively worse than the point assignments not reflecting real structure.

2. **Why DBSCAN underperformed here specifically:** DBSCAN is designed to find naturally dense regions separated by sparser gaps, and works best when a dataset has that kind of density structure. This dataset's four clustering features don't show that kind of gap-separated density — consistent with the PCA visualization showing one continuous, overlapping blob of points rather than isolated dense islands. Without genuine density gaps to detect, DBSCAN tends to either over-fragment the data into many small, arbitrary clusters (which is what happened here — 12 clusters, several nearly identical except for satisfaction level) or mislabel points as noise, rather than finding the broad, evenly-distributed segments that K-Means and Agglomerative are built to handle.

3. **Consistency with earlier findings:** this result lines up with everything else discovered in this notebook — the elbow method chart showed no sharp bend, the PCA plot showed continuous overlapping structure rather than isolated groups, and even K-Means' own best silhouette score (0.225) was only moderate, not strong. The customer base in this dataset appears to vary fairly continuously across these four features rather than splitting into naturally dense, well-separated groups — a data characteristic that inherently favors centroid-based methods like K-Means over density-based methods like DBSCAN.

**Conclusion carried forward:** K-Means with K=8 is the final recommended clustering solution for this project's customer segmentation task, to be used for any further business interpretation or downstream use of these customer segments.
