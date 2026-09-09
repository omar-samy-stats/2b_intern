# Supervised ML Model — Task 3 — Notebook Summary

## Purpose of this notebook
This notebook takes the consumer electronics sales dataset (the same clean dataset used in the EDA stage) and builds, evaluates, and compares multiple supervised classification models to predict `PurchaseIntent` (binary: 0 = no intent, 1 = intent). The goal is to select the best-performing model to move forward toward production, based on accuracy, precision, recall, overfitting behavior, and cross-validation stability.

## Preprocessing steps

### Feature and encoding decisions
Only three predictors were selected for modeling: `CustomerGender`, `CustomerAge`, and `CustomerSatisfaction`. This choice traces directly back to the EDA stage's findings — these were the three features that showed statistically significant relationships with PurchaseIntent (via t-tests, chi-square, and correlation), while ProductPrice, PurchaseFrequency, ProductCategory, and ProductBrand were excluded as weak/non-significant predictors. No additional encoding was needed for the chosen predictors: CustomerGender was already properly binary-encoded (0/1), and CustomerAge and CustomerSatisfaction were already numeric.

### Train/test split
Data was split 80/20 (test_size=0.2) using stratified sampling on the target variable, with a fixed random_state for reproducibility. Stratifying on PurchaseIntent ensures both the train and test sets preserve the same class balance (~56.6% / ~43.4%) seen in the full dataset, rather than risking an uneven split by chance.

### Feature scaling
StandardScaler was applied only to CustomerAge and CustomerSatisfaction (not CustomerGender, since it's already binary), transforming them to have mean 0 and standard deviation 1. Critically, this was done *after* the train/test split — the scaler was fit only on the training data and then applied to transform both train and test sets, which avoids data leakage (letting information about the test set influence the scaling parameters).

## Models built and their individual results

### 1. Logistic Regression (linear baseline model)
- **Test Accuracy: 85.7%**, Train Accuracy: 85.4%
- Precision: 0.84 (class 0) / 0.87 (class 1); Recall: 0.83 (class 0) / 0.88 (class 1); F1: 0.83 / 0.87
- Macro avg F1: 0.85, Weighted avg F1: 0.86
- Confusion matrix: 649 true negatives, 131 false positives, 127 false negatives, 893 true positives
- **Interpretation documented in the notebook:** precision and recall are close to each other within each class, and close across both classes — no class is collapsing to a much worse score than the other, which would be the signature of class imbalance actually hurting the model. Since that pattern isn't present, and macro/weighted averages are nearly identical, plain accuracy was judged to be a trustworthy metric for this model despite the mild class imbalance in the dataset. This model serves as the linear baseline against which the more complex models are compared.

### 2. Random Forest (non-linear ensemble model)
- **Test Accuracy: 94.0%**, Train Accuracy: 95.6%
- Precision: 0.94 (class 0) / 0.94 (class 1); Recall: 0.92 (class 0) / 0.96 (class 1); F1: 0.93 / 0.95
- Confusion matrix: 715 true negatives, 65 false positives, 43 false negatives, 977 true positives
- **Overfitting check documented in the notebook:** the train/test accuracy gap is small (95.6% vs 94%), which was explicitly interpreted as showing no meaningful overfitting — the model generalizes well rather than having memorized the training data.
- **Comparison to baseline:** substantially outperforms Logistic Regression's 85.7%, which was interpreted as evidence that the true relationship between the features and PurchaseIntent has meaningful non-linear structure that a linear model like Logistic Regression cannot capture but Random Forest can.
- **Feature importance (Random Forest):** CustomerGender ≈ 0.35, CustomerAge ≈ 0.33, CustomerSatisfaction ≈ 0.32 — notably, Random Forest spreads importance fairly evenly across all three features, without one feature dominating.

### 3. K-Nearest Neighbors (KNN, instance-based non-linear model)
- Configured with n_neighbors = 5
- **Test Accuracy: 94.0%**, Train Accuracy: 95.6% — the reported metrics (precision, recall, F1, confusion matrix) for KNN in this run came out numerically identical to Random Forest's results. This is worth flagging honestly: whether this reflects the two models genuinely converging on the exact same decision boundary on this dataset, or a notebook execution/output artifact (e.g., a stale cell output), isn't fully certain from the notebook alone, and would be worth re-verifying if precise KNN-specific numbers are needed.

### 4. XGBoost (gradient-boosted trees)
- **Test Accuracy: 94.0%**, Train Accuracy: 95.6% — matching Random Forest's headline accuracy exactly
- Same precision/recall/F1 breakdown and confusion matrix as Random Forest (715/65/43/977)
- A direct check (`np.array_equal`) confirmed that XGBoost's predictions and KNN's predictions on the test set were **exactly identical**, row for row — a notable and somewhat unusual finding suggesting the decision problem here is clean/separable enough that different algorithm families converge on the same predictions for this particular dataset.
- **Feature importance (XGBoost):** CustomerGender ≈ 0.78, CustomerSatisfaction ≈ 0.15, CustomerAge ≈ 0.07 — shown as a horizontal bar chart. Unlike Random Forest's evenly-spread importances, XGBoost assigns the large majority of predictive weight to CustomerGender alone, with CustomerAge treated as comparatively minor. This difference between the two tree-based models' importance rankings (RF spreads importance evenly; XGBoost concentrates it heavily on Gender) is a meaningful comparison point between otherwise similarly-performing models.
- **Notebook's interpretation of why Random Forest and XGBoost perform so similarly:** both are tree-based models that split data by asking threshold questions (e.g., "is Satisfaction ≥ 4?"), which suits this dataset well since the underlying decision rule is clean and axis-aligned — given a rule this clear-cut, both algorithms independently discover essentially the same optimal split points, explaining their near-identical performance.

### 5. Support Vector Machine (SVM, RBF kernel)
- **Test Accuracy: 93.7%**
- Precision: 0.94 (class 0) / 0.93 (class 1); Recall: 0.91 (class 0) / 0.96 (class 1); F1: 0.93 / 0.94
- Confusion matrix: 711 true negatives, 69 false positives, 45 false negatives, 975 true positives
- Performs close to, but very slightly below, Random Forest/XGBoost — still dramatically ahead of the Logistic Regression baseline.
- A markdown heading titled "Discriminative VS Generative algorithm" appears directly before this model in the notebook, without further elaboration written underneath it — noting this as a heading/thought placeholder in the notebook rather than a section with developed content.

### 6. MLP Neural Network (Multi-Layer Perceptron)
- Architecture: two hidden layers (64 and 32 neurons), ReLU activation, max_iter=40
- **Test Accuracy: 94.0%**, matching the same precision/recall/F1/confusion matrix values as Random Forest and XGBoost (0.94/0.94 precision, 0.92/0.96 recall, confusion matrix 715/65/43/977)
- A `ConvergenceWarning` was raised, indicating the model had not fully converged within the 40-iteration cap — meaning its true best performance with full convergence wasn't captured in this specific run, though the reported metrics still matched the top-performing tree-based models.
- **Training loss curve:** plotted over iterations, showing loss dropping sharply from about 0.58 down to roughly 0.20 within the first ~25 iterations, then flattening out gradually to settle around 0.18 by iteration ~175-180 — a classic diminishing-returns training curve shape.
- **Train vs. test loss curve (separate deeper run, using warm_start to continue training incrementally):** train loss steadily decreases from about 0.50 down to about 0.19 over 40 iterations, while test loss also decreases initially but plateaus and flattens around 0.24-0.25 starting from roughly iteration 10 onward, staying essentially flat for the remaining iterations while train loss keeps slowly declining. This growing (though modest) gap between train loss continuing to fall while test loss plateaus is the classic early signature of a model starting to overfit — worth noting as a more sensitive overfitting signal than accuracy alone, since accuracy stayed at 94% while this loss-curve gap shows the model's fit to training data was still improving after test performance had already leveled off.

## Cross-validation results (5-fold, stratified where applied)

- **Random Forest — first run (plain KFold, no stratification, no shuffle):** scores of [1.0, 1.0, 1.0, 1.0, 0.7656], mean ≈ 95.3%, but with a large standard deviation (≈0.094) — the unstratified, non-shuffled split produced one fold with a much worse score than the other four (perfect 1.0 in four folds is itself a signal something about that split wasn't well-randomized, likely because the data wasn't shuffled and folds may have landed on non-representative slices).
- **Random Forest — second run (StratifiedKFold, shuffled, random_state=42):** scores of [0.959, 0.956, 0.947, 0.944, 0.960], mean ≈ 95.3%, standard deviation dropped sharply to ≈0.0066 — a far more consistent, trustworthy result once proper stratified shuffling was used. This version is the reliable one to cite.
- **XGBoost (same StratifiedKFold setup):** produced the identical scores to Random Forest's stratified run — [0.959, 0.956, 0.947, 0.944, 0.960], mean ≈ 95.3%, std ≈ 0.0066 — reinforcing the earlier finding that these two models behave almost identically on this dataset.
- **Logistic Regression (StratifiedKFold, shuffled, no fixed random_state this time):** scores around [0.863, 0.858, 0.855, 0.853, 0.846], mean ≈ 85.5%, std ≈ 0.0057 — consistent with its single-split test accuracy of 85.7%, confirming the baseline's performance ceiling is real and stable, not a fluke of one particular split.
- **SVM (StratifiedKFold, shuffled, no fixed random_state):** scores around [0.849, 0.856, 0.861, 0.849, 0.843], mean ≈ 85.2%, std ≈ 0.0060 — notably, this cross-validated SVM mean (≈85.2%) is much lower than the single test-split SVM accuracy reported earlier (93.7%). This discrepancy is worth flagging: it suggests the SVM's strong single-split result may have been at least partly influenced by that particular train/test split rather than being fully representative of its average performance across different data partitions — cross-validation here reveals SVM's real generalization performance is much closer to Logistic Regression's than the original single split suggested.

## Additional exploratory analysis within this notebook

### Gender × Satisfaction interaction (pivot table and heatmap)
A pivot table (visualized as a red-to-green heatmap) was built showing the mean PurchaseIntent for every combination of CustomerGender and CustomerSatisfaction level. The pattern found: for CustomerGender = 0, mean purchase intent stays very low (around 0.04-0.05) at satisfaction levels 1-3, then jumps sharply to around 0.72-0.73 at satisfaction levels 4-5. For CustomerGender = 1, mean purchase intent is already much higher (around 0.71-0.72) even at satisfaction levels 1-3, and climbs further to around 0.96 at satisfaction levels 4-5. This reveals a clear non-linear interaction effect between Gender and Satisfaction — the relationship isn't simply additive, and this kind of interaction is exactly the sort of pattern that non-linear models (Random Forest, XGBoost, SVM, MLP) can capture naturally but a plain linear Logistic Regression model captures less effectively, which helps explain the large accuracy gap between the linear baseline and the non-linear models.

### Why accuracy is high with no overfitting (train accuracy not far above test accuracy)
The Gender × Satisfaction pivot table above is the direct explanation for why the non-linear models reach ~94% accuracy while still showing only a small train/test gap (e.g., Random Forest's 95.6% train vs. 94% test) rather than the large train/test gap that usually signals overfitting. Because CustomerGender and CustomerSatisfaction combine to produce purchase-intent groups that are already close to cleanly separated in the real data (roughly 0.04-0.05 mean intent in the "low" combinations vs. roughly 0.96 mean intent in the "high" combinations, with comparatively little overlap in between), the underlying classification problem itself is close to genuinely separable using just these two features. A model doesn't need to memorize noise or fit complex, dataset-specific quirks to achieve high accuracy here — the true signal in the data is simply strong and clean. That's why a tree-based model can fit the training data well (high train accuracy) and generalize almost as well to unseen data (similarly high test accuracy): it isn't learning spurious patterns that only exist in the training set, it's learning a real, strong relationship that holds consistently across the whole dataset. This is also consistent with why Random Forest and XGBoost independently arrive at very similar decision rules (as noted above) — when the true separating pattern in the data is this clean, there's little room for different models, or a model overfitting to training-specific noise, to diverge from that same, correct, generalizable rule.

### How each model architecture actually catches the Gender × Satisfaction interaction
Each non-linear model captures the same underlying interaction, but through a different mechanism:

- **Random Forest & XGBoost (tree-based):** trees work by asking sequential threshold questions on one feature at a time (e.g., "is CustomerGender = 1?", then within that branch, "is CustomerSatisfaction ≥ 4?"). Nesting these splits carves the feature space into the same four regions the pivot table showed, each with a consistent purchase-intent rate. This means trees capture the interaction explicitly and directly, without needing a manually engineered interaction term — nested splits structurally *are* an interaction. This is exactly what Logistic Regression cannot do, since it only combines features through a single weighted sum and cannot represent "the effect of Satisfaction depends on Gender" unless that interaction is explicitly engineered as its own feature, which wasn't done here.

- **KNN:** works purely on distance in feature space rather than thresholds or splits. Because customers with similar Gender/Satisfaction values sit close together, and because those same-region customers overwhelmingly share the same PurchaseIntent label, the 5 nearest neighbors to any given point are very likely to vote the same way. KNN captures the interaction implicitly, not structurally — it relies on the four Gender × Satisfaction groups being tightly clustered and well-separated in space rather than learning an explicit rule. This is also consistent with why KNN and XGBoost produced literally identical predictions in this notebook: when groups are this cleanly separated, "find the nearest neighbors" and "apply the correct nested threshold rule" converge on the same answer for almost every point.

- **MLP Neural Network (ReLU activation):** a single neuron's weighted sum alone is just as linear as Logistic Regression, but the ReLU activation (`max(0, x)`) breaks that linearity by "turning off" below a threshold and passing through linearly above it — effectively acting as a soft, learnable threshold gate. With multiple ReLU neurons in a hidden layer, the network can combine several of these threshold gates together (one neuron partially activating for "Gender is high," another for "Satisfaction is high"), and neurons in the next layer can combine those signals through learned weights to produce a much higher output only when both conditions hold together. Stacking ReLU layers like this lets the network approximate the same kind of AND-like interaction that a tree's nested splits represent directly — learned through weighted combinations across layers instead of explicit thresholds.

In short: trees represent the interaction explicitly via nested threshold splits, KNN captures it implicitly by relying on well-separated groups being close together in space, and the neural network approximates it indirectly by combining multiple threshold-like ReLU gates through learned weights across layers.

### Duplicate-row check on the predictor subset
Checking `CustomerGender`, `CustomerSatisfaction`, and `CustomerAge` together for unique combinations returned only 520 distinct combinations out of 9,000 rows — meaning many rows share identical predictor values (which is expected, since Age has a limited range of integers, Satisfaction only takes 5 values, and Gender only 2 — the combination space is inherently small relative to 9,000 rows).

### Random Forest confusion matrix (visualized)
A labeled confusion matrix display (Blues color scheme, labeled "No Intent" / "Intent" instead of raw 0/1) was generated for the Random Forest model, visually presenting the same values noted above: 715 correctly predicted "No Intent," 977 correctly predicted "Intent," 65 actual "No Intent" cases misclassified as "Intent," and 43 actual "Intent" cases misclassified as "No Intent" — visually reinforcing that the model makes noticeably fewer false negatives (missed real purchase intent) than false positives.

## Overall model comparison and conclusion

Ranking by test accuracy (single split): Random Forest, XGBoost, MLP all tied at 94.0%; SVM close behind at 93.7%; Logistic Regression trailing well behind at 85.7% as the weakest, purely linear baseline.

However, once cross-validation is taken into account, the picture sharpens: **Random Forest and XGBoost are the most reliably strong performers**, both averaging ≈95.3% across stratified folds with very low variance (std ≈0.0066), meaning their strong performance is consistent and not dependent on a lucky split. **SVM's true average performance (≈85.2% under cross-validation) turned out to be much closer to Logistic Regression's (≈85.5%) than its single-split 93.7% suggested** — a meaningful caution about not trusting a single train/test split's numbers at face value. MLP's headline accuracy matched the top tier, but its train-vs-test loss curve showed early signs of overfitting (test loss plateauing while train loss kept declining) and it hadn't fully converged within the iteration budget used, so its numbers should be read as promising but not as thoroughly validated as Random Forest's and XGBoost's cross-validated results.

**Final takeaway carried forward:** Random Forest and XGBoost are the strongest, most trustworthy candidates for production use on this problem, backed by both a small train/test accuracy gap (no overfitting) and stable, low-variance cross-validation scores. The notebook's own closing note states that the models' cross-validation results were satisfactory enough that hyperparameter tuning was judged unnecessary — the models already performed at their practical best without needing further tuning.
