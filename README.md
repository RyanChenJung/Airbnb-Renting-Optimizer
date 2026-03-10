# A Hybrid Machine Learning Approach to Understanding Airbnb Revenue Drivers in Chicago

## 📌 Overview
Short-term rental markets exhibit substantial heterogeneity in performance, even among listings with similar structural characteristics. This project develops a machine learning-based decision framework to predict listing-level annual revenue and explain the factors driving performance differences in the Chicago Airbnb market. By integrating structured listing attributes, engineered geospatial features, and natural language processing (NLP) signals derived from descriptions and guest reviews, this study provides actionable insights for hosts and investors seeking to maximize revenue.

## 📊 Methodology
The project follows a comprehensive, multi-layered modeling approach:

* **Target Variable Definition**: The target variable is `log_revenue`, which was calculated by converting calendar availability into an occupancy proxy, estimating annual revenue, consolidating the distribution at the 99th percentile to mitigate extreme outliers, and applying a natural log transformation.
* **Geospatial Feature Engineering**: We integrated Chicago Open Data to compute proximity-based features using cKDTree algorithms. Features include distance to transit, crime density, and gravity-based models capturing the decaying influence of proximity to the Lake Michigan shoreline.
* **NLP for Guest & Host Signals**:
    * *Listing Descriptions*: Preprocessed using SpaCy and Gensim Phrases, followed by Latent Dirichlet Allocation (LDA) to extract eight latent marketing themes (e.g., luxury narratives, walkable neighborhoods).
    * *Guest Reviews*: Utilized BERTopic with transformer-based embeddings to capture sentence-level semantic meaning, producing probability vectors for distinct guest experience themes.
* **Unsupervised Learning**: Applied K-Means and UMAP followed by HDBSCAN density-based clustering to segment the market based on refined NLP experience features and geospatial market indicators.
* **Supervised Learning**: Trained and evaluated Random Forest and XGBoost models to predict `log_revenue`. Models were compared across a baseline feature set and an enriched 61-predictor dataset, utilizing SHAP (SHapley Additive exPlanations) values for interpretability.

## 📈 Key Results
The integration of advanced spatial and semantic features yielded significant improvements over baseline models:

* **Supervised Prediction**: The advanced XGBoost model achieved the best predictive performance with an R² of 0.4121 and an RMSE of 1.1183. The enriched feature set improved the R² by approximately 16 percentage points compared to the baseline models.
* **Revenue Drivers**: Nightly price and competitive positioning (neighborhood relative pricing) are the strongest revenue predictors. Host operational scale and NLP topic features—specifically serviced living suites and walkable neighborhoods—also heavily influence predicted revenue.
* **Market Segmentation**: Unsupervised learning successfully grouped the market into two dominant operational segments: professionally managed hospitality-style listings and locally operated home-sharing properties. Replacing aggregate ratings with granular NLP metrics drastically improved cluster quality, raising the DBCV score from 0.05 to 0.91.

---

## 🗄️ Repository Structure & Usage

```text
├── data/               # Contains raw datasets (listings.csv, reviews.csv, calendar.csv, neighbourhoods.geojson)
├── notebooks/          # EDA, geospatial engineering, NLP pipelines, and clustering analysis
├── src/                # Modularized scripts for feature engineering, model training, and evaluation
├── requirements.txt    # Project dependencies
└── README.md
```

```bash
# 1. Clone the repository
git clone [https://github.com/RyanChenJung/Airbnb-Renting-Optimizer.git](https://github.com/RyanChenJung/Airbnb-Renting-Optimizer.git)
cd Airbnb-Renting-Optimizer

# 2. Install dependencies
pip install -r requirements.txt

# 3. Execute the data processing pipeline
python src/preprocess_and_engineer.py

# 4. Train supervised and unsupervised models
python src/train_models.py
```

---

## 👥 Contributors
* Ryan Chen
* Shenna Lu
* Zimeng Yi
* Nyssa Yota
* Szuyu Chi
* Ellie Yang
