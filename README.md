# **🏠 House Price Prediction**

---
## **🏠 Project Vision**
I’m building a Real Estate House Price Prediction system with a mindset of continuous improvement—not just to develop a model that predicts prices, but to evolve it into a complete, production-ready MLOps ecosystem.

The goal is to progressively improve the model’s accuracy, experiment with better features and algorithms, incorporate real-world data, and build an infrastructure capable of handling the entire ML lifecycle—from data collection and preprocessing to model training, evaluation, deployment, monitoring, and continuous retraining.

🚀 This is not a one-time prediction project. It is an evolving ML system designed to become better with every iteration.
---

## **🎯 Problem Statement**
Given a set of house features (X), predict the expected sale price (Y) so that [agents/sellers/a listing platform] (Z) can set or validate a listing price at the time a property is entered into the system (T), with the goal of reducing time-on-market and pricing disputes while maintaining trust in the platform's valuation tool (B).
---
## 💡** Solution**
We designed an end-to-end MLOps pipeline (using ZenML for orchestration and MLflow for experiment tracking and model serving) that automates the full lifecycle of the price-prediction model — from data ingestion and feature engineering, through training and evaluation, to continuous deployment as a REST API. When a new model is trained, the pipeline evaluates it and automatically promotes it to a live MLflow prediction server if it meets the deployment criteria, replacing the previous version with zero manual intervention. Agents or downstream systems can then query the live endpoint (/invocations) with a house's features and receive a price estimate in real time, ensuring the valuation tool always reflects the latest trained model rather than a stale, manually-updated one.
---
## **⚙️Project Structure**
```
prices-predictor-system/
│
├── data/                          # Raw and/or processed datasets (inferred)
│   └── AmesHousing.csv            # Ames Housing dataset used for training
│
├── pipelines/                     # ZenML pipeline definitions (confirmed import path)
│   ├── training_pipeline.py       # Defines ml_pipeline() — ingestion → FE → train → evaluate
│   └── deployment_pipeline.py     # Defines continuous_deployment_pipeline() & inference_pipeline()
│
├── steps/                         # Individual ZenML steps used inside the pipelines (inferred)
│   ├── data_ingestion_step.py     # Loads raw data into a DataFrame
│   ├── feature_engineering_step.py# Encoding, scaling, missing-value handling, etc.
│   ├── data_splitter_step.py      # Train/test split
│   ├── model_building_step.py     # Trains the regression model, logs to MLflow
│   ├── model_evaluator_step.py    # Computes metrics (RMSE, R², etc.)
│   └── model_deployer_step.py     # Deploys model via MLFlowModelDeployer if criteria pass
│
├── src/                           # Reusable helper/utility code (inferred, optional)
│   └── ...                        # Custom transformers, config loaders, etc.
│
├── run_pipeline.py                # CLI entry point: runs the training pipeline (CONFIRMED)
├── run_deployment.py              # CLI entry point: runs continuous deployment +
│                                   # inference pipeline, or stops the service with
│                                   # --stop-service (CONFIRMED)
├── sample_predict.py              # Example client: sends a sample house record to the
│                                   # live MLflow /invocations endpoint (CONFIRMED)
│
├── requirements.txt                # Python dependencies (inferred — not uploaded)
├── config.yaml / .zen/             # ZenML stack & pipeline configuration (inferred)
└── README.md                      # This file
```

## 🛠️Tech Stack

| Layer | Tool |
|---|---|
| Pipeline orchestration | [ZenML](https://zenml.io/) |
| Experiment tracking & model registry | [MLflow](https://mlflow.org/) |
| Model serving | MLflow Model Deployer (local REST server, `/invocations`) |
| Language | Python |
| CLI | [Click](https://click.palletsprojects.com/) |
| Inference client | `requests` (JSON over HTTP) |
| Console output | [Rich](https://github.com/Textualize/rich) |


## **📊 Dataset**
The model is trained on the **Ames Housing dataset** — a widely used residential real estate dataset covering property sales in Ames, Iowa. It includes ~80 features describing each property's physical characteristics, quality ratings, and sale conditions, such as:
| Column | Description |
|--------|-------------|
| 'Structural' | Gr Liv Area, Total Bsmt SF, 1st Flr SF, 2nd Flr SF, Garage Area, Garage Cars |
| 'Quality/condition' | Overall Qual, Overall Cond |
| 'Age' | Year Built, Year Remod/Add, Garage Yr Blt |
| 'Amenities' | Fireplaces, Wood Deck SF, Open Porch SF, Pool Area |
| 'Transaction Context' | Mo Sold, Yr Sold |
The target variable is 'Sale Price'. (This matches the feature set your 'sample_predict.py' sends to the endpoint.)
**The Equation**
Since I don't have the actual model-training code (you'd flagged that gap earlier), I can't state your exact fitted model — but here's how to represent it depending on what algorithm is used:
If it's a linear/regularized regression (e.g. Ridge/Lasso baseline), the general form is:
### Linear / Regularized Regression (baseline)

If the trained model is a linear model (e.g. Linear Regression, Ridge, or Lasso), the predicted sale price is a weighted sum of the input features:

$$\hat{y} = \beta_0 + \sum_{i=1}^{n} \beta_i x_i + \epsilon$$

Where:
- $\hat{y}$ — predicted sale price
- $\beta_0$ — intercept
- $x_i$ — input feature $i$ (e.g. living area, overall quality)
- $\beta_i$ — learned coefficient for feature $i$
- $\epsilon$ — residual error term

### Tree-Based Ensemble (Random Forest / Gradient Boosting)

If the trained model is a tree-based ensemble, there is no single closed-form equation. Instead, the prediction is the sum of outputs across $T$ regression trees:

$$\hat{y} = \sum_{t=1}^{T} f_t(x), \quad f_t \in \mathcal{F}$$

Where:
- $\hat{y}$ — predicted sale price
- $T$ — total number of trees in the ensemble
- $f_t(x)$ — the prediction from the $t$-th regression tree given input $x$
- $\mathcal{F}$ — the space of possible regression trees
---

## **🔮 Prediction**
Given input X, predict Y for user/system Z at decision time T, to optimize business outcome B
| Element | This System |
|---------|-------------|
| **Input X** | Structural + lot features of a house (sqft, quality/condition ratings, year built, garage, porches, etc.) |
| **Predicted Output Y** | Estimated sale price (point estimate; ideally a range) |
| **User/System Z** | Could be framed multiple ways — pick one explicitly: (a) a real estate agent setting a listing price, (b) a self-serve valuation widget for sellers, (c) an internal pricing engine feeding a listings platform |
| **Decision Time T** | At listing time (before the house goes on market) vs. at offer/negotiation time — these have very different latency and accuracy needs |
| **Business Outcome B** | Depends on Z: for an agent tool, B = faster, more competitive listings and fewer overpriced properties sitting stale; for a platform, B = user trust/engagement (accurate estimates keep people coming back); for a lender, B = reduced default risk from bad collateral valuation |
---

## 🏗️ MLOps Architecture
![image Alt](https://github.com/bmwi7/House_Price-Pred-Using-MLops/blob/41d84c0b8ae34efa43b8fefb5ad2b5036fc3700f/mlops_pipeline_flow.png)

---
## **📊 Results**
### Who uses this model?
* Direct users: real estate agents/brokers (comps and pricing suggestions), homeowners/sellers (self-serve valuation), buyers (fairness check on asking price), internal analysts.
* Indirect/system consumers: a CRM or listing platform that calls the API server-side, or a lending/underwriting system that uses the estimate as one input to loan decisions.
* This matters because "agent using it as one signal among many" tolerates more model uncertainty than "system auto-populates a legally binding number."

### Device
* Current setup is server-side inference only — a local MLflow server (127.0.0.1:8000) hit via requests.post. There's no on-device/edge deployment here.
* In production this would sit behind a cloud endpoint, called from a web app, mobile app backend, or another internal service — not run on a phone/browser directly. If you eventually needed offline/on-device estimates (e.g. an agent's mobile app in a low-connectivity area), you'd need a much smaller distilled model.

### Regulation
* Housing valuation brushes up against Fair Housing Act and ECOA (Equal Credit Opportunity Act) if the output ever feeds a lending decision — disparate impact on protected classes is a real risk with location-correlated features.
* If used in mortgage/appraisal contexts: USPAP appraisal standards, and potentially FCRA adverse-action notice requirements (you may need to explain why a price came out low).
* If personal data (owner info) is attached: GDPR/CCPA data-handling obligations.
* Net effect: even though this looks like "just a regression model," if it touches lending or listing decisions, you likely need feature auditing (drop/monitor proxies for race/ethnicity like zip code) and an explainability layer.

### Latency
* Single-record synchronous REST calls, no batching in the sample client — fine for "agent looks up one house," not fine for a bulk revaluation job over a whole portfolio.
* The model itself (looks like a standard sklearn/gradient-boosted regressor on ~35 tabular features) is cheap to score, so latency is dominated by network/serving overhead, not model compute — meaning you have headroom to add complexity (ensembling, SHAP explanations) without hurting speed much.

### Accuracy vs Latency
* Given how cheap inference is here, this is a case where you can buy real accuracy gains almost for free — stacking, feature engineering, even a small ensemble — because the latency budget isn't the constraint. The real cost driver would be feature-computation time if you added external data (comps, school ratings, walkability scores) that require live lookups.

### Performance vs. interpretability
* For price prediction that affects money and possibly lending, interpretability should win over marginal accuracy gains. A well-tuned gradient boosting model + SHAP/feature-importance output, rather than a deep net, gives you both decent accuracy and a defensible "why" — important for disputes ("why did the model say my house is worth less than my neighbor's?").

### Risk vs Speed
* Mispricing risk here is asymmetric and high-stakes (six-figure decisions), so this leans toward prioritizing risk mitigation over raw speed: confidence intervals/prediction ranges rather than a single point estimate, a human-in-the-loop review for outlier predictions, and monitoring for drift (housing markets shift fast — a model trained on older Ames-style data will decay).
---

## **🚀 Future Improvements**

- **Automated retraining trigger** — currently a human re-runs `run_pipeline.py`; add a scheduled or drift-triggered retraining job instead of manual kickoff.
- **Data/model drift monitoring** — housing markets shift; add drift detection (e.g. Evidently) to flag when the live model's predictions start degrading against real outcomes.
- **Explainability layer** — integrate SHAP/feature-importance output alongside predictions, especially important given the fair-housing/lending sensitivity of price estimates.
- **Prediction intervals instead of point estimates** — return a confidence range rather than a single number, given the high stakes of mispricing.
- **Bias/fairness auditing** — audit for proxy features (e.g. location-correlated fields) that could create disparate impact if the model ever feeds lending or listing decisions.
- **Batch scoring endpoint** — current client sends one record at a time; add a batch-scoring path for portfolio-wide revaluation jobs.
- **CI/CD integration** — wire the pipeline into GitHub Actions (or similar) so training/deployment runs automatically on new data or code changes, with test coverage on the feature engineering and model steps.
- **Cloud deployment** — move from the local MLflow daemon to a cloud-hosted endpoint (e.g. SageMaker, Vertex AI, or a containerized service) for real production traffic.

## **👨‍💻Author**

**[Sahil Pathan]**
[GitHub](https://github.com/your-username) · [LinkedIn](www.linkedin.com/in/sahil-pathan-5379b1282) ·
