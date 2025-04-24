<h1>🚚 Supply Chain Risk Prediction & Dashboard</h1>  
<h2>🔍 Problem Statement</h2>  
Supply chain disruptions can cause massive financial losses. The goal of this project is to predict key disruption metrics like:  
  
- Penalty Cost (USD)  
- Compensation Paid (USD)  
- Disruption Cost (USD)   
- Time to Recovery (Days)  
  
...based on only three intuitive inputs:  
- On-Time Delivery Rate  
- Financial Stability Score  
- Disruption Severity  
  
These predictions help businesses proactively assess the risk level of suppliers and make strategic sourcing decisions.  
  
<h2>Machine Learning Pipeline</h2>  
The full analysis and model training were conducted in Google Colab using the file Copy_of_AI_SupplyChain.ipynb. Here's a summary of the workflow:  
  
✅ Step 1: Data Preparation  
- Cleaned and encoded categorical variables  
- Focused only on key numerical predictors  
- Trained a MinMaxScaler for the 3 input features  
  
✅ Step 2: Model Training  
- Built Random Forest Regressors for each of the 4 target KPIs  
- Trained only using:  
  - On_Time_Delivery_Rate  
  - Financial_Stability_Score  
  - Severity (encoded)  
- Evaluated using RMSE and R² score  
  
✅ Step 3: Explainability  
- Used SHAP (SHapley Additive exPlanations) to interpret model outputs  
- Visualized feature contributions for each KPI prediction

<h2>🖥️ Interactive Streamlit Dashboard</h2>  
An intuitive Streamlit web app was developed to allow users to interact with the model in real-time.
  
🔧 Features:  
- Adjustable sliders for:  
  - Delivery Rate  
  - Financial Score  
  - Severity (categorical)  
- Instant predictions for 4 KPIs  
- Optional SHAP bar charts to understand prediction logic  
- Dynamic risk assessment:  
  - 📗 Low, ⚠️ Medium, or 🚨 High  
- Business recommendation based on risk level
  
<h2>🏢 Business Value</h2>  
This project allows companies to:  
- Evaluate supplier risk before onboarding or contract renewal  
- Identify potential financial impact due to disruptions  
- Use explainable models to justify sourcing decisions  
- Improve supply chain resilience and response time  
  
It's a powerful tool for procurement teams, logistics managers, and risk analysts.  
  
<h2>🧠 Tech Stack</h2>  
- Python, Pandas, Scikit-learn  
- SHAP for explainability 
- Streamlit for web UI  
- Matplotlib for visualization  
  

