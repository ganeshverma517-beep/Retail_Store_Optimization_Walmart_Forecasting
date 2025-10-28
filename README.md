#  Retail Store Optimization — Walmart Sales Forecasting

###  Overview
This project analyzes historical Walmart sales data to identify patterns, cluster stores, and forecast future sales using classical machine learning and time-series models.

###  Techniques Used
- Exploratory Data Analysis (EDA)
- K-Means Clustering for store segmentation
- ARIMA-based forecasting for weekly sales trends
- Business insights & recommendations

###  Dataset
Kaggle Competition: [Walmart Recruiting - Store Sales Forecasting](https://www.kaggle.com/competitions/walmart-recruiting-store-sales-forecasting)

###  Files in Repository
| File | Description |
|------|--------------|
| `Retail_Store_Optimization.ipynb` | Clean Jupyter Notebook with analysis, visuals, and insights |
| `retail_store_optimization.py` | Optimized Python script version |
| `Retail_Store_Optimization_Walmart_Sales_Report.docx` | Final written report |
| `README.md` | Project documentation |

###  How to Run
1. Download the Walmart dataset from the Kaggle link above.
2. Place files in the `/kaggle/input/` directory or adjust paths in the code.
3. Run the notebook cells sequentially on [Kaggle](https://www.kaggle.com) or locally with Jupyter.

###  Outputs
- Visualizations: Sales trends, seasonality heatmap, cluster profiles.
- Forecast results: ARIMA model predicting next 8 weeks of store sales.
- CSV outputs:
  - `/kaggle/working/Store_Clusters.csv`
  - `/kaggle/working/Store_Forecast.csv`

###  Author
**Ganesh Chandra Verma**  
*Project developed as part of Retail Optimization case study submission.*
