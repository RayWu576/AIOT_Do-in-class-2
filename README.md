# AIOT Do-in-class-2

50 Startups - Multiple Linear Regression with CRISP-DM and 5 Feature Selection Methods.

## Models
- **M0**: Baseline
- **M1**: Correlation (absolute Pearson correlation)
- **M2**: Mutual Information
- **M3**: RFE
- **M4**: RFECV
- **M5**: LASSO

Correlation, Mutual Information, and RFE choose their feature count using training-only cross-validation. All five methods select features before fitting LinearRegression. Baseline retains all features; nested CV RMSE determines the best model.

The two-panel `results/model_comparison.png` shows test RMSE and R-squared for 1 through 5 features. `results/feature_count_metrics.csv` contains each plotted score and selected subset. RFECV's fixed-count curve uses its RFE elimination path; LASSO's fixed-count curve ranks absolute standardized LassoCV coefficients, breaking zero-coefficient ties by input column order. These curves describe fixed-count subsets, while the model comparison reports RFECV's and LASSO's automatically selected subsets.

## Structure
- `main.py`: Main modeling and evaluation script
- `50_Startups.csv`: Dataset
- `results/`: Output plots, metrics, and best model artifacts

## Interactive dashboard

執行 `python main.py` 會同步更新分析結果及 `dashboard/dist/data.js`、`data.json`。

直接用瀏覽器開啟 `dashboard/dist/index.html` 即可操作，不需要 Node.js 或後端伺服器。
也可以在專案根目錄執行 `python -m http.server 8000 --directory dashboard/dist`，再開啟 `http://localhost:8000`。

功能：五種方法與固定特徵數切換、雙互動折線圖、即時預測、最多五個情境比較、OLS 係數與統計、殘差排序、CSV 匯出、深色模式及手機版面。

網站數值來自 Python 實際擬合的 25 組模型。訓練集統計與測試集評分分開標示，固定 k 曲線與依 nested CV 選擇的正式模型排名分開呈現。情境保留於本次頁面，重新整理會清除。線上網站是發布時的資料快照；本機重新執行後需重新發布才能更新線上版本。
