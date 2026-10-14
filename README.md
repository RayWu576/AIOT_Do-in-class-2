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


## 實驗版本與驗證

目前採用 Correlation、Mutual Information、RFE、RFECV、LASSO 五種特徵選擇方法，選完特徵後皆用 LinearRegression 評估。它們不是五種不同的預測器；相同特徵組合的曲線重疊是正常現象。

前一次五個預測模型的實驗結果保留於 `results/previous_predictive_models/`，不與目前結果混用。

安裝 `requirements.txt` 後執行 `python main.py` 更新結果。安裝 Playwright 與 Microsoft Edge 後，可執行 `python tests/verify_dashboard.py` 檢查 25 組特徵子集、Python／網站數值及互動功能。


## 問題下拉選單：Boston Housing

首頁可切換「50 Startups：公司利潤預測」與「Boston Housing：房價預測」，五種特徵選擇方法保持不變。模型、圖表、係數、預測輸入、單位與殘差會同步切換。切換資料集會清除情境比較，避免混合不同單位。

Boston 原始資料來自使用者指定的 [Kaggle / altavish / Boston Housing Dataset](https://www.kaggle.com/datasets/altavish/boston-housing-dataset)，保存於 `datasets/boston_housing/HousingData.csv`。506 筆、13 個輸入欄位、MEDV 目標；此版本 CRIM、ZN、INDUS、CHAS、AGE、LSTAT 各有 20 筆缺值。採完整資料列分析，清理後 394 筆，固定 seed=0 分割為 315 筆訓練、79 筆測試。未使用其他版本替代或人工填入數據。

MEDV 的單位為千美元，例如 30 代表 30,000 美元。這是歷史區域房價中位數，不能當作當前單一物件估價。

- `python main.py`：依序更新兩個資料集。
- `python housing_experiment.py`：只重跑 Boston Housing。
- Boston 的結果在 `results/boston_housing/`；50 Startups 的結果仍在 `results/`。
- 兩個問題各自做相同的外層 5 折、內層 4 折交叉驗證。Boston 產生 65 組固定 k 結果（5 方法 × 13 特徵數）。
- 網站使用 `data.json` 的 `datasets.startups` 與 `datasets.boston`，保留頂層 startups 欄位相容舊驗證流程。重新執行任何一個問題會保留另一問題的已匯出資料。
- 未選用的預測欄位會淡化；Boston 的 CHAS 限制為 0 或 1。測試殘差每頁 20 筆，可排序與換頁。

執行 `python tests/verify_housing.py` 檢查來源缺失值與清理筆數、65 組 Python／網站一致性、資料集切換、單位、預測輸入、CSV、分頁及手機版面。
