# AIOT Do-in-class-2

## 任意資料集分析（本機開發版，尚未推送）

保留原有介面、兩個靜態資料集與全部既有實驗結果。新資料使用共用 Python 引擎；一般瀏覽器 JavaScript 只負責視覺化與已匯出線性模型的預測，不執行 Scikit-learn。

### 啟動上傳網站

```powershell
python -m pip install -r requirements.txt
python -m uvicorn api:app --host 127.0.0.1 --port 8000
```

開啟 **http://127.0.0.1:8000**，展開「上傳自己的 CSV／Excel」，即可：

1. 上傳 CSV／.xlsx（最多 10 MB、5,000 筆），預覽前 10 筆及欄位統計。
2. 選擇連續數值目標 Y、輸入 X、目標單位與測試集比例。
3. 設定種子、內外層 CV、k 比較範圍；啟動非同步五方法分析。
4. 在原下拉選單切換新結果，查看圖表、公式、係數、訓練診斷與任意類別輸入。
5. 匯出目前模型 CSV、完整分析 JSON 或診斷 PNG。可清除伺服器暫存上傳。

**GitHub Pages 無法執行 FastAPI／Python**，仍提供原兩個資料集的靜態備援。上傳功能請由上面的本機網址開啟；若要對外提供上傳功能，需另外部署 Python 服務並設置適當的存取與資料保護。預設僅監聽 loopback，不要求 API 金鑰。

### 命令列分析

```powershell
python main.py data.csv --target SalePrice --features Area Rooms Region --counts 1 2 4 --output results/custom-run-001
python -m regression.cli data.xlsx --target Outcome --test-size 0.25 --seed 42 --inner-folds 3 --outer-folds 3 --output results/custom-run-002
```

輸出目錄必須不存在，避免覆寫先前結果；會建立 `analysis.json`、`model_comparison.csv`、`models.joblib`、`linearity.png`、`qq.png`。

不帶參數的 `python main.py` 與 `python housing_experiment.py` 仍保留原實驗重現流程。既有 JSON／CSV／圖檔不會因使用上傳服務或新 CLI 被更動。歷史靜態 Boston 結果採移除缺值列；**新引擎對相同原始 Boston 檔採訓練折內補值，因此保留 506 筆，數值不必與歷史結果相同**。

### 引擎設計與限制

- `regression/loading.py`：檔案解析、欄位型別、樣本、缺值、重複、常數與 ID 檢查；CSV 使用 UTF-8／CP950，Excel 使用首個工作表。
- 缺失或非有限目標先移除；非數值或少於 5 種值的目標拒絕。至少 20 筆有效、非重複觀察，並檢查所選 CV 下的訓練樣本數。
- 預設排除 ID／序號、訓練資料中的常數／全缺值與完全複製目標的欄位。合法的完美線性關係只提醒檢查，不直接排除。其他業務洩漏需由使用者依欄位意義取消勾選。
- 數值中位數補值、類別眾數補值及 OneHotEncoder(drop=first, handle_unknown=ignore) 都在每個訓練折擬合；沒有先對全資料 fit_transform 再 CV。未知類別視為全部 dummy 為 0；JSON 含完整編碼對照。
- 五種方法沿用 `feature_selection.py` 的 FixedCountSelector。RFECV 在原始資料的內層 CV 中比較 RFE 消除路徑；LASSO 的新引擎嚴格模式把標準化放入 Lasso 調參的訓練 pipeline。固定 k 的定義仍是所選輸入數，而非非零 Lasso 係數數量。
- Baseline 保留所有有效編碼輸入；最終預測器一律 LinearRegression。k 與方法只依訓練資料的內層／外層分數決定；不使用保留測試集選模型。內層驗證／外層驗證都重新擬合原始資料的前處理與選取器。
- 每個方法列出 Train RMSE、Nested CV RMSE、Test RMSE／MAE／MSE／R²／Adjusted R² 與所選欄位。正式排名依外層平均 RMSE，固定 k 分數供探索。
- 以訓練殘差產生線性圖、Q-Q、Shapiro–Wilk、Breusch–Pagan 與 VIF。DW 僅在使用者聲明列順序有意義時，依原始列序計算。自由度不足、奇異設計或未定義統計輸出 null，網站顯示「—」；OLS 推論未校正特徵選擇效應。
- 預設上限：60 個編碼特徵、最多 12 個特徵的 8 個比較點、內外層各 3 折（可設定 2～5）、15,000 工作量單位、180 秒。限制可透過 Python 設定降低或在明確上限內提高；API 不允許超過伺服器上限。
- API 同時最多 1 個分析程序與 1 個等待工作。程序超時會終止；CLI 在模型擬合之間檢查時間。上傳最多 8 個 session、24 個暫存分析，1 小時後惰性清除，或服務停止時清除。上傳與產生的模型只在系統暫存目錄，不放入 Git。
- 未選中的類別、補值常數與前處理資訊會隨結果保存。瀏覽器的編碼線性預測已和後端 sklearn pipeline 對照；新結果只在本次頁面記憶體內，重新整理後須由匯出或再次分析取得。

### 驗證與修改說明

```powershell
python -m pip install -r requirements-dev.txt
python -m unittest discover -s tests -p "test_*.py" -v
python tests/verify_dashboard.py
python tests/verify_housing.py
# 先啟動上述 uvicorn，再執行：
python tests/verify_upload.py
```

測試包含原兩個資料集、20／39 數值特徵、類別／缺值、Excel、無效資料、驗證隔離、奇異統計、真實 API 程序與任意資料預測。瀏覽器測試使用已安裝的 Microsoft Edge。各檔案用途與相容性設計見 [IMPLEMENTATION.md](IMPLEMENTATION.md)。

---


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
