# StartupLab — 50 Startups Predictive Model Comparison

比較五個真正的預測模型：Linear Regression、Ridge、Lasso、Elastic Net、Random Forest。
所有模型直接產生預測，不再只是「不同特徵選擇方法 + 同一個線性迴歸」。

## 執行

```powershell
python -m pip install -r requirements.txt
python main.py
```

直接開啟 `dashboard/dist/index.html` 即可操作，或執行：

```powershell
python -m http.server 8000 --directory dashboard/dist
```

網站：https://raywu576.github.io/AIOT_Do-in-class-2/

## 實驗設計

- 50 筆資料，固定 80% 訓練、20% 測試，seed=0。
- 五個模型共用相同特徵排序規則：每次僅在訓練折內使用 `SelectKBest(f_regression)`，等同按絕對 Pearson 相關排名。k 是輸入數，而不是非零係數數。
- 固定 k=1～5 的曲線：每個模型在訓練集做 4 折 GridSearchCV，再評估保留測試集。所有模型在同一 k 使用相同特徵，隔離模型本身的差異。
- 正式模型排名：外層 5 折、內層 4 折 nested CV，同時調整 k 與參數。最終最佳模型依外層平均 RMSE 選出，測試集不參與選擇。
- Ridge alpha：0.01、0.1、1、10、100；Lasso：1、10、100、1000、5000；Elastic Net alpha：0.01、0.1、1、10，l1_ratio：0.2、0.8。
- 線性模型在訓練 pipeline 內標準化；Random Forest 使用 60 棵樹、min_samples_leaf=2、max_depth 為 3 或 None。
- 模型曲線可能接近，尤其調參偏好弱正則化時。有限樣本與有限參數網格不代表未來表現保證。

## 網站功能

模型與輸入數切換、雙折線圖、係數或森林特徵重要性、訓練內參數、即時預測、最多五個情境比較、殘差排序、CSV 匯出、深色模式與手機版面。
森林的所有樹會序列化到網站，瀏覽器遍歷實際樹節點計算平均預測；線性模型係數已還原到原始金額單位。僅 Linear Regression 顯示 OLS 推論，正則化模型不套用一般 OLS p 值或信賴區間。

## 輸出

- `results/model_comparison.csv`：五個模型的 nested CV 排名與保留測試集表現。
- `results/feature_count_metrics.csv`：25 組模型、輸入數、調參分數與測試分數。
- `results/best_startup_model.pkl`：依 nested CV 選出的 sklearn pipeline。
- `results/dashboard_models.pkl`：25 組已擬合的 pipeline，供驗證與重現。
- `results/best_model_coefficients.csv`：最佳線性模型的係數；若最佳模型為森林則為空表。
- `results/best_model_importances.csv`：最佳森林的重要性；線性模型則為空值。
- `results/vif_results.csv`：最佳模型輸入的共線性指標。
- `results/previous_feature_selection/`：先前「五種特徵選擇方法」實驗的歷史結果，不能與新實驗混用。

`python main.py` 同步更新本機網站數據；push 到 main 後 GitHub Pages 自動發布資料快照。

## 驗證

```powershell
python -m pip install playwright
python tests/verify_dashboard.py
```

測試使用已安裝的 Microsoft Edge，驗證全部 25 組模型的資料、75 個瀏覽器預測與 Python 一致、模型切換、森林重要性、正則化統計空值、CSV 匯出及手機寬度。
