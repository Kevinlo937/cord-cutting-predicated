# ISP 訂戶離退預測系統 (ISP Customer Churn Prediction System)

本專案旨在構建一套端到端（End-to-End）的機器學習管道，利用 **帳務行為**、**維修工單** 與 **客服互動（含 LLM 語意分析）** 等多源異質數據，預測 ISP 寬頻用戶的流失風險。透過進階的特徵篩選策略（SFSS）與不平衡資料處理機制，本系統不僅產出高準確率的預測名單，更結合 ROI 分析模型提供具體的商業決策支持。

## 📂 專案架構與 Notebook 流程

本專案共包含 9 支核心 Jupyter Notebook，依據資料處理流水分為五大階段：

### Phase 1: 資料前處理 (Data Preparation)

*   **`topMSO_Dataset.ipynb`**
    *   **功能**：基礎資料清洗與標籤定義。
    *   **核心邏輯**：清洗 25 年跨度的訂閱資料，鎖定 CM/EPON 寬頻產品。處理重複客編（Deduplication），並將「欠款斷線」與「拆機」狀態統一標準化為流失標籤（Churn）。
    *   **輸出**：用戶狀態快照與基礎工單資料。

*   **`topMSO_add_Vars.ipynb`**
    *   **功能**：帳務資料清洗與特徵初步提取。
    *   **核心邏輯**：利用 Regex 修復混亂的日期格式，並基於「繳費週期」邏輯修復異常的起訖日。計算「平均繳款延遲日數」與提取「最常見繳費方式」。
    *   **輸出**：含帳務行為的寬表。

### Phase 2: 特徵工程 (Feature Engineering)

*   **`topMSO_addVars_oneHot.ipynb`**
    *   **功能**：類別特徵編碼與降維。
    *   **核心邏輯**：針對繳費方式與週期進行基數縮減（Cardinality Reduction），保留 Top 7 主流通路，其餘歸類為其他，並執行 One-Hot Encoding。經 EDA 發現用戶呈現「提前繳費」與「嚴重拖欠」的雙峰分佈。
    *   **輸出**：數值化的帳務特徵集。

*   **`topMSO_csr_prepared.ipynb`**
    *   **功能**：非結構化客服紀錄（NLP）特徵提取。
    *   **核心邏輯**：結合 **Jieba 關鍵字分析** 與 **LLM 情緒評分**。計算客戶的來電頻率、主要痛點類別（如報修vs費率），並量化其潛在的離退情緒分數。
    *   **輸出**：客戶級別的行為與語意特徵表。

### Phase 3: 資料整合 (Data Integration)

*   **`topMSO_SFSS.ipynb`**
    *   **功能**：多源異質資料合併與缺失值填補。
    *   **核心邏輯**：整合三大特徵來源（工單 a、帳單 b、客服 c）與狀態/存活資料 d 並進行全域缺失值處理，建構 SFSS 前的統一訓練資料集。
    *   **輸出**：包含 90+ 特徵的完整訓練資料集。

### Phase 4: 特徵選擇與模型優化 (Feature Selection & Tuning)

*   **`SFSS_Advanced_Template.ipynb`**
    *   **功能**：進階監督式特徵選擇（SFSS）。
    *   **核心邏輯**：整合 Filter (F-test, MI)、Wrapper (RFECV) 與 Embedded (Lasso, RF, XGB) 多種方法。利用 **Borda 排名聚合（Rank Aggregation）** 找出共識度最高的關鍵特徵，並透過穩定性選擇（Stability Selection）剔除雜訊。
    *   **輸出**：精選特徵子集與重要性報告。

*   **`topMSO_optuna.ipynb`**
    *   **功能**：XGBoost 超參數自動化調優。
    *   **核心邏輯**：使用 **Optuna** 進行貝葉斯優化。針對類別不平衡，將 `scale_pos_weight` 納入搜尋空間，並基於 Feature Importance 進行二次特徵篩選，使 AUC 提升至 0.70 以上。
    *   **輸出**：最佳超參數配置與最佳特徵列表。

### Phase 5: 訓練、預測與商業落地 (Training & Deployment)

*   **`topMSO_TrainAfterSFSS.ipynb`**
    *   **功能**：混合採樣與多模型訓練。
    *   **核心邏輯**：應用 **SMOTE-Tomek** 混合採樣解決 15:1 的嚴重不平衡。訓練 CNN-1D、LSTM、MLP、XGBoost 與 Random Forest 等多種架構，並導入 **Focal Loss** 與動態閾值優化（Threshold Optimization）以最大化 F1-Score。
    *   **輸出**：訓練好的模型檔 (.pkl/.keras) 與閾值設定。

*   **`topMSO_predict_1225.ipynb`**
    *   **功能**：批次預測、回測驗證與 ROI 分析。
    *   **核心邏輯**：載入訓練模型對新資料進行推論，並透過 **多模型投票（Ensemble Voting）** 鎖定高風險客群。結合行銷成本與挽留成功率進行 **ROI 模擬**，計算淨利潤以輔助決策。
    *   **輸出**：流失預測名單與商業價值分析報表。

---

## 🚀 關鍵技術亮點

1.  **SFSS (Supervised Feature Selection Strategy)**：
    不依賴單一指標，而是透過 Borda Count 聚合多種特徵選擇演算法的排名，有效平衡了線性與非線性特徵的捕捉能力，篩選出具備高度穩健性的預測因子。

2.  **不平衡資料處理 (Imbalanced Learning)**：
    針對 ISP 產業典型的低流失率（<10%）場景，本專案採用了「混合採樣 (SMOTE + Tomek Links)」結合「代價敏感學習 (Focal Loss / Scale_Pos_Weight)」，強迫模型專注於學習困難的流失樣本。

3.  **NLP 與 LLM 整合**：
    突破傳統僅用通話次數的限制，引入 LLM 對客服對話紀錄進行情緒評分，並提取具體的抱怨類別，為模型提供了深層的語意特徵。

4.  **商業價值導向 (Profit-driven Evaluation)**：
    模型評估不只看 AUC 或 Accuracy，而是直接計算 **ROI (投資報酬率)**。分析顯示，針對多模型共識的高風險客戶進行精準挽留，ROI 最高可達 300% 以上。

---

## 📚 參考文獻

本專案的方法論參考了以下隨機森林與特徵工程領域的經典文獻：

*   **Breiman, L.** (2001). Random forests. *Machine learning*, 45(1), 5-32. (定義了 Permutation Importance)
*   **Strobl, C., et al.** (2008). Conditional variable importance for random forests. *BMC bioinformatics*. (解決高相關特徵的偏誤)
*   **Kursa, M. B., & Rudnicki, W. R.** (2010). Feature selection with the Boruta package. (全相關特徵選擇)
*   **Altmann, A., et al.** (2010). Permutation importance: a corrected feature importance measure. (引入 P-value 檢定)
*   **Fisher, A., et al.** (2019). All Models are Wrong, but Many are Useful: Learning a Variable's Importance by Studying an Entire Class of Prediction Models Simultaneously. *JMLR*. (Model Class Reliance)
*   **Jean-Charles de Borda** (1781). Mémoire sur les élections au scrutin. (Borda Count 排名聚合法的數學基礎)
