# topMSO 客戶離退預測專案

本專案包含針對原始有線電視用戶資料的預處理、視覺化分析，以及機器學習模型建置，目的是探索並預測用戶離退（Churn）行為，支援企業精準營運決策。

---

## 📂 程式說明

### 1. `topMSO_Dataset.ipynb`

* **功能**：針對原始資料檔進行預處理，清除離散度過大、非關注服務、甚至不合理資料。
* **內容**：關聯多份原始資料，建構組合後的清洗資料集，以供後續分析與模型使用。
* **輸出檔案**：

| 檔案名稱                       | 說明                                                                                                                             |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| `cleaned_dataset2.csv`     | 去識別化後的用戶服務資料集。因檔案過大，提供 [Google Drive 下載連結](https://drive.google.com/file/d/1YK3mAFEyT3nr9Qba7fbNOBz4u88KRyI1/view?usp=sharing) |
| `cleaned_ds_doService.csv` | 去識別化後的派工單資料集。因檔案過大，提供 [Google Drive 下載連結](https://drive.google.com/file/d/1YL82PAH9FkJwbzUo_LB3aAGyX5os70nq/view?usp=sharing)  |
| `serviceDay_diff.csv`      | 以最後一次派工單為基準，統計 30 天、60 天、90 天內及以上的派工次數之資料集。                                                                                    |
| `ds_doService_combo.csv`   | 將派工單 + 用戶服務 + 派工次數整合的合併資料集。                                                                                                    |

---

### 2. `topMSO_visualization.ipynb`

* **功能**：以視覺化方式理解資料特性，針對可能與拆機/離退行為相關的因素進行探索。
* **重點分析項目**：

  * 派工次數與等待時間分布
  * 資料間的邏輯與統計相關性驗證

---

## 🧠 機器學習模組

### 📊 資料集說明

| 檔案名稱                    | 說明                                                               |
| ----------------------- | ---------------------------------------------------------------- |
| `ds_numService.csv`     | 由 `topMSO_LogisticRegression.ipynb` 經過資料過濾與欄位數值化後產生，並應用於後續機器學習訓練 |
| `train_data_column.csv` | 整合訓練資料與欄位名稱，取代原本 `ai_train_data.csv` 及 `欄位名稱.txt`                |
| 🔁 `SMOTE` 擴充技術         | 由於正負樣本分布不均，訓練資料經過 SMOTE 處理以平衡樣本比例                                |

### 🧪 使用模型與程式

| 程式檔案名稱                            | 說明                                                                       |
| --------------------------------- | ------------------------------------------------------------------------ |
| `topMSO_LogisticRegression.ipynb` | 邏輯回歸模型建置與評估                                                              |
| `topMSO_DecisionTree.ipynb`       | 決策樹分類模型訓練與視覺化                                                            |
| `topMSO_KNN.ipynb`                | K-最近鄰（KNN）分類模型分析，**本方法於各模型中表現最佳，accuracy最高，因此另儲存預訓練模型檔 `knn_model.pkl`** |
| `topMSO_SVM.ipynb`                | 支援向量機（SVM）模型建構                                                           |
| `topMSO_predict.ipynb`            | 載入 `knn_model.pkl` 模型，針對近期資料集 `ds_test_大屯2025Q1.csv` 進行離退預測並輸出名單         |
| `topMSO_DL_train.ipynb`           | 深度學習模型訓練與評估，包含 MLP、CNN 和 LSTM 三種模型架構，使用 TensorFlow/Keras 實作              |

---

## 📂 平行專案測試資料集

以下資料並未應用於本專案程式碼中，主要由其他平行專案用於測試與驗證模型行為：

| 檔案名稱                        | 說明                             |
| --------------------------- | ------------------------------ |
| `ai_pred_data_20250331.csv` | 2025 年用戶預測資料，供模型推論用            |
| `20250331-離退清單.xls`         | 2025 年 3 月實際離退用戶清單，用於驗證預測結果準確性 |
| `train_data_column.csv`     | 同時應用於該專案訓練模型之欄位與資料合併格式         |

---

## 📦 原始資料說明

### 1. `用戶服務資料2.zip`

* **內容**：個資去識別化後的客戶基本服務資料
* **格式**：CSV 檔，欄位間以 `^` 符號分隔

### 2. `工單.zip`

* **內容**：2022\~2024 年的用戶派工單資料
* **格式**：CSV 檔，欄位間以 `^` 符號分隔

---

## 🆕 Notebook Workflow (2022–2024 Billing & CSR Refresh)

下列五本 Notebook 以 **資料前處理 → 特徵衍生 → 資料整併 → 模型訓練** 的流水線形式，重新整備 2022‒2024 帳務、派工、客服來電三大資料源。

| Notebook                            | 主要目的                                          | 重要輸入                                                                                                   | 重要輸出                                                                   |
| ----------------------------------- | --------------------------------------------- | ------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------- |
| **`topMSO_add_Vars.ipynb`**         | 為派工資料集新增「平均繳款延遲日數、最常見繳款方式」兩項帳務特徵              | 2022‑2024 帳單原始檔、`ds_numService.csv`                                                                    | `bill_22_24_filtered.csv`, `ds_numService_with_billing.csv`            |
| **`topMSO_addVars_oneHot.ipynb`**   | 將 繳別、最常見繳款方式、產品名稱\_數值 三欄做 One‑Hot 編碼          | `ds_numService_with_billing.csv`                                                                       | `ds_numService_with_billing_onehot.csv`                                |
| **`topMSO_csr_prepard.ipynb`**      | 解析 2024 全年度客服來電紀錄，產出用於模型的來電特徵                 | `csr_record_2024.csv`                                                                                  | `final_customer_features_dataset.csv`                                  |
| **`topMSO_csr_service_bill.ipynb`** | 將派工+帳務（已編碼）+來電特徵整合，並補上 2024/12/31 使用狀態(label) | `ds_numService_with_billing_onehot.csv`, `final_customer_features_dataset.csv`, `cleaned_dataset2.csv` | `csr_service_bill.csv`                                                 |
| **`topMSO_ML_train.ipynb`**         | 以整合後資料集分別訓練 KNN / SVM / Random Forest 三類模型    | `csr_service_bill.csv`                                                                                 | `KNN_model.pkl`, `SVM_model.pkl`, `RF_model.pkl`, `model_metrics.json` |
| **`topMSO_DL_train.ipynb`**         | 以深度學習方法建構離退預測模型，比較 MLP、CNN、LSTM 三種架構效能       | `csr_service_bill.csv`                                                                                 | 訓練完成的 MLP、CNN、LSTM 模型                                                |

> **欄位標準化規則**
>
> * **繳別**：僅保留 1, 3, 6, 12, 15；其餘皆設為 99（其他週期）。
> * **最常見繳款方式**：僅保留 7‑11 CVS 臨櫃、廠商代收、簡訊 7‑11、金融機構轉帳、當月繳帳單、信用卡扣款、APP 7‑11，其餘合併為「其他繳款方式」。
> * **產品名稱\_數值**：0=CM、1=EPON。

完整流程請依序執行上表 Notebook，即可得到最新合併資料集與三組機器學習模型。

### 深度學習模型說明 (`topMSO_DL_train.ipynb`)

* **功能**：使用深度學習方法建構客戶離退預測模型，比較不同神經網路架構的效能
* **主要步驟**：
  * 載入整合後的 `csr_service_bill.csv` 資料集
  * 資料預處理：清理缺失值、特徵選擇、標準化
  * 使用 SMOTE 處理樣本不平衡問題（離退樣本約佔 7.9%）
  * 建構並訓練三種深度學習模型：
    * **MLP (多層感知器)**：包含 64 與 32 個神經元的兩個隱藏層
    * **CNN (卷積神經網路)**：使用一維卷積處理序列特徵
    * **LSTM (長短期記憶網路)**：捕捉特徵間的序列關係
  * 模型評估與比較：使用分類報告與混淆矩陣評估各模型效能
* **技術特點**：
  * 使用 TensorFlow/Keras 框架實作
  * 針對不同模型調整資料形狀（MLP 使用標準特徵，CNN/LSTM 使用序列形式）
  * 採用二元交叉熵損失函數與 Adam 優化器
  * 每個模型訓練 20 個 epochs，批次大小為 32
* **應用場景**：適用於需要高精度離退預測的情境，特別是當特徵間存在複雜非線性關係時

---

## 🎯 專案目標

* 建構拆機/離退預測模型
* 探索影響客戶忠誠度的因素
* 提供數據支撐的營運優化建議

---

## 📬 聯絡人

研究負責人：[kevinlo937@gmail.com](mailto:kevinlo937@gmail.com)
（如需資料或技術討論，請透過專案頁留言）
