# topMSO 客戶離退預測資料預處理與視覺化專案

本專案包含針對原始有線電視用戶資料的預處理與資料視覺化，目的是為後續進行用戶離退（Churn）預測模型建置奠定基礎。

---

## 📂 程式說明

### 1. `topMSO_Dataset.ipynb`

- **功能**：針對原始資料檔進行預處理，清除離散度過大、非關注服務、甚至不合理資料。
- **內容**：關聯多份原始資料，建構組合後的清洗資料集，以供後續分析與模型使用。
- **輸出檔案**：

| 檔案名稱              | 說明                                                                 |
|-----------------------|----------------------------------------------------------------------|
| `cleaned_dataset2.csv`     | 去識別化後的用戶服務資料集。因檔案過大，提供 [Google Drive 下載連結](https://drive.google.com/file/d/1YK3mAFEyT3nr9Qba7fbNOBz4u88KRyI1/view?usp=sharing) |
| `cleaned_ds_doService.csv` | 去識別化後的派工單資料集。因檔案過大，提供 [Google Drive 下載連結](https://drive.google.com/file/d/1YL82PAH9FkJwbzUo_LB3aAGyX5os70nq/view?usp=sharing) |
| `serviceDay_diff.csv`      | 以最後一次派工單為基準，統計 30 天、60 天、90 天內及以上的派工次數之資料集。 |
| `ds_doService_combo.csv`   | 將派工單 + 用戶服務 + 派工次數整合的合併資料集。                           |

---

### 2. `topMSO_visualization.ipynb`

- **功能**：以視覺化方式理解資料特性，針對可能與拆機/離退行為相關的因素進行探索。
- **重點分析項目**：
  - 派工次數與等待時間分布
  - 資料間的邏輯與統計相關性驗證

---

## 🧠 訓練與預測資料

本部分為筆者同事基於相同任務所展開的機器學習訓練與預測工作所產出之資料：

| 檔案名稱                    | 說明 |
|-----------------------------|------|
| `ai_train_data.csv`         | 訓練用資料集，使用 `,` 分隔。最後一欄為 `y`，其中 `1` 表示持續使用、`0` 表示離退。 |
| `欄位名稱.txt`              | 對應 `ai_train_data.csv` 的欄位名稱說明檔。 |
| `ai_pred_data_20250331.csv` | 2025 年 3 月 26 日產出的離退預測結果檔。 |

---

## 📦 原始資料說明

### 1. `用戶服務資料2.zip`
- **內容**：個資去識別化後的客戶基本服務資料
- **格式**：CSV 檔，欄位間以 `^` 符號分隔

### 2. `工單.zip`
- **內容**：2022~2024 年的用戶派工單資料
- **格式**：CSV 檔，欄位間以 `^` 符號分隔

---

## 📌 專案目標

本專案為 topMSO 計畫之一部分，透過資料整理與視覺分析，協助建立後續用於：
- 拆機/離退預測模型建構
- 客戶忠誠度分析
- 精準行銷與派工優化決策

---

## 📬 聯絡人

研究負責人：G13370001 羅文龍  
（如需資料或技術討論，請透過專案頁留言）

