# HiDF Deepfake Detection with Vision Transformer

本專案使用 Vision Transformer（ViT）進行 Deepfake 人臉影像二元分類，涵蓋 HiDF 資料前處理、模型訓練、自建資料整理、Retrain 與最終測試的完整流程。影像先經 YOLO 人臉偵測與裁切，再輸入 `vit_base_patch16_224` 進行 Real／Fake 分類。

在 HiDF 測試集上達到 90.42% Accuracy，經自建資料 Retrain 後於另一組實際情境影像上達到 85.96% Accuracy。完整的研究背景、方法與結果分析請參考 [`docs/project_report.pdf`](docs/project_report.pdf)。

> 本儲存庫為大學三年級專題的第一版實作紀錄。自建資料集已無法還原當時的切分，報告中的實驗結果無法重新產生，程式碼亦與當時版本存在部分差異，詳見〈已知限制與差異〉。

## 延伸應用

本專案訓練出的 ViT 權重已整合至
[ViTalyzer](https://github.com/YOUCI940805/Discord-Deepfake-Detection-Bot)，
一個結合 YOLO 人臉偵測的 Discord 檢測機器人，並提供 Occlusion 關注區域視覺化。
該專案示範了本模型的實際部署方式，也凸顯出訓練情境與真實輸入的落差：
使用者上傳的影像會經過平台壓縮與縮放，人臉角度與拍攝條件也不受控，
這些都與訓練時使用的裁切人臉不同，模型在這些條件下的表現尚未經系統性評估。

## 實驗結果

### 各階段模型表現

| 模型階段 | 測試資料 | 測試數量 | Accuracy | Weighted F1 |
|---|---|---|---|---|
| 主訓練 | HiDF 測試集 | 501 | 0.9042 | 0.9038 |
| Retrain 後 | 額外實際情境資料 | 399 | 0.8596 | 0.8604 |

兩個階段使用的測試資料來源與類別分布不同，因此上述數值**不能直接用來判定 Retrain 使模型表現提升或下降**。主訓練結果反映模型在 HiDF 同來源測試集上的分類能力，Retrain 後結果則反映模型在另一組實際情境影像上的表現。

### 主訓練模型（HiDF 測試集，501 張）

| 類別 | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| Fake | 0.9591 | 0.8440 | 0.8979 | 250 |
| Real | 0.8612 | 0.9641 | 0.9098 | 251 |
| Accuracy | — | — | 0.9042 | 501 |
| Macro avg | 0.9102 | 0.9041 | 0.9038 | 501 |

混淆矩陣：

| 真實＼預測 | Fake | Real |
|---|---|---|
| **Fake** | 211 | 39 |
| **Real** | 9 | 242 |

模型在 HiDF 測試集對 Real 類別的辨識較穩定，但仍有 39 張 Fake 被誤判為 Real。

### Retrain 後模型（實際情境資料，399 張）

| 類別 | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| Fake | 0.9042 | 0.8839 | 0.8939 | 267 |
| Real | 0.7754 | 0.8106 | 0.7926 | 132 |
| Accuracy | — | — | 0.8596 | 399 |
| Macro avg | 0.8398 | 0.8473 | 0.8433 | 399 |

混淆矩陣：

| 真實＼預測 | Fake | Real |
|---|---|---|
| **Fake** | 236 | 31 |
| **Real** | 25 | 107 |

模型對 Fake 類別仍具一定辨識能力，但部分 Real 影像可能因光線、壓縮、拍攝品質或裁切差異而被誤判為 Fake。此測試集的類別分布不平衡，Accuracy 需搭配各類別 F1-score 一併解讀。

## 研究方法

### 研究流程

| 階段 | 輸入 | 主要處理 | 輸出 |
|---|---|---|---|
| 資料整理 | HiDF／自建影像 | 類別平衡與資料切分 | Train／Validation／Test |
| 人臉前處理 | 原始影像 | YOLO 偵測、裁切、尺寸統一 | 人臉影像 |
| 模型訓練 | 人臉影像 | ViT 主訓練與 Retrain | 最佳模型權重 |
| 模型評估 | 獨立測試資料 | 分類預測與指標計算 | 報表與混淆矩陣 |

### 資料組成

| 階段 | 資料來源 | 用途 | 數量 | 切分方式 |
|---|---|---|---|---|
| 主訓練 | HiDF | 訓練、驗證、測試 | 5,000 | 8:1:1 |
| Retrain | 自行蒐集資料 | 追加訓練、驗證 | 300 | 約 80%／20% |
| 額外測試 | 實際情境影像 | 獨立測試 | 399 | 不參與訓練 |

自行蒐集資料中的 Real 類別以手機拍攝的原況照片為主，Fake 類別則由 X、Facebook 等公開網路平台蒐集。上述網路影像僅用於課程研究與模型測試，不在本儲存庫中重新散布。

### 主訓練設定

| 項目 | 設定 | 項目 | 設定 |
|---|---|---|---|
| 模型 | vit_base_patch16_224 | Epoch | 30 |
| Batch size | 8 | Image size | 224×224 |
| Learning rate | 1×10⁻⁴ | Weight decay | 0.01 |
| Optimizer | AdamW | Loss | CrossEntropyLoss |
| Seed | 42 | 輸出權重 | best_vit_model.pth |

### Retrain 設定

| 項目 | 設定 | 項目 | 設定 |
|---|---|---|---|
| 基礎權重 | best_vit_model.pth | Epoch | 12 |
| Batch size | 8 | Image size | 224×224 |
| Learning rate | 3×10⁻⁵ | Weight decay | 0.0001 |
| Optimizer | AdamW | Loss | CrossEntropyLoss |
| 輸出權重 | best_vit_model_retrain.pth | 資料用途 | Train／Validation |

Retrain 並非從隨機權重重新訓練，而是載入主訓練完成的 `best_vit_model.pth` 進行追加訓練，目的是讓模型接觸不同於 HiDF 的影像來源與拍攝條件。

## Notebook 說明

執行順序：

1. `01_prepare_hidf_dataset.ipynb`
2. `02_train_vit_hidf.ipynb`
3. `03_prepare_retrain_dataset.ipynb`
4. `04_retrain_and_test_vit.ipynb`

### 01_prepare_hidf_dataset.ipynb

- 讀取 HiDF 原始 Real 與 Fake 圖片
- 使用 `metadata.csv` 取得人物與族群資訊
- 依族群平衡抽樣 Real 與 Fake 圖片
- 切分 Train、Validation 與 Test
- 使用 YOLO 偵測並裁切最大人臉
- 產生 `dataset_vit`

YOLO 找不到人臉的圖片會被跳過並寫入 `hidf_failed_images.csv`，因此各 split 實際輸出的張數會少於依比例計算的數量。

### 02_train_vit_hidf.ipynb

- 載入 `dataset_vit`
- 建立 Vision Transformer 模型（ImageNet 預訓練）
- 執行模型訓練與驗證，訓練增強僅使用水平翻轉
- 依驗證集 Accuracy 儲存最佳模型權重
- 在 Test 資料集上進行評估

### 03_prepare_retrain_dataset.ipynb

- 讀取自行蒐集的 Real 與 Fake 圖片
- 依類別分層切分 Train、Validation 與 Test
- 寫出切分紀錄 `retrain_split_manifest.csv`
- 使用 YOLO 偵測並裁切人臉
- 產生 `dataset_vit_retrain`

原始圖片只讀取，不移動也不刪除。

### 04_retrain_and_test_vit.ipynb

- 載入第一次訓練完成的 ViT 模型權重
- 使用自建資料進行追加訓練
- 套用資料增強提高模型對影像變化的適應能力
- 依驗證集 Accuracy 儲存最佳模型權重
- 在 Test 資料集上進行最終評估

測試圖片已由 `03` 完成裁臉，此階段不會重複執行 YOLO。

## 已知限制與差異

### 本儲存庫的性質

本儲存庫的程式碼是專題完成後重新整理的版本，與當時實際執行的程式碼不完全相同。訓練完成的模型權重仍保留，但因自建資料集包含實際人物的照片，基於肖像權與隱私考量，資料與權重均不公開。自建資料集在專題結束後持續調整，已無法還原當時的資料切分，因此上述實驗結果無法重新產生。報告中使用的獨立測試流程（399 張實際情境影像）未收錄於本儲存庫。

### 程式碼與報告的差異

Notebook 中的參數已對齊 `docs/project_report.pdf` 記錄的實驗設定，但部分流程仍存在以下差異。這些差異均予以保留，不另行修改程式碼，以避免產生從未實際執行過的內容。

| 項目 | 報告記錄 | 本儲存庫程式碼 |
|---|---|---|
| Retrain 資料切分 | 300 張分為 Train／Validation（約 80%／20%） | `03` 切分為 Train／Validation／Test（8:1:1） |
| 最終測試資料 | 399 張獨立實際情境影像 | `04` 於 `dataset_vit_retrain/test` 評估，該流程未收錄 |
| Retrain 資料增強 | 提及仿射變換、銳利度調整 | `04` 實作為 RandomResizedCrop、水平翻轉、ColorJitter、GaussianBlur、JPEG 壓縮、白平衡 |
| 主訓練測試集張數 | 501 張（Fake 250／Real 251） | 依現行切分邏輯計算應為 500 張 |

### 方法上的限制

- 兩階段模型未使用同一組獨立測試集，因此無法直接證明 Retrain 的改善效果。
- 額外測試資料的 Real／Fake 類別數量不平衡，Accuracy 需搭配各類別 F1-score 解讀。
- 網路 Fake 影像來源分散，生成方法與平台壓縮方式未完整標註。
- 研究範圍限於單張人臉影像，結果不能直接推論至影片型 Deepfake。
- 資料切分在圖片層級進行 shuffle。HiDF 的 Fake 影像檔名格式為 `{face_id}_{body_id}`，同一個 `face_id` 可能出現在多張 Fake 影像中，因此 Train 與 Test 之間可能存在部分身分重疊。

## 未來方向

- 重新建立具完整來源紀錄與固定切分的資料集，以提高實驗可再現性。
- 在同一獨立測試集上比較 Retrain 前後模型，量化追加訓練的實際效果。
- 加入跨資料集測試，評估模型面對不同 Deepfake 生成方法時的泛化能力。
- 進一步比較不同人臉框擴張比例與裁切策略對分類結果的影響。
- 擴充至影片資料，納入時間序列與影格一致性資訊。

## 執行方式

### 安裝環境

建議使用：

- Python 3.10 或 3.11
- NVIDIA GPU
- CUDA 相容版本的 PyTorch
- Jupyter Notebook 或 JupyterLab

安裝所需套件：

```bash
pip install -r requirements.txt
```

如果需要使用 NVIDIA GPU，請先依照 PyTorch 官方網站提供的指令安裝與 CUDA 相容的 PyTorch 版本。

`requirements.txt` 依當時的開發環境列出所需套件，未指定版本號。在不同環境下執行時，可能需要自行處理套件版本相容問題。

### 資料準備

本專案不提供 HiDF 原始資料、`metadata.csv`、YOLO 人臉偵測權重及訓練完成的模型權重。

請自行取得 HiDF 資料，並整理成：

```text
dataset/
├── real/
│   └── ...
└── fake/
    └── ...
```

將 HiDF 提供的 Metadata 檔案放在專案根目錄：

```text
metadata.csv
```

另外需要準備 YOLO 人臉偵測模型：

```text
yolov11n-face.pt
```

### Retrain 原始資料格式

自行蒐集的 Retrain 圖片請放置於：

```text
retrain_raw/
├── real/
│   └── ...
└── fake/
    └── ...
```

執行 `03_prepare_retrain_dataset.ipynb` 後，程式會產生：

```text
dataset_vit_retrain/
├── train/
│   ├── fake/
│   └── real/
├── val/
│   ├── fake/
│   └── real/
└── test/
    ├── fake/
    └── real/
```

### 專案檔案結構

```text
project/
├── 01_prepare_hidf_dataset.ipynb
├── 02_train_vit_hidf.ipynb
├── 03_prepare_retrain_dataset.ipynb
├── 04_retrain_and_test_vit.ipynb
├── docs/
│   └── project_report.pdf
├── README.md
├── requirements.txt
├── metadata.csv
├── yolov11n-face.pt
├── dataset/
├── dataset_vit/
├── retrain_raw/
└── dataset_vit_retrain/
```

其中資料集、模型權重、Metadata 與 YOLO 權重不會上傳至 GitHub。

### 注意事項

- 執行前請確認各 Notebook 中的路徑設定。
- GitHub 版本預設使用相對路徑。
- `01` 的 `OVERWRITE_OUTPUT` 預設為 `False`，`03` 預設為 `True`。後者會在執行時直接刪除既有的輸出資料夾。
- 模型訓練時間會依 GPU、資料量及訓練參數而有所不同。

## 第三方資料與模型來源

### HiDF 資料集

本專案使用 HiDF（Human-Indistinguishable Deepfake Dataset）進行模型訓練與評估。

本儲存庫不包含或重新散布 HiDF 的影像、影片及 Metadata。使用者須自行從 HiDF 官方來源取得相關檔案，並遵守資料集的 Creative Commons Attribution-NonCommercial 4.0 International（CC BY-NC 4.0）授權條款。

- HiDF 官方專案：https://github.com/DSAIL-SKKU/HiDF
- HiDF 資料集：https://zenodo.org/records/16140829
- HiDF 論文：https://doi.org/10.1145/3711896.3737399
- 資料集授權：https://creativecommons.org/licenses/by-nc/4.0/

使用本專案進行研究時，請引用：

Chaewon Kang, Seoyoon Jeong, Jonghyun Lee, Daejin Choi,
Simon S. Woo, and Jinyoung Han.
"HiDF: A Human-Indistinguishable Deepfake Dataset."
Proceedings of the 31st ACM SIGKDD Conference on Knowledge
Discovery and Data Mining, 2025.
https://doi.org/10.1145/3711896.3737399

### YOLOv11n-Face

本專案使用 YOLOv11n-Face 進行人臉偵測與裁切，並透過 Ultralytics 套件載入及執行模型。

本儲存庫不包含或重新散布 `yolov11n-face.pt` 權重。使用者須自行從原始專案取得權重，並遵守該專案及相關套件的授權條款。

- YOLO-Face 模型來源：https://github.com/akanametov/yolo-face
- Ultralytics：https://github.com/ultralytics/ultralytics

YOLO-Face 及 Ultralytics 均為其各自作者或權利人的專案。本專案與上述專案沒有隸屬或官方合作關係。

## License

本專案採用 [MIT License](LICENSE) 授權。
