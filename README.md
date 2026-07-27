# HiDF Deepfake Detection with Vision Transformer

本專案使用 Vision Transformer（ViT）進行 Deepfake 人臉影像二分類，包含 HiDF 資料前處理、模型訓練、自建資料整理、Retrain 與最終測試。

## 專案流程

請依照以下順序執行：

1. `01_prepare_hidf_dataset.ipynb`
2. `02_train_vit_hidf.ipynb`
3. `03_prepare_retrain_dataset.ipynb`
4. `04_retrain_and_test_vit.ipynb`

## Notebook 說明

### 01_prepare_hidf_dataset.ipynb

主要功能：

- 讀取 HiDF 原始 Real 與 Fake 圖片
- 使用 `metadata.csv` 取得人物與族群資訊
- 平衡抽樣 Real 與 Fake 圖片
- 切分 Train、Validation 與 Test
- 使用 YOLO 偵測並裁切人臉
- 產生 `dataset_vit`

### 02_train_vit_hidf.ipynb

主要功能：

- 載入 `dataset_vit`
- 建立 Vision Transformer 模型
- 執行模型訓練與驗證
- 儲存最佳模型權重
- 在 Test 資料集上進行評估

### 03_prepare_retrain_dataset.ipynb

主要功能：

- 讀取自行蒐集的 Real 與 Fake 圖片
- 切分 Train、Validation 與 Test
- 使用 YOLO 偵測並裁切人臉
- 產生 `dataset_vit_retrain`

### 04_retrain_and_test_vit.ipynb

本階段載入已使用 HiDF 完成初始訓練的 ViT 模型，並使用自行建立的 Real／Fake 人臉資料進行 Retrain。

Retrain 的目的，是讓模型進一步適應與 HiDF 不同的影像來源與拍攝條件，觀察模型經過追加訓練後，在自建資料上的辨識表現。

主要功能：

- 載入第一次訓練完成的 ViT 模型權重
- 使用自建資料進行追加訓練
- 使用資料增強提高模型對影像變化的適應能力
- 根據驗證集 Macro F1 儲存最佳模型
- 在獨立 Test 資料集上進行最終評估

## 安裝環境

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

## 資料準備

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

## Retrain 原始資料格式

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

## 專案檔案結構

```text
project/
├── 01_prepare_hidf_dataset.ipynb
├── 02_train_vit_hidf.ipynb
├── 03_prepare_retrain_dataset.ipynb
├── 04_retrain_and_test_vit.ipynb
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

## 注意事項

- 執行前請確認各 Notebook 中的路徑設定。
- GitHub 版本預設使用相對路徑。
- 重新建立資料集時，請注意 `OVERWRITE_OUTPUT` 設定。
- 啟用覆寫可能會刪除原本已產生的資料夾。
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
“HiDF: A Human-Indistinguishable Deepfake Dataset.”
Proceedings of the 31st ACM SIGKDD Conference on Knowledge
Discovery and Data Mining, 2025.
https://doi.org/10.1145/3711896.3737399

### YOLOv11n-Face

本專案使用 YOLOv11n-Face 進行人臉偵測與裁切，並透過 Ultralytics 套件載入及執行模型。

本儲存庫不包含或重新散布 `yolov11n-face.pt` 權重。使用者須自行從原始專案取得權重，並遵守該專案及相關套件的授權條款。

- YOLO-Face 模型來源：https://github.com/akanametov/yolo-face
- Ultralytics：https://github.com/ultralytics/ultralytics

YOLO-Face 及 Ultralytics 均為其各自作者或權利人的專案。本專案與上述專案沒有隸屬或官方合作關係。