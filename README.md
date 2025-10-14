# ft_bmshj2018_coco_relu

**COCO2017 (224×224 crops) を用いて CompressAI の `bmshj2018_factorized` の GDN/IGDN を置換し、  
再構成品質（L1 + (1 − MS-SSIM)）あるいは RD 損失でファインチューニングするレシピ**。  
既存の実行フローとディレクトリ構成（`scripts/` でデータ準備 → `src/` の学習スクリプト → `checkpoints/` 保存 → `recon/` に再構成 PNG 出力）を踏襲しています。

> ⚠️ 注意: GDN/IGDN は**可逆性やゲイン正規化**のために設計された層であり、単純置換すると RD 特性を崩す可能性があります。  
> 本レシピは **置換したブロックを部分的に再学習**することで画質を補償する実験的構成です。  
> 速度や NPU 互換性を検証するための基盤として活用してください。

---
## 0. ディレクトリ構造

```
ft_bmshj2018_coco_relu/
├── README.md                        # 本ファイル（手順・解説）
├── requirements.txt                 # 依存ライブラリ一覧（CompressAI, torch 等）
├── scripts/
│   ├── coco_prepare.py              # COCO 224x224 クロップ生成スクリプト
│   ├── export_onnx.py               # ONNX 形式エクスポートツール
│   └── utils/                       # 前処理補助スクリプト（任意）
├── src/
│   ├── train.py    # 統合学習スクリプト（ReLU/GDNishLite + QAT 対応）
│   ├── qat_utils.py                 # 軽量QAT（FakeQuant）モジュール
│   ├── replace_gdn_npu.py           # NPU向け GDNishLite 置換モジュール
│   ├── npu_blocks.py                # GDNishLiteEnc/Dec 実装
│   ├── model_utils.py               # GDN置換・局所FP32設定などの補助関数
│   ├── dataset_coco.py              # COCO224 Datasetクラス
│   ├── losses.py                    # 再構成損失・RD損失・PSNR/MS-SSIM計算
│   └── eval.py                      # 検証用スクリプト
├── checkpoints/                     # 学習済みモデルの保存（.pt）
├── recon/                           # 再構成画像出力ディレクトリ
│   ├── origin/                      # 元画像（val）
│   ├── e001/, e002/, ...            # 各エポックごとの再構成結果
└── onnx/                            # export_onnx.py で生成したONNXモデル
```

> **補足**  
> - 統合スクリプト `train.py` は ReLU / GDNishLite / QAT すべてに対応。  
> - `qat_utils.py` は FakeQuant 挿入・observer制御を担います。  
> - `replace_gdn_npu.py` と `npu_blocks.py` は NPU フレンドリー活性（ReLU6/HardSigmoid）ブロック用です。  
---

## 1. 環境セットアップ

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 2. データ準備（224×224 クロップ）

COCO2017 をダウンロード済み（`train2017/` と `val2017/` が存在）なら、次で 224×224 クロップ済みデータを生成します。

```bash
python -m scripts.coco_prepare \
  --coco_dir /path/to/coco2017 \
  --out_dir  /path/to/coco224 \
  --include_val true
```

---

## 3. 学習（統合スクリプト：ReLU / GDNishLite を切替）

学習用スクリプトは **`src/train.py`** に統合されています。  
`--act` で置換方式を選べます：

- `--act relu` …… GDN/IGDN を **ReLU** に置換（従来の置換）  
- `--act gdnish` …… GDN/IGDN を **GDNishLiteEnc/Dec**（NPU/ONNX フレンドリー）に置換

### 3.1 最小実行例（ReLU 置換・再構成損失）
```bash
python -m src.train \
  --act relu \
  --coco_dir /path/to/coco224 \
  --use_prepared true \
  --quality 8 --epochs 10 --batch_size 16 \
  --lr 1e-4 --alpha_l1 0.4 \
  --replace_parts encoder --train_scope replaced+hyper \
  --recon_every 2 --recon_count 16 \
  --save_dir ./checkpoints --recon_dir ./recon \
  --amp true --amp_dtype bf16 \
  --local_fp32 entropy \
  --loss_type recon \
  --sched cosine --optimizer adamw --weight_decay 1e-4
```

### 3.2 RD 最適化（ReLU 置換・bpp も最小化）
```bash
python -m src.train \
  --act relu \
  --coco_dir /path/to/coco224 \
  --use_prepared true \
  --quality 8 --epochs 10 --batch_size 16 \
  --lr 1e-4 --alpha_l1 0.4 \
  --replace_parts encoder --train_scope replaced+hyper \
  --recon_every 2 --recon_count 16 \
  --save_dir ./checkpoints --recon_dir ./recon \
  --amp true --amp_dtype bf16 \
  --local_fp32 entropy+decoder \
  --loss_type rd --lambda_bpp 0.01 \
  --sched onecycle --onecycle_pct_start 0.1 --optimizer adam
```

### 3.3 GDNishLite 置換（エンコーダのみ学習：`replaced+hyper`）
```bash
python -m src.train \
  --act gdnish \
  --coco_dir /path/to/coco224 \
  --use_prepared true \
  --quality 8 --epochs 10 --batch_size 16 \
  --lr 1e-4 --alpha_l1 0.4 \
  --replace_parts encoder --train_scope replaced+hyper \
  --enc_t 2.0 --enc_kdw 3 --enc_eca true --enc_residual true \
  --dec_k 3 --dec_gmin 0.5 --dec_gmax 2.0 --dec_residual true \
  --recon_every 2 --recon_count 16 \
  --save_dir ./checkpoints_npu --recon_dir ./recon_npu \
  --amp true --amp_dtype bf16 \
  --local_fp32 entropy \
  --loss_type rd --lambda_bpp 0.01 \
  --sched cosine --optimizer adamw --weight_decay 1e-4
```

### 3.4 GDNishLite 置換（**全層学習**：`train_scope all`）
```bash
python -m src.train \
  --act gdnish \
  --coco_dir /path/to/coco224 \
  --use_prepared true \
  --quality 8 --epochs 12 --batch_size 16 \
  --lr 1e-4 --alpha_l1 0.4 \
  --replace_parts all --train_scope all \
  --enc_t 2.0 --enc_kdw 3 --enc_eca true --enc_residual true \
  --dec_k 3 --dec_gmin 0.5 --dec_gmax 2.0 --dec_residual true \
  --recon_every 2 --recon_count 16 \
  --save_dir ./checkpoints_npu --recon_dir ./recon_npu \
  --amp true --amp_dtype bf16 \
  --local_fp32 entropy+decoder \
  --loss_type rd --lambda_bpp 0.01 \
  --sched onecycle --onecycle_pct_start 0.1 --optimizer adam
```

### 3.5 主な引数（更新点含む）
- **置換/学習範囲**
  - `--act {relu, gdnish}`：置換方式の選択
  - `--quality`：CompressAI の画質インデックス（0〜8）
  - `--replace_parts`：置換ブロック（`encoder` / `decoder` / `all`）
  - `--train_scope`：再学習範囲（`replaced` / `replaced+hyper` / `all`）
- **GDNishLite 特有の引数（`--act gdnish` のみ有効）**
  - Encoder: `--enc_t`（1×1拡張率, 2.0 推奨）、`--enc_kdw`、`--enc_eca {true,false}`、`--enc_residual {true,false}`
  - Decoder: `--dec_k`、`--dec_gmin`、`--dec_gmax`、`--dec_residual {true,false}`
- **損失**
  - `--loss_type {recon, rd}`：`recon = alpha*L1 + (1-alpha)*(1-MS-SSIM)`、`rd = recon + lambda_bpp*bpp`
  - `--alpha_l1`、`--lambda_bpp`
- **AMP / 数値安定化**
  - `--amp {true,false}`、`--amp_dtype {fp16,bf16}`（bf16 推奨）
  - `--local_fp32 {none, entropy, entropy+decoder, all_normexp, custom}`（危険演算のみ FP32 実行）
- **学習率最適化**
  - Optimizer: `adam` / `adamw`（`--weight_decay` あり）
  - Scheduler: `none` / `cosine` / `onecycle` / `plateau`
  - `--lr_warmup_steps`：線形ウォームアップ（OneCycle 以外と併用可）
- **安定化その他**
  - `--max_grad_norm`、`--overflow_check`、`--overflow_tol`
- **再構成保存**
  - `--recon_every`、`--recon_count`（`./recon/origin/NNNNN.png` に元画像も保存・番号対応）
- **W&B と EB-aux（補助最適化）**
  - `--wandb {true,false}`：Weights & Biases
  - `--eb_aux {true,false}`, `--eb_aux_lr`：**EntropyBottleneck の補助損失**を別オプティマイザで最適化可能（学習後に `model.update()` で CDF 更新）

> **変更点（重要）**  
> - 旧オプション **`--amp_warmup_steps` は削除**。AMP 安定化は **bf16 + 局所FP32 + LRウォームアップ + 勾配クリップ**の組合せで対応。  
> - 損失は常に **FP32** で計算、MS-SSIM 入力は **[0,1] に clamp**。

### 実装上の工夫（更新）
- ReLU 系は **`inplace=False`** で勾配を安定化
- 順伝播は AMP（bf16 推奨）、損失は FP32 にキャスト
- **局所FP32**（entropy/decoder 周辺）で危険演算のみ AMP 無効化
- Cosine / OneCycle / Plateau などのスケジューラ対応（LR ウォームアップ併用）
- 勾配クリッピング、forward 出力の NaN/Inf/過大値チェック

---

## 3'. 軽量 QAT（FakeQuant のみ / AMP 非推奨）

**目的**：量子化耐性の改善（学習時に Q/DQ の影響を模擬）  
**実装**：`src/qat_utils.py` の FakeQuant ラッパを Conv/Linear に in-place で挿入。  
EntropyBottleneck / Hyperprior 近傍は自動除外されます。

---

### 実行例（ReLU + RD 学習 + QAT）

```bash
python -m src.train \
  --act relu \
  --replace_parts encoder --train_scope all \
  --qat true --qat_scope encoder --qat_exclude_entropy true \
  --qat_calib_steps 2000 --qat_freeze_after 8000 \
  --coco_dir /path/to/coco224 --use_prepared true \
  --quality 8 --epochs 10 --batch_size 16 \
  --lr 1e-4 --alpha_l1 0.4 \
  --loss_type rd --lambda_bpp 0.01 \
  --sched cosine --optimizer adamw --weight_decay 1e-4
```

---

### 🔸 QATパラメータ詳細

| パラメータ | 型 / 既定値 | 説明・挙動 |
|-------------|-------------|-------------|
| `--qat` | bool (`false`) | QATを有効化（FakeQuant挿入） |
| `--qat_scope` | {"encoder","decoder","all"} | どのブロックに挿入するか |
| `--qat_exclude_entropy` | bool (`true`) | EntropyBottleneck/Hyperprior近傍を除外 |
| `--qat_calib_steps` | int (`2000`) | 観測器（observer）のレンジ学習期間 |
| `--qat_freeze_after` | int (`8000`) | このステップ以降、観測器を固定（scale確定） |
| `--qat_act_observer` | {"ema","minmax"} | 活性観測方式（EMA推奨） |
| `--qat_w_per_channel` | bool (`true`) | Conv重みをper-channel量子化 |
| `--qat_disable_observer_step` | int (`<0`) | 指定step以降observer無効化（任意） |
| `--qat_eval_fakequant` | bool (`true`) | 検証時もFakeQuantを有効化 |
| `--amp` | bool (`false` 推奨) | QATとAMPの併用は非推奨（NaNを招きやすい） |

---

### 🔸 用語補足：「Quant値とは？」

量子化（Quantization）では、連続値（float）を整数（int8など）に離散化します。  
このとき得られる整数値を **Quant値（量子化値）** と呼びます。  
QATでは順伝播時に `x → Quant(x)` を模擬し、逆伝播では  
量子化誤差を無視して勾配を流す（Straight-Through Estimator: STE）ことで、  
モデルが量子化誤差を含む出力にも頑健になるよう訓練されます。  
FakeQuant層は内部的にはfloat演算ですが、出力レンジをint8相当範囲に制限します。

---

### 🔸 QATのフェーズ動作（詳細）

| フェーズ | ステップ範囲 | 挙動 |
|-----------|---------------|------|
| **Calibration** | 0〜`qat_calib_steps` | min/maxをEMAで推定（スケール観測中） |
| **Freeze（固定スケール学習）** | `qat_calib_steps`〜`qat_freeze_after` | 観測器をfreezeし、固定スケールでFakeQuantを継続 |
| **Eval-ready（推論模擬段階）** | `> qat_freeze_after` | 観測器を完全固定し、量子化スケール下で安定学習を継続 |

> ⚙️ **補足**：  
> `qat_freeze_after` 以降も Conv/Linear 層は学習を継続します。  
> 凍結されるのは「観測器のスケール範囲」のみで、勾配更新は停止しません。

---

### 🔸 freeze と disable の違い

| 状態 | Observer更新 | FakeQuant適用 | Conv/Linear学習 | 説明 |
|------|----------------|----------------|-------------------|------|
| **freeze** | ❌ しない | ✅ 継続する | ✅ 継続する | 量子化スケールを固定して訓練（本実装のデフォルト） |
| **disable** | ❌ しない | ❌ 停止する | ✅ 継続する | 量子化模擬を完全に無効化（float挙動に戻す） |

> 本実装の `qat_utils.py` は `freeze` のみを自動制御します。  
> 推論時は `eval()` 状態でFakeQuantが残る設計になっており、  
> 実際のINT8挙動に近い再現を維持します。

---

### 🔸 QAT 安定化 Tips

- AMPは**無効推奨**（`--amp false`）
- 学習率は**1e-4以下**が安定
- `entropy+decoder` で局所FP32を有効化（`--local_fp32 entropy+decoder`）
- まずは **encoderのみQAT** から始めると安定
- bf16よりもfp32モードでの訓練が安定
- 長めのウォームアップ (`--lr_warmup_steps 1000`) 併用可
- 学習後は `model.update()` を実行してCDFテーブルを再構築


## 4.
## 4. 評価

```bash
python -m src.eval \
  --coco_dir /path/to/coco224 \
  --use_prepared true \
  --checkpoint ./checkpoints/best_msssim.pt
```

出力: 平均 **PSNR** / **MS-SSIM**

---

## 5. 仕組みの概要

- `compressai.zoo.bmshj2018_factorized(quality, pretrained=True)` をロード
- `--act` に応じて **GDN/IGDN を ReLU または GDNishLiteEnc/Dec に置換**
- `set_trainable_parts(model, scope)` により学習対象を制御  
  - 例: `--replace_parts decoder --train_scope replaced+hyper` → デコーダ置換＋hyperprior を学習
- **損失**:  
  - `recon`: `alpha * L1 + (1 - alpha) * (1 - MS-SSIM)`（既定 `alpha=0.4`）  
  - `rd`   : 上記に `+ lambda_bpp * bpp`（`likelihoods` から bpp を算出）
- AMP+FP32 ハイブリッド計算 + 局所FP32 で安定化
- 再構成画像は `recon/` に保存し、W&B にはグリッド形式でログ

---

## 6. ONNX 形式でのエクスポート（追加）

`bmshj2018_factorized`（置換後）を **full / encoder / decoder** 単位で ONNX へ出力できます。

```bash
# フルモデル
python -m scripts.export_onnx \
  --ckpt checkpoints/last.pt \
  --part full \
  --out onnx/full.onnx \
  --input_size 224

# エンコーダのみ
python -m scripts.export_onnx \
  --ckpt checkpoints/last.pt \
  --part encoder \
  --out onnx/enc.onnx \
  --input_size 224

# デコーダのみ（latent チャネルは自動推定）
python -m scripts.export_onnx \
  --ckpt checkpoints/last.pt \
  --part decoder \
  --out onnx/dec.onnx \
  --input_size 224
```

オプション:
- `--replace_parts {encoder,decoder,all}`：置換対象（エクスポート前に適用）
- `--dynamic`：動的軸（H/W）を ONNX に付与
- `--opset`：既定 13

---

## 7. 注意点 / Tips（更新）
- 置換直後は不安定になりやすいので **学習率 1e-4 以下**を推奨
- `train_scope=replaced+hyper` が安定しやすい
- NaN/Inf が発生した場合は
  - **BF16 AMP** を優先（`--amp_dtype bf16`）
  - **局所FP32** を強める（`--local_fp32 entropy+decoder` や `all_normexp`、`custom`）
  - **学習率を下げる**（例: `--lr 5e-5`）
  - **勾配クリッピング** を強める（例: `--max_grad_norm 0.5`）
  - **LRウォームアップ** を長くする（`--lr_warmup_steps`）
- 旧オプション `--amp_warmup_steps` は **削除済み**。代替は上記で対応。

---

## 8. ライセンスとクレジット

- 本レシピは研究・実験目的での利用を想定しています。  
- 元モデル・コードは [CompressAI](https://github.com/InterDigitalInc/CompressAI) の `bmshj2018_factorized` を利用しています。

---

## 9. モデル構造（初学者向けやさしい解説）

### 9.1 どんなモデル？
`bmshj2018_factorized` は**学習ベース画像圧縮**モデルです。  
画像 `x` を **エンコーダ**で低次元表現 `y`（潜在表現）に圧縮し、**量子化**→**エントロピー符号化**でビット列にします。  
復元側は **エントロピー復号**→**デコーダ**で `x_hat` を再構成します。

```
x ──Encoder──> y ──Quantize──> ẏ ──EntropyCoder──> bitstream
x_hat <──Decoder──  ẏ  <──EntropyDecoder<── bitstream
```

### 9.2 GDN と置換ブロックの違い
- **GDN/IGDN**: 各チャネル同士の関係を正規化し、**歪みとレートの両立**に寄与（圧縮向けに設計）。  
- **ReLU 置換**: シンプルで高速・NPU 互換性が高いが、**分布の形**が変わりレート（bpp）が悪化することがあるため、**置換後に再学習**して性能を取り戻す。  
- **GDNishLite（本リポ自作）**: ReLU6 / HardSigmoid と DepthwiseConv, ECA を活用し、**量子化耐性とNPU互換性**を保ちながら GDN の“抑制/増幅”を近似。

### 9.3 本リポジトリの工夫
- `--replace_parts` で置換範囲を選択（エンコーダ／デコーダ／両方）。
- `--train_scope` で **hyperprior / entropy_bottleneck** も一緒に再学習可能。  
  置換による表現分布の変化を**確率モデル側**でも吸収させる狙い。

---

## 10. 損失関数の見方（初学者向け）

### 10.1 再構成損失
- **L1**：平均絶対誤差（小さいほど良い）  
- **MS-SSIM**：視覚的類似度（1 に近いほど良い）  
- 本レシピの**再構成損失**は `alpha * L1 + (1 - alpha) * (1 - MS-SSIM)`

### 10.2 RD 損失と bpp
- **bpp**：1 画素あたりのビット数（小さいほど高圧縮）  
- **RD 損失**：`loss = recon + λ * bpp`（`λ` が大きいほど圧縮率重視）

### 10.3 おすすめの見方
1. `RD` が**低下**（総合目標）  
2. `BPP` が**低下**（圧縮率改善）  
3. `1-MSSSIM` / `L1` が**上昇していない**（画質悪化の監視）  
4. `PSNR`, `MS-SSIM` が**維持/改善**

### 10.4 典型的なトレードオフ
- `λ` を上げる → `BPP` は下がるが画質低下しやすい  
- `α` を上げる → L1 重視、テクスチャの粒状感が変化することあり

### 10.5 再構成画像と元画像の対応
- `./recon/<tag>/NNNNN.png`（復元）と `./recon/origin/NNNNN.png`（元画像）は **番号で 1:1 対応**

---

## 11. 学習時の最適化手法（初学者向け）

### 11.1 数値安定化の基本セット
- **AMP を BF16 で使用**（`--amp true --amp_dtype bf16`）  
- **損失は FP32** で計算  
- **局所FP32**：`--local_fp32 entropy+decoder` 推奨  
- **勾配クリップ**：`--max_grad_norm 1.0`（デフォルト）

### 11.2 学習率スケジューリング
- **CosineAnnealingLR**：`--sched cosine`  
- **OneCycleLR**：`--sched onecycle --onecycle_pct_start 0.1`  
- **ReduceLROnPlateau**：`--sched plateau`

### 11.3 つまずいた時の対処
- 収束が遅い/発散する → `lr` を下げる、`--lr_warmup_steps` 増やす、`--local_fp32` を広げる、`train_scope` を `replaced+hyper` に
- 画質は良いが bpp が高い → `--lambda_bpp` を上げる
- bpp は下がるが画質が悪い → `--lambda_bpp` を下げる or `alpha_l1` を下げる

### 11.4 ログ活用（W&B）
- `--wandb true` で有効化、主要メトリクスと再構成グリッドを記録

---

## 12. 自作活性化ブロックを追加して学習する手順

1. **ブロック実装**
   - 例：`src/my_blocks.py` に自作ブロック（ONNX/NPU 互換演算で構成）を実装  
   - 初期化は **恒等近傍**に寄せる（1×1 は単位行列、DW は中心に極小値など）

2. **置換関数の実装**
   - 例：`src/replace_gdn_myact.py` に  
     `replace_gdn_with_myact(model, mode, **kwargs)` を実装  
   - `model.g_a`（エンコーダ）/`model.g_s`（デコーダ）配下で GDN/IGDN を検出し、自作ブロックに差し替え

3. **統合スクリプトへ選択肢追加**
   - `src/train.py` の `--act` に `myact` を追加し、モデル構築部で分岐：  
     `if args.act == "myact": model = replace_gdn_with_myact(model, mode=args.replace_parts, ...)`  
   - 必要なハイパラ（`--myact_*`）を `get_args()` に追加

4. **学習実行**
   - 例：
   ```bash
   python -m src.train \
     --act myact \
     --coco_dir /path/to/coco224 \
     --use_prepared true \
     --quality 8 --epochs 10 --batch_size 16 \
     --lr 1e-4 --alpha_l1 0.4 \
     --replace_parts all --train_scope replaced+hyper \
     --recon_every 2 --recon_count 16 \
     --save_dir ./checkpoints_my --recon_dir ./recon_my \
     --amp true --amp_dtype bf16 \
     --local_fp32 entropy+decoder \
     --loss_type rd --lambda_bpp 0.01 \
     --sched cosine --optimizer adamw --weight_decay 1e-4
   ```

5. **QAT を行う場合**
   - FakeQuant（per-channel weight、symmetric）を Conv 前後に挿入。  
   - HardSigmoid / ReLU6 は INT8 量子化に親和。

---

以上。初心者は **「3. 学習」→「10. 損失の見方」→「11. 最適化」** の順で触るのがおすすめです。
