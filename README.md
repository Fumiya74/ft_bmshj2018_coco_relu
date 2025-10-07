# ft_bmshj2018_coco_relu

**COCO2017 (224×224 crops) を用いて CompressAI の `bmshj2018_factorized` の GDN/IGDN を ReLU に置換し、  
再構成品質（L1 + (1 − MS-SSIM)）あるいは RD 損失でファインチューニングするレシピ**。  
既存の実行フローとディレクトリ構成（`scripts/` でデータ準備 → `src/` の学習スクリプト → `checkpoints/` 保存 → `recon/` に再構成 PNG 出力）を踏襲しています。

> ⚠️ 注意: GDN/IGDN は**可逆性やゲイン正規化**のために設計された層であり、ReLU に単純置換すると RD 特性を崩す可能性があります。  
> 本レシピは **置換したブロックを部分的に再学習**することで画質を補償する実験的構成です。  
> 速度や NPU 互換性を検証するための基盤として活用してください。

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
python -m scripts.coco_prepare   --coco_dir /path/to/coco2017   --out_dir  /path/to/coco224   --include_val true
```

---

## 3. 学習（ReLU 置換モデルのファインチューニング）

### 最小実行例（再構成損失のみ）
```bash
python -m src.train_finetune_relu   --coco_dir /path/to/coco224   --use_prepared true   --quality 8 --epochs 10 --batch_size 16   --lr 1e-4 --alpha_l1 0.4   --replace_parts encoder --train_scope replaced+hyper    --recon_every 2 --recon_count 16   --save_dir ./checkpoints --recon_dir ./recon   --amp true --amp_dtype bf16   --local_fp32 entropy   --loss_type recon   --sched cosine --optimizer adamw --weight_decay 1e-4
```

### RD 最適化（bpp も最小化）
```bash
python -m src.train_finetune_relu   --coco_dir /path/to/coco224   --use_prepared true   --quality 8 --epochs 10 --batch_size 16   --lr 1e-4 --alpha_l1 0.4   --replace_parts encoder --train_scope replaced+hyper   --recon_every 2 --recon_count 16   --save_dir ./checkpoints --recon_dir ./recon   --amp true --amp_dtype bf16   --local_fp32 entropy+decoder   --loss_type rd --lambda_bpp 0.01   --sched onecycle --onecycle_pct_start 0.1 --optimizer adam
```

### 主な引数（更新点含む）
- **モデル置換/学習範囲**
  - `--quality`: CompressAI の画質インデックス（0〜8）
  - `--replace_parts`: GDN→ReLU に置換するブロック（`encoder` / `decoder` / `all`）
  - `--train_scope`: 再学習範囲  
    - `replaced`: 置換したブロックのみ  
    - `replaced+hyper`: 置換ブロック＋hyperprior＋entropy_bottleneck  
    - `all`: 全層学習
- **損失**
  - `--loss_type {recon, rd}`:  
    - `recon` = `alpha*L1 + (1-alpha)*(1-MS-SSIM)`  
    - `rd` = 上記に **`+ lambda_bpp * bpp`** を加えた RD 損失
  - `--alpha_l1`: L1 の重み（デフォルト 0.4）
  - `--lambda_bpp`: RD のレート係数（bpp に掛ける、デフォルト 0.01）
- **AMP / 数値安定化**
  - `--amp {true,false}`: AMP 有効/無効（デフォルト true）
  - `--amp_dtype {fp16,bf16}`: AMP の dtype（デフォルト bf16 推奨）
  - `--local_fp32 {none, entropy, entropy+decoder, all_normexp, custom}`:  
    危険演算のみ **局所FP32** で実行して安定化  
    - `entropy`: `EntropyBottleneck` / `GaussianConditional` をFP32  
    - `entropy+decoder`: 上記＋ Softmax/LogSoftmax/Exp/Log もFP32  
    - `all_normexp`: Norm/Exp/Log 系を広くFP32  
    - `custom`: クラス名部分一致をカンマ区切りで指定（例: `--local_fp32 custom --local_fp32_custom "EntropyBottleneck,Softmax"`）
- **学習率最適化**
  - Optimizer: `--optimizer {adam, adamw}`（`--weight_decay` あり）
  - Scheduler: `--sched {none, cosine, onecycle, plateau}`
    - `cosine`/`plateau` は **epoch 単位**更新
    - `onecycle` は **step 単位**更新（`--onecycle_pct_start`）
  - `--lr_warmup_steps`: **線形ウォームアップ**のステップ数（OneCycle 以外と併用）
- **安定化その他**
  - `--max_grad_norm`: 勾配クリッピング閾値（既定 1.0、0 以下で無効）
  - `--overflow_check`: true の場合、forward 出力に NaN/Inf/過大値が出たら即停止
  - `--overflow_tol`: 過大値検出の閾値（既定 1e8）
- **再構成保存**
  - `--recon_every`: 何エポックごとに再構成画像を保存するか
  - `--recon_count`: 再構成に使う val サンプル枚数
  - **更新**: `./recon/origin/NNNNN.png` に **元の val 画像**を一度だけ保存（番号で `./recon/<tag>/NNNNN.png` と 1:1 対応）

> **変更点（重要）**  
> - 旧オプション **`--amp_warmup_steps` は削除** しました。  
>   AMP の安定化は **`bf16` の利用**、**局所FP32**、**LRウォームアップ/スケジューラ**、**勾配クリップ**で行います。  
> - 損失は常に **FP32** で計算し、MS-SSIM 入力は **[0,1] に clamp** しています。

### 実装上の工夫（更新）
- ReLU は **`inplace=False`** に設定し、勾配計算の安定性を確保
- 順伝播は AMP (推奨: **BF16**) を利用しつつ、**損失計算は FP32** にキャスト
- **局所FP32**（entropy/decoder 周りなど）で危険演算のみ AMP 無効化
- 学習率は **Cosine/OneCycle/Plateau** 等のスケジューラに対応、`--lr_warmup_steps` と併用可
- 勾配クリッピングで発散を抑制
- forward 出力の NaN/Inf/過大値を即検知し、エラーで停止

---
## 3'. 学習（**NPU フレンドリー版**: `train_finetune_npu.py`）

`GDN/IGDN` を **NPU で動かしやすい非線形ブロック**（`GDNishLiteEnc/Dec`）に置換して学習するスクリプトです。  
量子化に強い **ReLU6 / HardSigmoid** と **Depthwise Conv** を用い、ONNX/NPU 互換性を重視しています。

### 主な追加オプション
- 置換対象: `--replace_parts {encoder,decoder,all}`（既定 `all`）
- エンコーダブロック: `--enc_t 2.0`（1×1の拡張率）, `--enc_kdw 3`, `--enc_eca true/false`, `--enc_residual true/false`
- デコーダブロック: `--dec_k 3`, `--dec_gmin 0.5`, `--dec_gmax 2.0`, `--dec_residual true/false`

### 実行例（**エンコーダのみ**学習：`replaced+hyper`）
```bash
python train_finetune_npu.py \
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

### 実行例（**全層**学習：`train_scope all`）
```bash
python train_finetune_npu.py \
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
---

## 4. 評価

```bash
python -m src.eval   --coco_dir /path/to/coco224 --use_prepared true   --checkpoint ./checkpoints/best_msssim.pt
```

出力: 平均 **PSNR** / **MS-SSIM**

---

## 5. 仕組みの概要

- `compressai.zoo.bmshj2018_factorized(quality, pretrained=True)` をロード
- `replace_gdn_with_relu(model, mode)` により、指定ブロックの GDN/IGDN を ReLU に置換
- `set_trainable_parts(model, scope)` により学習対象を制御  
  - 例: `--replace_parts decoder --train_scope replaced+hyper` → デコーダ置換＋hyperprior を学習
- **損失**:  
  - `recon`: `alpha * L1 + (1 - alpha) * (1 - MS-SSIM)`（既定 `alpha=0.4`）  
  - `rd`   : 上記に `+ lambda_bpp * bpp`（`likelihoods` から bpp を算出）
- AMP+FP32 ハイブリッド計算 + 局所FP32 で安定化
- 再構成画像は `recon/` に保存し、W&B にはグリッド形式でログ

---

## 6. ONNX 形式でのエクスポート（追加）

`bmshj2018_factorized`（ReLU 置換後）を **full / encoder / decoder** 単位で ONNX へ出力できます。

```bash
# フルモデル
python -m scripts.export_onnx   --ckpt checkpoints/last.pt   --part full   --out onnx/full.onnx   --input_size 224

# エンコーダのみ
python -m scripts.export_onnx   --ckpt checkpoints/last.pt   --part encoder   --out onnx/enc.onnx   --input_size 224

# デコーダのみ（latent チャネルは自動推定）
python -m scripts.export_onnx   --ckpt checkpoints/last.pt   --part decoder   --out onnx/dec.onnx   --input_size 224
```

オプション:
- `--replace_parts {encoder,decoder,all}`: ReLU 置換の対象（エクスポート前に適用）
- `--dynamic`: 動的軸（H/W）を ONNX に付与
- `--opset`: 既定 13

---

## 7. 注意点 / Tips（更新）
- ReLU 置換直後は不安定になりやすいので **学習率 1e-4 以下**を推奨
- `train_scope=replaced+hyper` が安定しやすい
- NaN/Inf が発生した場合は
  - **BF16 AMP** を優先（`--amp_dtype bf16`）
  - **局所FP32** を強める（`--local_fp32 entropy+decoder` や `all_normexp`、あるいは `custom`）
  - **学習率を下げる**（例: `--lr 5e-5`）
  - **勾配クリッピング** を強める（例: `--max_grad_norm 0.5`）
  - **LRウォームアップ** を長くする（`--lr_warmup_steps`）
- 旧オプション `--amp_warmup_steps` は **削除済み**。代替は上記の組み合わせで対応してください。

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

### 9.2 GDN と ReLU の違い
- **GDN/IGDN**: 各チャネル同士の関係を正規化し、**歪みとレートの両立**に寄与（圧縮向けに設計された活性化）。  
- **ReLU**: シンプルで高速・NPU 互換性も高いが、**分布の形**が変わり、レート（bpp）が悪化することがあります。  
このため、**置換後に再学習**して性能を取り戻します。

### 9.3 本リポジトリの工夫
- `--replace_parts` で置換範囲を選択（エンコーダだけ、デコーダだけ、両方）。
- `--train_scope` で**ハイパープライヤ（hyperprior）**や**エントロピー・ボトルネック**も一緒に再学習可能。  
  置換による表現分布の変化を**確率モデル側**でも吸収させる狙いです。

---

## 10. 損失関数の分析方法（初学者向け）

### 10.1 再構成損失
- **L1**：平均絶対誤差（小さいほどよい）。  
- **MS-SSIM**：人間の視覚に近い類似度（1 に近いほどよい）。  
- 本レシピの**再構成損失**は `alpha * L1 + (1 - alpha) * (1 - MS-SSIM)`。  
  - `alpha` が大きい → L1 を重視（細部や平均的な誤差）  
  - `alpha` が小さい → MS-SSIM を重視（知覚的な見た目）

### 10.2 RD 損失と bpp
- **bpp (bits per pixel)**：1 画素あたりのビット数。小さいほど高圧縮。  
- **RD 損失**：`loss = recon + λ * bpp`。`λ`（ランダウ係数）が大きいほど**圧縮率重視**になります。

### 10.3 何をどう見ればいい？
- 本スクリプトは **学習中に以下を標準出力と W&B に記録**します：
  - `RD`（総合損失）, `L1`, `1-MSSSIM`, `BPP`, `λ`, `MSSSIM`, `PSNR`
- **おすすめの見方**
  1. `RD` が**下がっているか**（総合目標）  
  2. `BPP` が**下がっているか**（圧縮率改善）  
  3. `1-MSSSIM` / `L1` が**上がっていないか**（画質劣化していないか）  
  4. `PSNR`, `MS-SSIM` が**維持 or 改善**しているか

### 10.4 典型的なトレードオフ
- `λ` を上げる → `BPP` は下がるが、`1-MSSSIM` や `L1` が上がりやすい（＝画質低下）。  
- `α`（alpha_l1）を上げる → L1 を重視、テクスチャの**粒状感**が変化することあり。

### 10.5 再構成画像と元画像の対応確認
- `./recon/<tag>/NNNNN.png`（復元） と `./recon/origin/NNNNN.png`（元画像）は **番号で 1:1 対応**です。  
  時系列で画質変化やアーティファクトの**視覚的比較**が簡単にできます。

---

## 11. 学習時の最適化手法（初学者向け）

### 11.1 数値安定化の基本セット
- **AMP を BF16 で使用**（`--amp true --amp_dtype bf16`）  
- **損失は FP32** で計算（内部で自動切替）  
- **局所FP32**：`--local_fp32 entropy+decoder` を推奨（確率モデルや Exp/Log 周りを FP32）  
- **勾配クリップ**：`--max_grad_norm 1.0`（デフォルト）

### 11.2 学習率スケジューリング
- **CosineAnnealingLR**：シンプルで安定。`--sched cosine`  
- **OneCycleLR**：立ち上がりが速い。`--sched onecycle --onecycle_pct_start 0.1`  
- **ReduceLROnPlateau**：停滞時に LR を自動で下げる。`--sched plateau`

### 11.3 つまずいた時の対処
- 収束が遅い／発散する
  - 学習率を 0.5〜0.2 倍に下げる（例 `1e-4 → 5e-5`）
  - `--lr_warmup_steps` を増やす（例 `1000`）
  - `--local_fp32 all_normexp` で FP32 範囲を広げる
  - `--train_scope replaced+hyper` にして確率モデルも一緒に再学習
- 画質は良いが bpp が高い
  - `--lambda_bpp` を少し上げる（例 `0.015` → 圧縮率重視）
- bpp は下がるが画質が悪い
  - `--lambda_bpp` を少し下げる、または `alpha_l1` を下げて MS-SSIM 比重を上げる

### 11.4 ログ活用（W&B）
- 有効化: `--wandb true`（必要なら `WANDB_PROJECT` を環境変数で指定）  
- 主なキー（学習中）  
  - 再構成: `train/msssim`, `train/psnr`  
  - RD 内訳: `train/rd/total`, `train/rd/recon_total`, `train/rd/l1`, `train/rd/1-msssim`, `train/rd/bpp`, `train/rd/lambda_bpp`  
- 検証: `val/ms_ssim`, `val/psnr`  
- 画像プレビュー: `recon/<tag>`（グリッド）

---

以上。初心者は **「3. 学習」→「10. 損失の見方」→「11. 最適化」** の順で触るのがおすすめです。


---

# 追加: NPU 向け GDN/IGDN 置換（非線形・量子化フレンドリー）

本レシピに **NPU フレンドリーな GDN/IGDN 置換ブロック**（ReLU6 / HardSigmoid / DepthwiseConv / ECA 等）を追加しました。  
除算・平方根を避け、**量子化に強い**活性化のみで構成されています。

## A. 同梱ファイル
- `src/npu_blocks.py`
  - **GDNishLiteEnc**（GDN代替）: `1x1 → ReLU6 → DW3×3 → ReLU6 → 1x1 → (ECA) [+ residual]`
  - **GDNishLiteDec**（IGDN代替）: `1x1 → ReLU6 → per-channel 1x1 → HardSigmoid(スケール) → 乗算 → k×k [+ residual]`
- `src/replace_gdn_npu.py`
  - `replace_gdn_with_npu(model, mode='all', ...)`：CompressAI の **GDN/IGDN を自動検出して置換**。
- `train_finetune_npu.py`
  - 置換済みモデルを学習する**代替スクリプト**（CLI で t, kdw, ECA, ゲイン範囲を指定可能）。

## B. 使い方（最短）
1) 上記 3ファイルをプロジェクトの `src/`（`train_finetune_npu.py` は直下）に配置。  
2) 実行:
```bash
python train_finetune_npu.py   --coco_dir /path/to/coco224   --quality 8 --epochs 10 --batch_size 16   --replace_parts all   --enc_t 2.0 --enc_kdw 3 --enc_eca true --enc_residual true   --dec_k 3 --dec_gmin 0.5 --dec_gmax 2.0 --dec_residual true   --loss_type rd --lambda_bpp 0.01
```

## C. 既存スクリプトからの呼び出し
`src/train_finetune_relu.py` を使い続けたい場合は、モデル構築直後の
```python
from src.replace_gdn_npu import replace_gdn_with_npu
model = replace_gdn_with_npu(model, mode=args.replace_parts)
```
に差し替えるだけでOKです。

## D. 推奨ハイパラと量子化ノート
- **Encoder(GDN)**: `t=2.0, kdw=3, ECA=on, residual=on`  
- **Decoder(IGDN)**: `k=3, g_min=0.5, g_max=2.0, residual=on`  
- **活性化**: ReLU6 / HardSigmoid（**量子化(QAT)に強い**分割線形）  
- **QAT**: 可能なら FakeQuant を追加（per-channel weight / symmetric）。

## E. ダウンロード
- 置換ブロック/スクリプト一式の ZIP（`src/` + 学習スクリプト）はこちら:  
  **[npu_gdn_replace_bundle.zip をダウンロード](sandbox:/mnt/data/npu_gdn_replace_bundle.zip)**  

---

---
# GDNishLiteEnc / GDNishLiteDec モジュール仕様

## 1. 背景と設計思想
`GDNishLiteEnc` および `GDNishLiteDec` は、CompressAI における  
**GDN (Generalized Divisive Normalization)** / **IGDN (Inverse GDN)** の代替として設計された  
**NPU・ONNX フレンドリーな非線形活性ブロック**です。

GDN/IGDN はチャネル間の相関をモデル化することで圧縮性能を高めますが、  
浮動小数演算・除算・平方根などが多く、量子化やNPU上での展開に不向きです。  
`GDNishLite*` はその特性を**近似的に再現しつつ、すべての演算をReLU系で構成**しています。

---

## 2. GDNishLiteEnc（Encoder側活性ブロック）

### 構成概要
```
Input
 └─> 1×1 Conv (expand: t×C)
      └─> ReLU6
          └─> Depthwise Conv (k_dw×k_dw)
               └─> ReLU6
                   └─> 1×1 Conv (reduce: C)
                       └─> [ECA block (optional)]
                           └─> Residual add (optional)
Output
```

### 機能意図
- **1×1 Conv + Depthwise Conv**  
  → GDN が担う「チャネル相関＋局所正規化」を線形混合と空間フィルタで模倣。  
- **ReLU6**  
  → NPU 量子化に強く、範囲が固定 `[0,6]`。分布安定化に寄与。  
- **ECA (Efficient Channel Attention)**  
  → チャネル依存の適応ゲインを再現（除算でなく乗算ゲート）。  
- **Residual**  
  → 恒等写像近傍で初期化可能にし、学習初期の崩壊を防止。

### 引数一覧
| 引数名 | 型 / 既定値 | 説明 |
|:-------|:-------------|:-----|
| `channels` | int | 入力チャネル数 |
| `t` | float = 2.0 | 1×1 Convでの拡張率。2.0で2倍チャネルへ一時展開 |
| `k_dw` | int = 3 | Depthwise Convのカーネルサイズ（3 or 5 推奨） |
| `use_eca` | bool = True | ECAブロックを挿入してチャネル依存ゲインを付与 |
| `residual` | bool = True | 残差接続を有効化。既定True |
| `eca_kernel` | int = 3 | ECA内部の1D畳み込みカーネル幅 |
| `init_identity` | bool = True | 初期化時に恒等近傍重みで安定化 |
| `act_type` | str = "relu6" | 活性化種別（`relu6` or `hardswish`） |
| `bn` | bool = False | BatchNormを入れる場合（通常False、NPU親和） |

### 推奨設定例
- 通常学習：`t=2.0, k_dw=3, use_eca=True, residual=True`
- 軽量化：`t=1.5, k_dw=3, use_eca=False`
- 高精度：`t=2.5, k_dw=5, use_eca=True, residual=True`

---

## 3. GDNishLiteDec（Decoder側活性ブロック）

### 構成概要
```
Input
 └─> 1×1 Conv (C→C)
      └─> ReLU6
          └─> Per-channel 1×1 Conv (scale predictor)
               └─> HardSigmoid → Clamp([g_min, g_max])
                   └─> Multiply (ゲイン適用)
                       └─> Depthwise Conv (k×k)
                           └─> [Residual add]
Output
```

### 機能意図
- **HardSigmoid ゲート**  
  → GDNの「√(β + Σγx²)」のような“ゲイン変調”を0〜1範囲で模倣。  
- **g_min/g_maxスケーリング**  
  → ゲインを `[g_min, g_max]` の範囲で再スケールし、1超の増幅も可能。  
- **Depthwise Conv**  
  → 局所的な復元を助ける平滑化を付加。  
- **Residual**  
  → 恒等出力を保持しつつ微分安定性を確保。

### 引数一覧
| 引数名 | 型 / 既定値 | 説明 |
|:-------|:-------------|:-----|
| `channels` | int | 入力チャネル数 |
| `k` | int = 3 | Depthwise Convのカーネルサイズ |
| `g_min` | float = 0.5 | HardSigmoid出力をスケーリングする下限 |
| `g_max` | float = 2.0 | 同上 上限（>1で増幅も可能） |
| `residual` | bool = True | 残差接続を有効化 |
| `act_type` | str = "relu6" | 中間活性化種別（ReLU6またはHardSwish） |
| `use_bn` | bool = False | BatchNorm使用可否（デフォルトFalse） |
| `init_identity` | bool = True | 恒等近傍初期化 |

### 推奨設定例
- 標準構成：`k=3, g_min=0.5, g_max=2.0, residual=True`
- 微細調整用：`g_min=0.7, g_max=1.5`（量子化時のダイナミックレンジを抑制）

---

## 4. 実装上のポイント
- すべての活性関数は **ONNX / NPU 互換演算**（ReLU6, HardSigmoid, DepthwiseConv）で構成。  
- **BatchNorm非依存設計**のため、INT8量子化でも分散揺れが少ない。  
- `init_identity=True` 時は初期パラメータを恒等近傍にし、既存学習済み重みで置換しても出力分布を崩さない。  
- ReLU6とHardSigmoidの組み合わせにより、**出力分布が安定かつ単調非減少**を保ち、GDN特有の非線形抑制特性を近似。

---

## 5. 推奨利用パターン
| 用途 | 設定 | 備考 |
|------|------|------|
| 学習済みGDNを置換・再学習 | `replace_parts all`, `--enc_t 2.0`, `--dec_gmin 0.5 --dec_gmax 2.0` | 既存重み初期化＋fine-tune |
| NPU向け軽量学習 | `--enc_t 1.5 --enc_eca false --dec_gmin 0.7 --dec_gmax 1.3` | 低消費電力優先 |
| 高画質再構成検証 | `--enc_t 2.5 --enc_eca true --dec_gmin 0.4 --dec_gmax 2.5` | 学習時にRD損失を採用 |

---

## 6. 学習時の安定化Tips
- `bf16` AMP 推奨（除算を含まないため誤差耐性が高い）  
- `--local_fp32 entropy+decoder` で確率モデル周りをFP32化  
- ReLU6により勾配爆発を防ぎつつ、HardSigmoid出力を`[g_min,g_max]`で制御  
- QATを行う場合は`FakeQuantObserver`をConv前後に挿入可

---

## 7. ONNX / NPU 特性
| 要素 | 対応 | 備考 |
|------|------|------|
| ReLU6 / HardSigmoid | ✅ | ONNX opset ≥ 13 対応 |
| Depthwise Conv | ✅ | Group = channels 指定 |
| ECA (1D Conv) | ✅ | Flatten → Conv1D で展開可 |
| Residual Add | ✅ | ElementwiseAdd |
| BatchNorm | ⛔ (非推奨) | 量子化変換時に非線形範囲が拡大するため |

---

## 8. 参考
- Ballé et al., *"End-to-end Optimized Image Compression"*, ICLR 2017  
- Ma et al., *"ECA-Net: Efficient Channel Attention"*, CVPR 2020  
- Han et al., *"Hard-Swish and Quantization-friendly Activations"*, Google AI, 2019
