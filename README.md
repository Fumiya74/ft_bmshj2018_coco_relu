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
python -m src.train_finetune_relu   --coco_dir /path/to/coco224 --use_prepared true   --quality 8 --epochs 10 --batch_size 16 --lr 1e-4 --alpha_l1 0.4   --replace_parts encoder --train_scope replaced+hyper   --recon_every 2 --recon_count 16   --save_dir ./checkpoints --recon_dir ./recon   --amp true --amp_dtype bf16   --local_fp32 entropy   --loss_type recon   --sched cosine --optimizer adamw --weight_decay 1e-4
```

### RD 最適化（bpp も最小化）
```bash
python -m src.train_finetune_relu   --coco_dir /path/to/coco224 --use_prepared true   --quality 8 --epochs 10 --batch_size 16 --lr 1e-4 --alpha_l1 0.4   --replace_parts encoder --train_scope replaced+hyper   --recon_every 2 --recon_count 16   --save_dir ./checkpoints --recon_dir ./recon   --amp true --amp_dtype bf16   --local_fp32 entropy+decoder   --loss_type rd --lambda_bpp 0.01   --sched onecycle --onecycle_pct_start 0.1 --optimizer adam
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
