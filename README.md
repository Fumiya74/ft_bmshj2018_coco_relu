# ft_bmshj2018_coco_relu

**COCO2017 (224×224 crops) を用いて CompressAI の `bmshj2018_factorized` の GDN/IGDN を ReLU に置換し、  
再構成品質（L1 + (1 − MS-SSIM)）でファインチューニングするレシピ**。  
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
python -m scripts.coco_prepare \
    --coco_dir /path/to/coco2017 \
    --out_dir  /path/to/coco224 \
    --include_val true
```

---

## 3. 学習（ReLU 置換モデルのファインチューニング）

```bash
python -m src.train_finetune_relu \
    --coco_dir /path/to/coco224 \
    --use_prepared true \
    --quality 8 \
    --epochs 10 \
    --batch_size 16 \
    --lr 1e-4 \
    --alpha_l1 0.4 \
    --replace_parts encoder \
    --train_scope replaced \
    --recon_every 2 \
    --recon_count 16 \
    --save_dir ./checkpoints \
    --recon_dir ./recon
```

### 主な引数
- `--quality`: CompressAI の画質インデックス（0〜8）。
- `--replace_parts`: GDN→ReLU に置換するブロックを選択  
  - `encoder` / `decoder` / `all`
- `--train_scope`: 再学習範囲を選択  
  - `replaced`: 置換したブロックのみ  
  - `replaced+hyper`: 置換ブロック＋hyperprior＋entropy_bottleneck  
  - `all`: 全層学習
- `--alpha_l1`: 損失関数の L1 重み。`loss = alpha*L1 + (1-alpha)*(1-MS-SSIM)`
- `--recon_every`: 何エポックごとに再構成画像を保存するか。
- `--recon_count`: 再構成に使う val サンプル枚数。

### 実装上の工夫
- ReLU は **`inplace=False`** に設定し、勾配計算の安定性を確保。
- 順伝播は AMP (半精度) を利用しつつ、損失計算は FP32 にキャストして NaN 発生を防止。

---

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

- `compressai.zoo.bmshj2018_factorized(quality, pretrained=True)` をロード。
- `replace_gdn_with_relu(model, mode)` により、指定ブロックの GDN/IGDN を ReLU に置換。
- `set_trainable_parts(model, scope)` により学習対象を制御。  
  - 例: `--replace_parts decoder --train_scope replaced+hyper` → デコーダ置換＋hyperprior を学習。
- 損失関数: `alpha * L1 + (1 - alpha) * (1 - MS-SSIM)`（既定 `alpha=0.4`）。
- 再構成画像は `recon/` に保存し、W&B にはグリッド形式でログ。

---

## 6. 注意点 / Tips

- ReLU 置換直後は不安定になりやすいので **学習率 1e-4 以下**を推奨。
- `train_scope=replaced+hyper` にすると安定しやすいケースがあります。
- NaN が発生した場合は AMP を無効化 (`torch.cuda.amp.autocast(enabled=False)`) して切り分け可能。

---

## 7. ライセンスとクレジット

- 本レシピは研究・実験目的での利用を想定しています。  
- 元モデル・コードは [CompressAI](https://github.com/InterDigitalInc/CompressAI) の `bmshj2018_factorized` を利用しています。