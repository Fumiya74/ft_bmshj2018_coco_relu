# ft_bmshj2018_coco_relu

**COCO2017 (224×224 crops) を用いて CompressAI の `bmshj2018_factorized` の GDN/IGDN を ReLU に置換し、  
再構成品質（L1 + (1 − MS‑SSIM)）でファインチューニングするレシピ**。  
既存の実行フローとディレクトリ構成（`scripts/` でデータ準備 → `src/` の学習スクリプト → `checkpoints/` 保存 → `recon/` に再構成 PNG 出力）を踏襲しています。

> ⚠️ 注意: GDN/IGDN は**可逆性やゲイン正規化**のために設計された層であり、ReLU に単純置換すると RD 特性を崩す可能性があります。  
> 本レシピは **デコーダ（またはオプションでエンコーダ/全体）を再学習**し、画質面でのフィットを狙う実験用構成です。  
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
python -m scripts.coco_prepare     --coco_dir /path/to/coco2017     --out_dir  /path/to/coco224     --include_val true
```

生成後の構成例：

```text
/path/to/coco224/
  ├─ train/  *.jpg (224x224)
  └─ val/    *.jpg (224x224)
```

> 💡 高速化・再現性のために**事前クロップ版を推奨**します。オンザフライのリサイズ/クロップも可能ですが、I/O が安定する事前版が便利です。

---

## 3. 学習（ReLU 置換モデルのファインチューニング）

```bash
python -m src.train_finetune_relu     --coco_dir /path/to/coco224     --use_prepared true     --quality 8     --epochs 10     --batch_size 16     --lr 1e-4     --alpha_l1 0.4     --train_parts decoder     --recon_every 2     --recon_count 16     --save_dir ./checkpoints     --recon_dir ./recon
```

### 主な引数
- `--quality`: CompressAI の画質インデックス（0〜8）。
- `--train_parts`: 学習対象を選択  
  - `decoder`（既定）: デコーダ側のみ学習  
  - `encoder`: エンコーダ側のみ学習  
  - `decoder+encoder`: 両方学習  
  - `all`: 全体学習
- `--alpha_l1`: 損失関数の L1 重み。`loss = alpha*L1 + (1-alpha)*(1-MS‑SSIM)`
- `--use_prepared`: `true` の場合、事前クロップ済みデータを使用。
- `--recon_every`: 何エポックごとに再構成画像を保存するか。
- `--recon_count`: 再構成に使う val サンプル枚数。
- `--recon_dir`: 再構成画像の保存先。

### W&B ログ利用
```bash
WANDB_PROJECT=bmshj2018_relu wandb login  # 初回のみ
python -m src.train_finetune_relu ... --wandb true
```

- `save_dir/wandb_run_id.txt` に run_id が保存され、`--resume` 時に同じ run に続きます。
- run_id には `train_parts` 情報が付与されるため、ダッシュボード上でどの部位を学習した run か一目で判別可能です。

---

## 4. 評価

```bash
python -m src.eval     --coco_dir /path/to/coco224     --use_prepared true     --checkpoint ./checkpoints/best_msssim.pt
```

出力: 平均 **PSNR** / **MS‑SSIM**

---

## 5. 仕組みの概要

- `compressai.zoo.bmshj2018_factorized(quality, pretrained=True)` をロード。
- `replace_gdn_with_relu(model, mode)` により、**学習対象に選ばれた部分だけ** GDN/IGDN を ReLU に置換。
  - 例: `--train_parts decoder` → デコーダ側のみ ReLU に置換し学習、エンコーダ側は GDN のまま固定。
- `set_trainable_parts(model, mode)` で更新対象のパラメータを限定。
- 損失関数: `alpha * L1 + (1 - alpha) * (1 - MS‑SSIM)`（既定 `alpha=0.4`）。
- 再構成画像はエポックごとに `recon/` 以下に PNG 出力し、W&B にはグリッド形式でログ。

---

## 6. 注意点 / Tips

- ReLU 置換直後は学習が不安定になりやすいため **学習率は小さめ（1e‑4）** を推奨。
- `decoder+encoder` を有効にすると改善する場合がありますが、発散や過学習のリスクあり。
- I/O 負荷が高い環境では `--recon_count 8` 程度に減らすと安定。
- オンザフライ前処理を使う場合は `--use_prepared false --input_size 224` を指定。

---

## 7. ライセンスとクレジット

- 本レシピは研究・実験目的での利用を想定しています。  
- 元モデル・コードは [CompressAI](https://github.com/InterDigitalInc/CompressAI) の `bmshj2018_factorized` を利用しています。