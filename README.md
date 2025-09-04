# ft_bmshj2018_coco_relu

**COCO2017 (224×224 crops) で CompressAI の `bmshj2018_factorized` の GDN/IGDN を ReLU に置換し、
再構成品質（L1 + (1 − MS‑SSIM)）でファインチューニングするレシピ**。  
既存の実行フローとディレクトリ構成（`scripts/` でデータ準備 → `src/` の学習スクリプト → `checkpoints/` 保存 → `recon/` に再構成 PNG 出力）を踏襲しています。

> ⚠️ 注意: GDN/IGDN は**可逆性やゲイン正規化**の観点で設計されており、単純な ReLU 置換は元の RD 最適化特性を崩します。  
> 本レシピは **デコーダ（およびオプションでエンコーダ）を再学習**して「画質面のフィット」を狙う構成です。速度や NPU 互換性のための実験枠として活用してください。

---

## 1. 環境セットアップ

```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 2. データ準備（224×224 クロップ）

**COCO2017 をダウンロード済み**（`train2017/` と `val2017/` が存在）なら、次で 224×224 の単純センタクロップ版を作ります。

```bash
python -m scripts.coco_prepare       --coco_dir /path/to/coco2017       --out_dir  /path/to/coco224       --include_val true
```

生成後の構成例：

```text
/path/to/coco224/
  ├─ train/  *.jpg (224x224)
  └─ val/    *.jpg (224x224)
```

> メモ: 高速化・再現性目的で**事前クロップ**を推奨しています。オンザフライでのリサイズ/クロップにも対応していますが、I/O が安定する事前版を推奨。

## 3. 学習（ReLU 置換モデルのファインチューニング）

```bash
python -m src.train_finetune_relu       --coco_dir /path/to/coco224       --use_prepared true       --quality 8       --epochs 10       --batch_size 16       --lr 1e-4       --alpha_l1 0.4       --recon_every 2       --recon_count 16       --save_dir ./checkpoints       --recon_dir ./recon
```

主要引数：

- `--quality`: CompressAI の画質インデックス（0〜8）。
- `--train_parts`: `decoder`（既定）, `decoder+encoder`, `all` から選択。RD モデルの安定性のため既定は decoder のみ。
- `--alpha_l1`: 損失 `alpha * L1 + (1 - alpha) * (1 - MS‑SSIM)` の L1 重み。
- `--use_prepared`: `true` の場合、事前クロップ済みの `train/` と `val/` を使用。
- `--recon_every`, `--recon_count`, `--recon_dir`: 再構成 PNG の書き出し頻度・枚数・出力先。

W&B ログを使う場合：

```bash
WANDB_PROJECT=bmshj2018_relu wandb login  # 一度だけ
python -m src.train_finetune_relu ... --wandb true
```

## 4. 評価

```bash
python -m src.eval       --coco_dir /path/to/coco224       --use_prepared true       --checkpoint ./checkpoints/best_msssim.pt
```

出力：平均 PSNR / MS‑SSIM を表示。

---

## 仕組みの概要

- `compressai.zoo.bmshj2018_factorized(quality, pretrained=True)` をロード。
- モデル内部の **GDN（`inverse=False`）/IGDN（`inverse=True`）を `nn.ReLU(inplace=True)` に置換**。
- 既定では **デコーダ `g_s` のみ学習**（`--train_parts decoder+encoder` でエンコーダも学習）。  
  符号化側（エントロピー・ハイパーネット）は固定し、**純粋な再構成誤差の最小化**にフォーカス。
- 損失は `alpha * L1 + (1 - alpha) * (1 - MS‑SSIM)`。既定 `alpha=0.4` は MS‑SSIM 寄り。

## 既知の注意点 / Tips

- 置換直後は学習安定のため **学習率を小さめ**（1e‑4 など）に。  
- `decoder+encoder` を解凍すると改善するケースがありますが、過学習や学習発散に注意。
- 速度優先であれば `--recon_count` を 8 に、ログを控えめにして I/O を抑制。
- オンザフライ前処理を使う場合は `--use_prepared false` とし、`--input_size 224` を指定。

---

## ライセンスとクレジット

- 本レシピは研究・実験目的です。  
- 元モデル・コードは [CompressAI](https://github.com/InterDigitalInc/CompressAI) の `bmshj2018_factorized` を利用しています。
