---
language:
- ja
license: cc-by-sa-4.0
base_model: convaiinnovations/laya-multilingual
library_name: laya
tags:
- japanese
- natural-language-inference
- decision-model
- lora
---

# Laya-JA

判断モデル [Laya Multilingual](https://huggingface.co/convaiinnovations/laya-multilingual)（322M）を日本語で追加学習した重みです。日本語の文章を読み、選択肢の確率、二値確率、0〜5の評点を返します。文章の生成はしません。エンコーダーの全層にLoRA（ランク16）を加え、推論・常識・類似度・文法を一つのモデルで学習しました。

コード、学習と評価の手順、全結果は [GitHub の nozomugit/laya-ja](https://github.com/nozomugit/laya-ja) にあります。Convai Innovations とは独立したコミュニティプロジェクトです。

## 成績

元になったモデル Jev、その公開版 Laya、このプロジェクトの Laya-JA を同じ問題で比べました。Laya と Laya-JA は GPU 上の float32 で評価しています。

| 課題・評価問題 | Jev | Laya | Laya-JA |
|---|---:|---:|---:|
| 推論 JNLI 2,434問 | 82.1% | 52.2% | **90.7%**（90.6〜90.9） |
| 推論 JSICK 512問 | 81.4% | 44.7% | **88.8%**（88.5〜89.1） |
| 推論 JaNLI 512問・二値 | 95.3% | 58.4% | **99.9%**（99.6〜100.0） |
| 時間推論 Jamp 348問 | **77.3%** | 44.3% | 76.4%（74.7〜77.9） |
| 意味推論 JSeM 384問 | **77.3%** | 50.0% | 65.4%（64.8〜66.1） |
| 独自の推論36問・学習外 | **100.0%** | 38.9% | 54.6%（47.2〜63.9） |
| 常識五択 JCommonsenseQA 1,119問 | **96.9%** | 54.6% | 70.5%（70.0〜70.8） |
| 意味類似度 JSTS・Pearson相関 | 0.899 | 0.601 | **0.911**（0.910〜0.911） |
| 文法判断 JCoLA 同分野・MCC | **0.236** | −0.005 | 0.183（0.164〜0.206） |
| 文法判断 JCoLA 異分野・MCC | **0.572** | 0.005 | 0.304（0.239〜0.352） |

- **Jev**：TypeSafe AI の非公開モデル（`jev-1.14.0`）。Laya が再現を目指している元のモデルです。同じ指示文と選択肢を API に送り、例示も追加学習もなしで答えさせました（2026年11月12日）。
- **Laya**：Jev のオープンソース版にあたる上流の多言語モデル（Laya Multilingual）。追加学習なし。
- **Laya-JA**：Laya に日本語で全層LoRAを加えた、このプロジェクトのモデル。全課題を一つのモデルで答えます。3シードの平均、括弧内は最小〜最大。
- 太字は Jev と Laya-JA のうち高いほうです。

## 速度

Laya-JA は手元の機械で動き、通信を待たずに答えます。LoRA を本体に織り込んであるので、速度は Laya と同じです。

| 条件 | Laya-JA | Laya |
|---|---:|---:|
| GPU（A100）、1問ずつ | **38ミリ秒**（95%点 40ミリ秒） | 38ミリ秒 |
| GPU（A100）、64問まとめて | **毎秒558問** | 毎秒573問 |
| CPU 4スレッド、1問ずつ | 128ミリ秒（95%点 135ミリ秒） | 130ミリ秒 |
| CPU 1スレッド、1問ずつ | 270ミリ秒（95%点 290ミリ秒） | 264ミリ秒 |

Jev は API での提供で、TypeSafe AI の公表値は通信を含めて1回70〜500ミリ秒です。同じ日にこちらから約9,800問を8並列で送ったときは、1回あたり平均で約0.25秒でした。Jev の値は通信を含む API の応答時間、上の表は手元の機械での計算時間で、測っている範囲が違います。

推論の問題300問（CPU は60問）を1回ずつ計測した中央値です。float32。計測は GitHub の `speed_bench.py`。

## 限界

- JNLI・JSICK・JaNLI・JSTS で Jev を上回っているのは、同じコーパスの訓練データで学習したモデルと、学習していないモデルの比較です。
- 学習用データのない問題には弱く、独自の推論36問は17〜23問の正解にとどまります（Jev は全問正解）。条件、否定、数量、時点を読み分ける推論は苦手です。
- JSeM の「判断できない」の再現率は約2割です。
- JaNLI の成績は同じ文型で学習した結果です。JSeM は、評価問題と重ならない JSeM の問題を学習に使っています。
- 文法判断の MCC は低く、実験用です。正解は JCoLA のラベルで、方言や会話での自然さは評価していません。
- 上流モデルの事前学習データと評価問題の重複は確認していません。

## ファイル

| 場所 | 中身 |
|---|---|
| `unified/` | 全課題を一つにまとめた完成版。LoRA を本体に織り込み済み（fp16）。`laya.load` で読めます。シード42。 |
| `adapters/unified-s42`〜`s44` | 同じモデルのアダプター（LoRA の更新と判断ヘッド）。上流のベースモデルに重ねて使います。 |
| `adapters/nli-s42`〜`s44` | 推論だけを学習したアダプター。 |
| `adapters/jcommonsenseqa-`・`jsts-`・`jcola-` | 課題ごとに学習したアダプター。 |
| `SHA256SUMS` | 全ファイルの SHA-256。 |

## 使い方

```bash
git clone https://github.com/nozomugit/laya-ja.git
cd laya-ja
python3.12 -m venv .venv && source .venv/bin/activate
python install.py
python predict.py lora-nli --adapter path/to/nli-s42.safetensors '月曜に資料を送り、水曜に返事を受け取った。' '資料を送ったのは返事を受け取る前だ。'
```

完成版は `laya.load("path/to/unified")` で読み込み、上流の Laya と同じ形式で質問できます。質問文と選択肢は学習時と同じもの（リポジトリの `laya_ja.py`）を使ってください。

## 学習

- ベース：`convaiinnovations/laya-multilingual`、リビジョン `1720e3e3357cfe1e281542e223f8273b0890ca34`。
- エンコーダー全層の線形層にLoRA（ランク16、alpha 32、dropout 0.05）、判断ヘッドも学習。学習率 1e-4、2エポック、実効バッチ32、シード42・43・44。モデル選定は調整用データの損失。
- 学習データ：JNLI 18,618、JSICK 4,000、JaNLI 4,000、Jamp 4,000、JSeM 1,544、合成論理 6,000、外部モデルの判定を付けた日本語MultiNLI 8,000、JCommonsenseQA 8,465、JSTS 11,816、JCoLA 6,471。評価・調整用の問題と同じ文の組や出典は学習から除いています。

## 出典とライセンス

この重みは CC BY-SA 4.0 で公開します。ベースモデルと上流のエンジンは Apache-2.0 です。

- [JGLUE](https://github.com/yahoojapan/JGLUE)（JNLI・JSTS・JCommonsenseQA）— Kurihara, Kawahara, Shibata
- [JCoLA](https://github.com/osekilab/JCoLA) — Someya, Sugimoto, Oseki
- [JSICK](https://github.com/verypluming/JSICK)・[JaNLI](https://github.com/verypluming/JaNLI) — Yanaka, Mineshima
- [Jamp](https://github.com/tomo-ut/temporalNLI_dataset) — Sugimoto, Onoe, Yanaka
- [JSeM](https://github.com/DaisukeBekki/JSeM) — Kawazoe, Tanaka, Mineshima, Bekki
- [日本語MultiNLI（cl-nagoya/nu-mnli）](https://huggingface.co/datasets/cl-nagoya/nu-mnli) — 名古屋大学。元の MultiNLI は Williams, Nangia, Bowman。
- 教師ラベルの作成には Qwen 系の公開モデル2つを使いました。

各データの著者とライセンスの詳細は、リポジトリの [NOTICE](https://github.com/nozomugit/laya-ja/blob/main/NOTICE) にあります。
