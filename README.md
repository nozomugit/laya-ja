# Laya-JA

判断モデル [Laya Multilingual](https://huggingface.co/convaiinnovations/laya-multilingual)（322M）の日本語性能を引き上げるプロジェクトです。日本語の文章を読み、選択肢、二値確率、0〜5の評点を返します。Convai Innovations とは独立したコミュニティプロジェクトです。

## 現在の成績

エンコーダーの全層にLoRA（ランク16）を加え、推論・常識・類似度・文法を一つのモデルで学習しました。推論時に外部の言語モデルは使いません。数値は3シードの平均、括弧内は最小〜最大です。

| 課題・評価問題 | 上流の多言語版 | 判定層のみの学習（従来） | 全層LoRA・一つのモデル（今回） | 参考：Jev 1.14（API） |
|---|---:|---:|---:|---:|
| 推論 JNLI 2,434問 | 52.2% | 81.4% | **90.7%**（90.6〜90.9） | 82.1% |
| 推論 JSICK 512問 | 44.7% | 83.4% | **88.8%**（88.5〜89.1） | 81.4% |
| 推論 JaNLI 512問・二値 | 58.4% | 89.3% | **99.9%**（99.6〜100.0） | 95.3% |
| 時間推論 Jamp 348問 | 44.3% | 63.5% | 76.4%（74.7〜77.9） | **77.3%** |
| 意味推論 JSeM 384問 | 50.0% | 46.9% | 65.4%（64.8〜66.1） | **77.3%** |
| 独自の推論36問・学習外 | 38.9% | 50.0% | 54.6%（47.2〜63.9） | **100.0%** |
| 常識五択 JCommonsenseQA 1,119問 | 54.6% | 61.8% | 70.5%（70.0〜70.8） | **96.9%** |
| 意味類似度 JSTS・Pearson相関 | 0.601 | 0.875 | **0.911**（0.910〜0.911） | 0.899 |
| 文法判断 JCoLA 同分野・MCC | −0.005 | 0.090 | 0.183（0.164〜0.206） | **0.236** |
| 文法判断 JCoLA 異分野・MCC | 0.005 | 0.116 | 0.304（0.239〜0.352） | **0.572** |

- **上流の多言語版からは全項目で改善しました。** 従来の判定層のみの学習と比べても全項目で上回っています。
- **Jev は Laya が再現を目指している TypeSafe AI の非公開モデルです。** 同じ指示文と選択肢を API に送り、例示も追加学習もなしで答えさせました（2026年11月12日、`jev-latest` が `jev-1.14.0` として応答）。今回のモデルは JNLI・JSICK・JaNLI・JSTS の学習用データで学習しているので、この4行で上回っているのは、同じコーパスで学習したモデルと学習していないモデルの比較です。
- **学習用データのない問題では Jev との差が大きく残ります。** 独自36問は Jev が全問正解、今回のモデルは17〜23問。常識五択も26ポイントの差があります。
- JaNLI は同じ文型で学習した結果です。JSeM は、評価問題と重ならない JSeM の問題を学習に使っています。独自36問は学習にもモデル選定にも使っていませんが、このリポジトリで公開している問題です。
- 評価は GPU 上の float32。「従来」は CPU で評価した値で、推論は一つのモデルで全コーパスに答える構成の値です。

[比較表・注意点・再現手順](LORA.md) ／ [推論](results/lora-nli-all.json)・[課題別と統合](results/lora-tasks.json)・[Jev](results/jev.json) の結果

## 試す

下のコマンドは、初版から公開している判定層のみの重みを読み込みます。上の表の全層LoRAの重みはまだ配布していません。[LORA.md](LORA.md) の手順で再学習すると `python predict.py lora-nli --adapter ...` で使えます。

Python 3.12 と CPU で実行できます。初回は Hugging Face から約644 MBの重みとトークナイザーを取得します。公開モデルの利用に API キーは不要です。

```bash
git clone https://github.com/nozomugit/laya-ja.git
cd laya-ja
python3.12 -m venv .venv
source .venv/bin/activate
python install.py
python predict.py jnli '整理券の所持者は全員入場できる。秋さんは持っていない。' '秋さんは入場できる。'
python predict.py jcommonsenseqa '鉛筆で書いた線を消すために使うものは？' '定規' '消しゴム' '封筒' '電卓' '切手'
python predict.py jcola '机の上に本がある。'
python predict.py jsts '男性が犬の散歩をしている。' '男性が犬を連れて歩いている。'
```

最初の例の正解は「判断できない」です。整理券を持つ人についての条件だけでは、持たない人が入れるかどうかは決まりません。公開した推論版は、この例を「矛盾」（約0.75）と誤判定しました。コマンドで実際の判定と確率を確かめられます。Windows では仮想環境を `.venv\Scripts\activate` で有効にできます。

| 読み込む重み | Laya 出力型 | 学習した判断 |
|---|---|---|
| `jnli` | choice | 前提から仮説が言える、矛盾する、判断できない |
| `jcommonsenseqa` | choice | 日本語の常識問題に五択で答える |
| `jcola` | noul | 文が JCoLA の容認可能性ラベル「1」と判定される確率 |
| `jsts` | score | 文の対の意味の近さを0〜5の期待値で返す |

Python からは `from laya_ja import load, query, nli_request`、続いて `agent = load("jnli")`、`query(agent, *nli_request(前提, 仮説))` で利用できます。内部では上流の `Agent.predict` が動きます。今回検証した出力は選択結果、二値確率、意味類似度の評点です。付随する action フィールドの再学習・評価は行っていません。用途に対応した質問文と尺度は `laya_ja.py` に収録しています。ローカルのベースモデルを使う場合は `python predict.py --model-path /path/to/base jnli ...` を使います。

## これまでの実験

外部モデルの併用や、判定層のみを学習した版など、ここまでの検証の記録です。

- [外部モデルを併用した日本語推論4課題](FOUR_TASKS.md)：Jamp 85.1%、JCommonsenseQA 89.9%、JSeM 79.4%、独自36問 91.7%。課題に応じて27Bと35Bの外部モデルを併用した成績です。
- [意味推論の追加検証](SEMANTIC_VOTE.md)：LAYA単体で編集箇所を学習した版と、外部モデルを併用する版の比較。
- [95%を目標にした追加実験](TARGET95.md)：Wikipedia読解、JaNLI、JSTS など。全課題での95%達成には至っていません。
- [Wikipedia 五択読解](GROUNDED_QA.md)：本文中から答えを選ぶ専用の `grounded-qa`。取り置いた70記事で 85.4%。
- [複数コーパスを使った追加実験](MULTICORPUS.md)：`jnli-specialist`、`multicorpus-nli`、`reading-qa`。
- [初版の記録](INITIAL.md)：判定層のみを学習した最初の公開版の成績と再現手順。

## 出典・利用条件

コードは Apache-2.0、配布する学習済みアダプターと独自評価例は CC BY-SA 4.0 です。ベースモデルは上流から取得します。データの著者・ライセンスの詳細は [NOTICE](NOTICE) を参照してください。

- [Laya upstream](https://github.com/NandhaKishorM/laya) · [Laya Multilingual model card](https://huggingface.co/convaiinnovations/laya-multilingual)
- [JGLUE — Kurihara, Kawahara & Shibata](https://github.com/yahoojapan/JGLUE)
- [JCoLA — Someya, Sugimoto & Oseki](https://github.com/osekilab/JCoLA)

文法判断の正解は JCoLA の収録文に付いたラベルです。方言、地域のことば、日常会話での自然さは本実験の評価範囲外です。また、JNLI の「判断できない」は前提と仮説の関係を示すクラスです。モデル自身の不確かさによる回答保留とは区別して扱います。
