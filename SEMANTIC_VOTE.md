# 日本語の意味推論を同じ問題で再評価する

今回採用したのは、LAYA 10%・Qwen 27B 45%・Qwen 35B 45%で三択の確率を合わせる方式。JSeMの固定384問で305問正解（79.4%）となり、従来の35B併用より15問増えた。80%に必要な308問まであと3問。独自36問は33問正解（91.7%）だった。

従来の固定JSeMは384問、前提文を分けた追加のJSeMは96問、独自の推論問題は36問。各構成は全問に一つずつ回答する。Jampは厳しい訓練分割で296/348（85.1%）、JCommonsenseQAは1,006/1,119（89.9%）。この二つは前回の公開測定値を参照した。

| 推論方式 | 固定JSeM | 追加JSeM | 独自推論 |
|---|---:|---:|---:|
| 従来のLAYA単体 | 180/384（46.9%） | 50/96（52.1%） | 18/36（50.0%） |
| LAYA単体・4アダプターと共有判定層 | 181/384（47.1%） | 54/96（56.2%） | 20/36（55.6%） |
| LAYA単体・文の編集箇所を追加学習 | 194/384（50.5%） | 51/96（53.1%） | 19/36（52.8%） |
| LAYA＋Gemma 26B | 170/384（44.3%） | 49/96（51.0%） | 22/36（61.1%） |
| LAYA＋Qwen 35Bによる反例探索 | 238/384（62.0%） | 68/96（70.8%） | 35/36（97.2%） |
| Gemma・反例探索・検証器・LAYA | 217/384（56.5%） | 68/96（70.8%） | 30/36（83.3%） |
| LAYA＋Qwen 27Bの直接判定 | 293/384（76.3%） | 74/96（77.1%） | 34/36（94.4%） |
| LAYA＋Qwen 27B・35Bの直接判定、固定配分 | 305/384（79.4%） | 79/96（82.3%） | 33/36（91.7%） |
| LAYA＋4種類の外部判定、固定配分 | 272/384（70.8%） | 76/96（79.2%） | 31/36（86.1%） |
| LAYA＋4種類の外部判定、開発問題で配分を学習 | 288/384（75.0%） | 76/96（79.2%） | 34/36（94.4%） |
| LAYA＋直接判定、開発問題で配分を学習 | 290/384（75.5%） | 77/96（80.2%） | 34/36（94.4%） |
| LAYA＋直接判定、不一致時のみ順序を入れ替えた二択 | 299/384（77.9%） | 78/96（81.2%） | 34/36（94.4%） |

以前の35B併用（290/384）と固定配分の直接判定（305/384）を問題ごとに照合すると、22問が誤答から正答へ、7問が正答から誤答へと変わった。異なる367前提を単位に5,000回再標本化した差の95%区間は+1.3〜+6.6ポイント。複数構成を比較した探索的結果である。

固定配分での主な誤りは、接続（77/97正解）、副詞（25/36）、助動詞（6/15）に残った。独立した開発240問のうち二つの直接判定が割れた39問で、「必ず真か」「必ず偽か」という二つの検証結果に掛ける重みを学習すると、選ばれた重みは0だった。

## 新しい前提文を使った検証

従来の2評価セットとも前提文が一致しないJSeMから、含意・矛盾・中立を各12問選んだ。残った候補を公開IDのハッシュ順に並べて選び、モデルの正誤で採否を決めていない。JNLI、JSICK、JaNLI、Jamp（厳しい訓練分割）の訓練側の前提文とも、空白と全半角を正規化して一致が0件であることを確認した。元モデルの事前学習履歴は不明。これは36問の追加検証である。追加検証は固定版のCPUスコアラーで4問ずつ、3スレッド、LAYA_NO_REPACK=1を指定して行った。

| 方式 | 新しいJSeM 36問 |
|---|---:|
| 従来のLAYA単体 | 19/36（52.8%） |
| LAYA＋35B直接 | 30/36（83.3%） |
| LAYA＋27B直接 | 27/36（75.0%） |
| LAYA＋27B・35B、固定配分 | 29/36（80.6%） |
| LAYA＋直接判定、開発問題で配分を学習 | 26/36（72.2%） |
| LAYA単体・4アダプターと共有判定層 | 20/36（55.6%） |
| LAYA単体・文の編集箇所を追加学習 | 19/36（52.8%） |
| LAYA＋直接判定、不一致時のみ順序を入れ替えた二択 | 26/36（72.2%） |

## 学習・比較条件

文の編集箇所を使う方式では前提と仮説を文字単位で並べ、挿入・削除・置換箇所の文字列とその長さを特徴量にした。JNLI 4,096問、JSICK 1,536問、Jamp 1,536問、合成論理1,800問、計8,968問で線形分類器を学習した。開発240問で選んだ混合比はLAYAの三択出力50%・編集特徴50%。4アダプター方式は同じエンコーダー出力に4個の小型判定層と共有判定層を適用し、確率の平均を補正する。どちらも推論時に外部言語モデルを使わない。編集特徴で現行の共有判定層を180問から194問正解へ改善した一方、以前に公開した別のLAYA単体構成の204/384（53.1%）には届いていない。[初期構成の比較](MULTICORPUS.md)

外部判定の補正にはJNLI・JSICK各48問と、有限の状況を列挙して正解を生成した日本語144問を使用した。開発データを5グループに分け、各グループを一度ずつ除外して温度とクラスごとの偏りを推定し、その取り置き予測から混合比を学習した。正解ラベルによる補正と配分学習は開発データ内で完結する。JSeMと独自36問には何度も評価に使ってきた公開問題が含まれる。

固定配分はLAYA 10%、35B直接25%、27B直接25%、Gemma 30%、35B反例探索10%。順に判定し、それまでの1位と2位の差が残りの全配分を超えれば、残りの判定器の回答にかかわらず最終選択が確定する。公開入力756件で、打ち切った判定と全判定を計算したラベルの一致を照合した。モデル呼び出し件数を結果JSONに収録した。実行時間の短縮率は測定していない。

編集箇所とアダプターの予測を対数確率でさらに統合する方式も開発データで学習した。固定JSeMでは180/384にとどまった。また、外部モデルの直接判定の幾何平均と編集特徴の組み合わせは、開発データでの対数損失が4候補中最も低かったが、JSeMは304/384だった。

## 意見が分かれた問題の再判定

固定配分の27Bと35Bが同じラベルを選べば、そのまま採点する。答えが割れたときだけ、その2ラベルを日本語の二つの解釈として35Bに提示する。A・Bを入れ替えた2通りで、それぞれA対Bの対数尤度差を計算する。順序の影響を相殺した差を±4に制限し、0.5倍を元の2候補の対数オッズに加える。残った第3候補の確率は保存する。係数は固定。全792問中114問で再判定し、両方の提示順で計228個の得点を取得した。予測器への入力は文章と元の判定のみで、正解や現象の分類は渡していない。

固定JSeMの54件の不一致を再判定し、固定配分から誤答→正答が4問、正答→誤答が10問。最終成績は299/384（77.9%）。追加96問は78/96（81.2%）、新しい36問は26/36（72.2%）だった。

再判定の実行では、最初の84プロンプトを4問ずつ・3スレッドで計算した。メモリ不足で中断したため、完了した問題をIDで照合し、残り144件は2問ずつ・5スレッドで再開した。集計時には全IDを照合し、欠落と重複が0件であることを確認した。CPUやバッチの違いで境界付近の判定が変わる可能性はある。

## 試す

READMEの環境を用意し、追加依存をインストールする。外部モデル併用版は使用する重みを取得し、[CPU実行ファイルのビルド手順](.github/workflows/four-task-reason.yml)に従って `score` を用意する。

```bash
python -m pip install -r requirements-four.txt
python native_edit_nli.py predict '青い箱が二つある。' '箱が一つ以上ある。'
python four_task_models.py dense
python four_task_models.py reason
python semantic_vote.py '青い箱が二つある。' '箱が一つ以上ある。' --engine ./score --direct-pair
```

再判定を使うには `python semantic_pairwise.py 前提文 結論文 --engine ./score` を実行する。

27Bのみを併用する場合は `--dense-only`、開発データで学習した直接判定の配分を使う場合は `--calibration weights/semantic-vote-direct.json`。4種類の外部判定を使う場合はGemmaの重みも `python four_task_models.py gemma` で取得する。

## 追試と集計

[既存のデータ生成手順](FOUR_TASKS.md)で本文とLAYA出力を準備する。Gemma、35B反例探索、27Bの実行では各8分割を連結し、順に `.cache/gemma.jsonl`、`.cache/countermodel.jsonl`、`.cache/dense.jsonl` に置く。35Bの直接判定には既存の検証実験の `.cache/semantic-predictions/direct-out.jsonl` を使う。[Gemmaの記録](https://github.com/nozomugit/laya-ja/actions/runs/43040117906) · [35B反例探索の記録](https://github.com/nozomugit/laya-ja/actions/runs/43040518093) · [27Bの記録](https://github.com/nozomugit/laya-ja/actions/runs/43046643705)。モデルの比較条件と集計値を公開し、下記のコマンドでローカルの採点結果を再集計できる。

```bash
python fit_vote_calibration.py
python fit_vote_calibration.py --direct-only --output weights/semantic-vote-direct.json
python evaluate_semantic_vote.py
python evaluate_semantic_vote.py --calibration weights/semantic-vote.json --result results/semantic-vote-calibrated.json --rows results/semantic-vote-calibrated.jsonl.gz
python evaluate_semantic_vote.py --calibration weights/semantic-vote-direct.json --result results/semantic-vote-direct.json --rows results/semantic-vote-direct.jsonl.gz
python audit_direct_vote.py prepare
./score .cache/four-models/reason/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf .cache/vote-audit/direct.jsonl .cache/vote-audit/direct35-out.jsonl 4 3 0
./score .cache/four-models/dense/Qwen3.6-27B-Q3_K_S.gguf .cache/vote-audit/direct.jsonl .cache/vote-audit/direct27-out.jsonl 4 3 0
python audit_direct_vote.py score
python evaluate_pairwise.py prepare --audit .cache/vote-audit
./score .cache/four-models/reason/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf .cache/pairwise-in.jsonl .cache/pairwise-out.jsonl 2 5 0
python evaluate_pairwise.py score --audit .cache/vote-audit
```

[固定配分の事前コミット](https://github.com/nozomugit/laya-ja/commit/69041b9f55fd14e392f228ed1ecd912c2d232e78) · [開発用の補正手順](https://github.com/nozomugit/laya-ja/commit/088171fcad6143a461b8156c74751b2cdfd31d01) · [直接判定の比較と新36問の設計](https://github.com/nozomugit/laya-ja/commit/50583b18a4c9623b9244e9e10417bcd7f7be3698)
