# 日本語の4課題で全問正答率80%を目指す実験

判定対象は前回と同じ Jamp 348問、JCommonsenseQA 1,119問、JSeM 384問、独自の推論36問。全件で一つの回答を返す。目標に必要な正解数は、それぞれ279、896、308、29。全課題は個別に判定する。

## 学習条件

- **時間関係**。日本語の数値・年月日時・期間を抽出し、2時点の大小や経過時間を特徴量にした。公式訓練データから開発・評価と前提が一致する行を除き、18本の決定木を訓練した。Jamp向けの既存LAYA判定器との配分と木の深さを開発192問で選んだ。訓練データ全体を使う版には、評価と同じ文型が含まれ得る。公式の厳しい文型分割で学習した版も、同じ評価348問に適用した。
- **常識五択**。LAYAの文脈分類器と、日本語の選択肢を一候補ずつ採点する公開のJev形式の判定モデルを使用した。公開モデルの訓練元にはJCommonsenseQAの公式訓練データが含まれるため、こちらの訓練側取り置き問題はそのモデルの独立評価には使えない。LAYAの確率10%、外部モデル90%を先に固定し、公式validation全1,119問を比較に使った。
- **意味推論**。同じ前提と仮説を、Qwen3.5-9Bに「含意・矛盾・判断できない」の順と、選択肢の順を変えた二つの固定指示で提示した。別構成としてQwen3.6-35B-A3Bに最大96トークンの推論を行わせ、同じ三択を採点した。LAYAとの配分はJNLIとJSICKから48問ずつ選んだ開発96問で選定した。クラスごとの判定の偏りを補正する候補は、開発用の片方のコーパスで学習し、もう片方に適用して選定した。
- **学習外の扱い**。JSeM・独自36問の正解、評価セット名、項目番号、言語現象の分類は推論器に渡していない。記事や文の事前学習時の混入は、外部モデルも含めて未確認。独自36問は以前から本リポジトリで公開している。JSeMは日本語の論理的含意問題から各正解クラス128問を取った固定評価である。

## 意味推論の独立検証

35Bモデルに、前提から仮説が**必ず真**か、**必ず偽**か、の二つを別々に尋ねた。各問の「はい」「いいえ」の対数尤度の差、通常の三択の尤度、LAYAの確率から三つの結論を採点する。係数の学習では、写真描写や二文の開発問題に加え、有限個の可能な状況から正解を導出した日本語144問を用いた。各問題の正解は3人×3属性の512通りの状況をすべて検査して決めた。合成問題は三つの語彙設定に分かれ、係数の正則化の強さは開発用5グループを順に取り置いて選んだ。

9Bの最初の評価で「判断できない」の見落としが多かったことを受けて試した方式である。従来の384問とは別に、これまで評価していない前提文を持つJSeMの96問（各クラス32問）を取り置いた。新旧セットの前提文の重複は0。新96問と4コーパスの訓練側の前提文にも一致はなかった。[分割と文型の監査](results/four-overlap.json) ／ [事前に固定した集計コード](https://github.com/nozomugit/laya-ja/commit/eaa705031148c7d92f9328d8ad2d2e660ac794e7)

## 利用

既存の導入手順でLAYAを用意し、追加依存を入れる。外部モデルは使う構成のものだけ取得する。各ファイルを公開元の固定リビジョンから取得し、SHA-256を照合する。

```bash
python -m pip install -r requirements-four.txt
python four_task_predict.py temporal '佐藤は準備に3日間を費やした。' '佐藤は準備を4日間以内に終えた。'
python four_task_models.py qa
llama-server -m .cache/four-models/qa/gguf/jev-Q4_K_M.gguf --embeddings --pooling last -np 8 -c 3072 -b 512 -ub 512 --port 18346
```

別端末で常識問題を解く。

```bash
python four_task_predict.py commonsense '板に書いたチョークの文字を消す道具は？' 黒板消し 巻尺 画びょう 水筒 時計
```

9Bの意味推論は `python four_task_models.py nli` を実行し、`llama-server -m .cache/four-models/nli/Qwen3.5-9B-Q4_K_M.gguf --port 18345` を起動する。別端末から使う。

```bash
python four_task_predict.py inference '青い封筒だけが三枚ある。' '赤い封筒が一枚ある。'
```

35Bを使う構成では `python four_task_models.py reason` で重みを取得する。CPU実行ファイルのビルド手順とllama.cppの固定コミットは[実験ワークフロー](.github/workflows/four-task-reason.yml)に掲載した。二つの外部モデルを必要とする配分の場合、CPU実行ファイルは順に起動する。

```bash
python four_task_predict.py inference-deep --engine ./score '青い封筒だけが三枚ある。' '赤い封筒が一枚ある。'
```

独立検証方式を一問試すには、同じCPU実行ファイルと35Bモデルを使う。

```bash
python four_task_predict.py inference-verify --engine ./score '青い封筒だけが三枚ある。' '赤い封筒が一枚ある。'
```

## 再現

[前回の手順](TARGET95.md)でデータとLAYAの文脈表現を準備した後、固定された入力と基準値を書き出す。

```bash
python four_task_data.py
python fit_temporal_relations.py
python fit_temporal_relations.py --training-file train_custom-template-hard.tsv --weights weights/temporal-hard.json.gz --result results/four-temporal-hard.json
python four_task_assist.py qa --head .cache/four-models/qa/gguf/head.npz --url http://127.0.0.1:18346 --splits qa-dev jcommonsenseqa
```

意味推論のCPU実験では、[入力照合プログラム](tools/four-task/prepare.py)が固定版コーパスから本文を復元し、前提と仮説のハッシュを全項目で検証する。評価器に渡すファイルは入力文とIDのみを含む。[9Bの実行](.github/workflows/four-task-evaluation.yml)と[35Bの実行](.github/workflows/four-task-reason.yml)が返す各分割のJSONLを、それぞれ `.cache/four-predictions/direct-out.jsonl`、`contrast-out.jsonl`、`reason-out.jsonl` に連結して集計する。量子化計算のバッチサイズやCPUが変わると、境界に近い項目の採点は変わり得る。

```bash
python four_task_report.py
python four_task_reason.py
```

独立検証は `python semantic_inputs.py` で開発用の有限世界と追加96問を用意する。[検証方式の実行](.github/workflows/four-task-semantic.yml)から得た3種類のJSONLを、`.cache/semantic-predictions/` に `must_true-out.jsonl`、`must_false-out.jsonl`、`direct-out.jsonl` として連結し、`python semantic_verifier.py` を実行する。

訓練質問の文字列検索を加えたLAYA常識分類器では、開発側で検索の重みが0になった。時間の上下限を手書きで判定する補助規則も開発側で正解を増やさなかった。[常識検索の比較](results/four-memory.json) ／ [区間規則の比較](results/four-intervals.json)

## 二つのモデルによる追試

従来の384問・独自36問・新たなJSeMの96問を同じ条件で評価した。Gemma 4 26B-A4Bには日本語で最大160トークンの推論を許した。Qwen3.6-35B-A3Bには、日本語の前提と仮説を提示し、仮説が真になる状況と偽になる状況を両方試すよう英語で指示し、最大256トークンを許した。各モデルとLAYAを90対10で混ぜる設定のほか、両モデル40%ずつ、LAYAと35B検証器を10%ずつとする共通設定も事前に固定した。[Gemmaの実行](https://github.com/nozomugit/laya-ja/actions/runs/43040117906) ／ [反例探索の実行](https://github.com/nozomugit/laya-ja/actions/runs/43040518093) ／ [統合方式の固定](https://github.com/nozomugit/laya-ja/commit/b75aa8bb610040e524f4265a1f5e45a21da73153)

反例探索の固定構成は独自36問のうち35問正解（97.2%）だった。唯一の誤答は「森さんに原さんが鍵を渡した」から「受取人は原さんだ」を含意すると判定した問題で、日本語の助詞が示す渡し手と受け手を逆に取った。同じ構成で元のJSeMは238/384（62.0%）だった。

各構成を新しい文で試せる。まず上の手順でCPU実行ファイルをビルドする。重みの並べ替え時にメモリが足りない場合は、実行前に `export LAYA_NO_REPACK=1` を設定できる。公開CPU実験は既定のメモリ設定で行った。

```bash
python four_task_models.py reason
python four_task_models.py gemma
python four_task_predict.py inference-countermodel --engine ./score '棚に残った陶器はすべて青い。そこに湯呑みが一つ残っている。その湯呑みは陶器だ。' 'その湯呑みは青い。'
python four_task_predict.py inference-gemma --single-judge --engine ./score '青い封筒だけが三枚ある。' '赤い封筒が一枚ある。'
python four_task_predict.py inference-cross-family --engine ./score '赤い印のある札を二枚以上選んだ。' '赤い印のある札を一枚以上選んだ。'
```

9Bだけを使う固定配分は `inference --direct --engine ./score`、35Bの96トークンの固定配分は `inference-deep --single-judge --engine ./score` で実行できる。`--engine` はLAYAを一度終了し、外部モデルを順にロードする。

判定結果は `python four_task_audit.py` で問題別に再集計できる。二つの追試では、ワークフローが返した8分割の `gemma.jsonl` と `countermodel.jsonl` をモデルごとに1本へ連結してから、次を実行する。

```bash
python semantic_verifier.py --outputs .cache/verifier.jsonl
python gemma_transfer.py
python countermodel.py
python cross_family.py
```

時間推論の厳しい文型分割でも追試した。人物ごとの出来事の前後関係を有向グラフにし、到達可能性から順序を判定した。日付・期間・習慣については、訓練文型ごとの信頼性が基準を満たした処理だけを採用した。訓練データは元の厳しい分割を維持し、決定木の251/348から278/348になった。規則が適用できない問題だけを35Bに渡す方式を事前に固定したところ、296/348（85.1%）になった。41の既知文型は194/194（100%）、38の未知文型は102/154（66.2%）。全訓練データを使う94.5%の構成では、これら38文型も訓練に含まれる。[文型の内訳](results/four-template-slices.json)[訓練文型ごとの適用結果](results/four-temporal-graph.json) ／ [適用外の問題を含む追試](https://github.com/nozomugit/laya-ja/actions/runs/43041671779)

```bash
python fit_temporal_graph.py
python four_task_predict.py temporal --graph 'マリはケンより先に到着した。ケンはユキより先に到着した。' 'ユキはマリより後に到着した。'
python four_task_predict.py temporal-transfer --engine ./score 'マリはケンより先に到着した。ケンはユキより先に到着した。' 'ユキはマリより後に到着した。'
python temporal_transfer.py
```

## LAYAだけでの意味推論

35Bの三択出力を教師として、LAYAの既存の二文表現に小さい線形の補正層を追加した。学習はJNLI・JSICK・有限世界の計240問で、正解ラベルと教師の確率を混ぜた。五つの開発グループを順に取り置いて正則化の強さを決めた。この構成は新しい質問に答えるときはLAYAだけで動く。独自36問は18問から20問に伸びた一方、JSeMは180問から166問に、追加96問は50問から47問に下がった。常用の意味推論モデルには採用していない。[学習・評価記録](results/four-native-distill.json)

```bash
python native_distillation.py
python four_task_predict.py inference-native '箱には赤い札が二枚以上ある。' '箱には札が一枚以上ある。'
```

27Bの密なモデルでも、同じ日本語の三択指示で直接採点した。LAYA 10%・27B 90%の単独併用と、LAYA 10%・27B 45%・35B 45%の固定配分を同じ意味推論セットで比較する。[配分を固定したコミット](https://github.com/nozomugit/laya-ja/commit/4711aec19d92b494aa11afdd44be5be1d610479f) ／ [27Bの実行](https://github.com/nozomugit/laya-ja/actions/runs/43046643705) ／ [問題別記録](results/four-dense.jsonl.gz)

```bash
python four_task_models.py dense
python four_task_predict.py inference-dense --single-judge --engine ./score '黄色い紙だけを三枚受け取った。' '黄色い紙を受け取った。'
python four_task_models.py reason
python four_task_predict.py inference-dense --engine ./score '黄色い紙だけを三枚受け取った。' '黄色い紙を受け取った。'
python dense_transfer.py
```

日本語で問い方と選択肢の順序を変えた判定も試した。35Bの3通りの指示を等配分し、LAYAを10%混ぜた。27Bを加えた方式は4つの外部判定を22.5%ずつに固定した。独自36問にも同じ方式を用いた。[配分と指示を固定したコード](https://github.com/nozomugit/laya-ja/commit/8a9d8ab01841a30675b4def50dced8d6f7d4edbc) ／ [実行記録](https://github.com/nozomugit/laya-ja/actions/runs/43049153900) ／ [問題別結果](results/four-readings.jsonl.gz)

```bash
python four_task_predict.py inference-readings --engine ./score '妹は姉より背が高い。' '姉は妹より背が低い。'
python reading_vote.py
```
