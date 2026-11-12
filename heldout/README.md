# 独自評価問題

Laya-JA の学習に使っていない問題で実力を測るための1,000問です。公開用200問（`dev.jsonl`）と、封印用800問（`test.jsonl`、このリポジトリには置かない）に分かれます。

| 区分 | 出力 | 公開用 | 封印用 |
|---|---|---:|---:|
| 推論（24分野） | 含意・矛盾・判断できない | 144 | 576 |
| 実務判断・選択 | 四択 | 24 | 96 |
| 実務判断・二値 | はい／いいえ | 16 | 64 |
| 実務判断・評点 | 0〜5 | 16 | 64 |

- **問題文と正答はAIが作成した候補版です。** 作成者と別の者による独立した解答と、人による監査はまだ行っていません。正答に誤りが残っている可能性があります。
- **作成したAIは、Laya-JA の学習データを設計した側とは別です。** 学習用の生成器や既存コーパスは渡していません。
- **封印用の問題文は、モデルを開発する側が読みません。** 評価は合計と分野別の数字だけを出すスクリプトで行います。誤答の分析には公開用を使います。
- **どちらも学習には使いません。**
- 版は `SHA256SUMS` で固定しています。評価スクリプトは、ファイルがこの版と一致しないと止まります。

学習データとの重なりは、文字2-gram の Jaccard 類似度で照合しました（`heldout_overlap.py`）。0.5以上の問題は公開用・封印用とも0問で、最大は公開用 0.329、封印用 0.352 でした。[照合の結果](../results/heldout-overlap.json)

各行が1問です。`type` は `nli`、`choice`（1始まりの選択肢番号）、`noul`（真偽値）、`score`（0〜5）。モデルに渡すのは `state`、`question`、必要な場合の `options` だけです。

```bash
python heldout_eval.py --checkpoint path/to/unified --upstream --out results/heldout-dev.json
python heldout_eval.py --problems path/to/test.jsonl --checkpoint path/to/unified --upstream --out results/heldout-test.json
```
