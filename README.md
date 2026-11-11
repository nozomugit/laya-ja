# Laya-JA

日本語推論および文法判断の学習済みアダプターを公開しています。JNLI の公式開発セット2,434問の正答率は、ベースの52.2%から72.1%に上がりました。独自36問では14問正解から12問正解に下がりました。JCoLA の成績と個々の候補の結果は `results/` を参照してください。常識五択と意味類似度の追加評価が進行中です。

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python install.py
python predict.py jnli '赤い箱を開けた。' '箱を開けた。'
```

出典・利用条件は [NOTICE](NOTICE) に記載しています。
