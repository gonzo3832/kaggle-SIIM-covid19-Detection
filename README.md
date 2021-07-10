#  はじめに
### kaggleを始めた理由
- 楽しみながら機械学習のスキルを向上させたい
- 建設的な趣味が欲しい

### kaggle日記を書く理由
- モチベーション維持
- 学んだことの整理（ラベリングとストック）

参考：https://zenn.dev/fkubota/articles/3d8afb0e919b555ef068

# kaggle-SIIM-covid19-Detection
![B0DD58C1-2752-4E7F-85CB-A59A1A3A90CE](https://user-images.githubusercontent.com/64478157/123980887-ddbbf200-d9fc-11eb-8677-ef24fb887df2.png)

```mermaid
gantt
  title timeline
  dateFormat YYYY-MM-DD
  section Official
  Competetion: a1, 2021-05-18, 2021-08-10
  Entry deadline: a3, 
  Team Merger deadline: a4, 
  Final submission deadline: a2, 

  section Score
  Join!:2021-06-15, 2020-06-16
```
↑ガントチャートをマークダウンで書きたいマン

## Dataset


## 210630
- 日記初めてみた
- couseraのpytorchを使った[画像分類チュートリアル](https://www.coursera.org/learn/covid-19-detection-x-ray/home/welcome)を完了
  - 基本的な流れは理解（なんとなく）
    - Dataset　classの作成→画像処理クラスの作成　→データの読み込み→ →model classの作成　→　train classの作成（optimaizer 何使うかとか決める） → modelに食わせる
  - Resnet18,,,? よくわからん状態,,,画像処理への理解が浅い、なんかで補強しなければ
  - てかこれBoxDetectionやないよね？
  - [pytorchでBoxDetection実装している人がいた](https://lsifrontend.hatenablog.com/entry/2019/12/20/195244)ので、とりあえず一発実装してみる（土曜中に終わらせたい、、）
  - pytorch は直感的、カスタマイズ性が良い等評判が良いので、メインで学習したい
- 気になっていること
  - 　コンペのDiscussion見ると、VOLO法のvote数が多い。pytorch で実践できるかな、、
- 学んだこと
  - [最適化メソッド](https://qiita.com/omiita/items/1735c1d048fe5f611f80)←神記事
  - [ミニバッチ学習](https://ai-trend.jp/basic-study/neural-network/sgd/)←学習の停滞が起こりづらい、と聞いて理解。データ数が少ないと、コスト関数の平面がより凸凹して落ち着きづらいイメージかな
- その他
  - [ResNetのまとめ記事]（https://qiita.com/ikeyasu/items/ea9ced2b8e0fcb3da2be）

## 210701
- 昨日のpytorchのbox Detectionは、そもそも今回の学習に使用できるか不明なので、[このお方のnotebookを参照してみる](https://www.kaggle.com/heyytanay/siim-pytorch-classification-only-training-effnets)
  - dicomモジュールでデータ読み込み
  - 8bit イメージデータに変換
  - model class（　EfficientNet model)の定義×２、Trainer classの定義、main functionで回す
    -　EfficeintNet  modelについて、[このサイト](https://qiita.com/Radley/items/e6cd148079468dbdb616)が詳しそう、明日勉強する
-　素朴な疑問
  -　notebookを量産するのは良いが、管理がめんどくさそうだ、、、
  -　自分の作ったnotebookを見返さないので、内容を忘れがち→fkubotaさんみたいに、githubで一元管理した方が良さそう？
  -　notebookの立ち上げ→データセットの読み込み　の手順が毎回めんどくさい
  -　実験計画をどう効率的に、戦略的に行うか考える必要がありそう。pipe lineってのがヒントっぽい。fkubotaさんのリポジトリ漁ってみる
  今日は残業おじさんだったのでここまで、、、
  した内容
## 210702
- fkubotaさんのリポジトリ漁った
  - EDAからスタートしてて、全ての特徴量についてコメントして記録に残している。最初に作ったノートブックで一通り理解したつもりになっていた。もう一度最初からやろう。
  - 写経したnotebookにコメント残して資料化している。同じ事やって学習ログを残す。
- 気づいた事
  - Readmeここで編集するのめんどい。ローカルリポジトリ上のファイルを好きなmarkdownエディタで編集して、git pushで更新するスタイルにしよう。
  - notebookのショートカットは覚えるべき
  - 写経が終わったら、最初に作ったdata preparationのノートブックをメンテして、EDA、読み込み、等に分けて、深堀する
  - その後ゼロつくのCNNの章に挑戦する。
   どうてもいいけど今日コロナワクチン受けた、、、腕痛い
    あと、notion使い始めた
    クロームのタブ開きすぎ問題を解決したい。毎日閉じるようにしようかな。
## 210703
- gitのお勉強した
- ステージとローカル、リモートリポジトリの関係等抑えてれば使えそう
## 210704
- gitで勉強した内容をnotionにまとめた
- notionおしゃん。kaggleの情報もここに整理しよう
## 210705
- この日は特に進捗なし
## 210706
- 公開notebookのEDAを自分なりにまとめようとする
	- dicomのヘッダ情報など、よく理解できていない
  	残業ありすぎる
- kaggleの環境をlocalに再現して実験を高速で回したい
	- dockerを半日で習得する

## 210707

- 七夕
- 定時だったので[Dockerで環境作ってみた](https://www.notion.so/Docker-9403bc842113451a93c95cafbfc5989d)
  - できたは良いが、思ったよりvscodeのjupyter使いづらい、、、

## 210708

- 残業地獄でキレそう
- 前に一通り回したnotebook見返した。
  - 画像の読み込みからEDAまで一通りやっている。とりあえず資料として残す。

## Dataset

