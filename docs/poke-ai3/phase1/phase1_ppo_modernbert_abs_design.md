# Phase 1 PPO + Absolute-Position ModernBERT Design

## 目的

`poke-ai3-python/python/poke_ai3_train/phase1_loop.py`の`dummy_gpu`と
`learn_dummy`を、Phase 1用の実際の推論・学習処理に置き換える。

最初の目標は、1v1同一Pokemon・2手だけの決定的ゲームで、PPO + GAEにより
Power80を選ぶ方策を学習できること。

この設計書では実装方針だけを定める。コード変更は別途判断後に行う。

## 決定済み方針

- 学習はPython/PyTorchで行う。
- Rust側の非同期executorはそのまま使う。
- モデルはModernBERTのTransformers実装を元に、リポジトリ内へvendorした改造版を使う。
- RoPEは使わない。
- Absolute Position Embeddingsを使う。
- 事前学習済み重みは使わず、ランダム初期化から学習する。
- Phase 1の報酬は勝敗のみ。
- 引き分け報酬は両者0。
- 学習中のMove選択はpolicyからサンプリングする。
- 評価/通常推論でもpolicy確率に従ってMove選択する。
- CUDA/FlashAttention未対応環境では、フォールバックせず明示的にエラー終了する。
- checkpointやtrajectoryなどのローカルデータは`data/poke-ai3/`配下に保存する。

## 依存関係

`poke-ai3-python/pyproject.toml`に以下を追加する想定。

- `torch`
- `transformers`
- `flash-attn`
- 必要になれば`accelerate`

`transformers`はバージョン固定する。ModernBERTのvendor元も、その固定バージョンの
ソースに合わせる。vendor時は元ファイルのライセンス表示と出典コメントを残す。

参考:

- Hugging Face ModernBERT docs: https://huggingface.co/docs/transformers/main/en/model_doc/modernbert
- Hugging Face Transformers ModernBERT source: https://github.com/huggingface/transformers/tree/main/src/transformers/models/modernbert

## ファイル配置案

```text
poke-ai3-python/
  python/poke_ai3_train/
    phase1_loop.py
    phase1_agent.py
    phase1_ppo.py
    phase1_encoding.py
    models/
      modernbert_abs/
        README.md
        configuration_modernbert_abs.py
        modeling_modernbert_abs.py
poke-ai3-python/
  data/
    checkpoints/
      phase1_latest.pt
    trajectories/
    logs/
```

既存の`phase1_loop.py`はCLIとRust executor接続を担当し、モデル、PPO、エンコード処理は
別ファイルへ分ける。1ファイルが大きくなりすぎないようにする。

## 入力表現

Phase 1の観測は、以下の5スロットに変換する。

```text
0: cls
1: my_hp
2: opp_hp
3: move_0_available
4: move_1_available
```

Rustから来る観測は概ね以下。

```json
{
  "game_id": 0,
  "player": "P1",
  "state": {
    "my_hp": 230,
    "opp_hp": 100,
    "available_moves": ["Power40", "Power80"]
  }
}
```

各スロットは、CLSだけをtoken embedding、状態スロットを連続特徴量からのprojectionとして扱う。
>User: clsはトークンにする
>Codex: 了解です。CLSはtoken embeddingにします。具体的には`cls_token_id`だけを持つ小さなembeddingを使い、その出力を先頭スロットに置きます。HPやmove availabilityはtoken id化せず、Linear/MLPでhidden vector化します。
- `cls`: `cls_token_id`をtoken embeddingへ入力する
- `my_hp`: HP値を正規化してLinear/MLPへ入力
- `opp_hp`: percent値を正規化してLinear/MLPへ入力
- `move_0_available`: 0/1をLinear/MLPへ入力
- `move_1_available`: 0/1をLinear/MLPへ入力

その後、5位置ぶんのAbsolute Position Embeddingsを足し、ModernBERT改造版encoderへ渡す。
token embeddingsはCLSのためだけに使う。token type embeddingsは使わない。
>User: token embeddingsをclsのために使う
>Codex: その方針に修正します。状態スロットまで全部token embeddingにすると、HPなどの連続量を離散化する必要が出るので、CLSだけtoken embedding、状態はfeature projectionという混合入力にします。

## モデル構造

大枠は以下。

```text
Phase1 observation batch
  -> Phase1EncoderInput
  -> cls token embedding + state slot feature projection
  -> + absolute position embeddings
  -> ModernBERT encoder without RoPE
  -> cls hidden state
  -> policy head: logits[2]
  -> value head: scalar
```

policy headは2手に対応する。

- index 0: Power40
- index 1: Power80

legal action maskを適用し、利用不能な手には非常に小さいlogitを入れる。
Phase 1では基本的に両方利用可能だが、終局状態や将来拡張のためmaskを入れる。

## ModernBERT改造方針

TransformersのModernBERT実装をvendorし、Phase 1に必要な範囲で最小改造する。

主な変更点:

- RoPE関連の生成・適用を取り除く。
- `input_ids`だけを前提にせず、CLS token embeddingと状態slot projectionを結合したhidden statesを受け取れる形にする。
- Absolute Position Embeddingsを追加する。
- FlashAttention経路を必須にする。
- 不要なMLM/head類は持ち込まないか、使わない。

最初は汎用Transformersモデルとして完全互換にすることを目指さない。
Phase 1 PPOで使うencoderとして安定して動くことを優先する。

## 推論処理

`dummy_gpu(data_chunk)`の置き換えとして、次の処理を行う。

1. `data_chunk["vec"]`をbatch化する。
2. 観測を5スロット特徴量へ変換する。
3. モデルからpolicy logitsとvalueを得る。
4. policy確率からactionをサンプリングする。
>User: 「masked categorical distributionからactionをサンプリングする」とは？
>Codex: 「使える手だけを候補にした確率分布からランダムに選ぶ」という意味です。例えばlogitsが`[1.0, 3.0]`ならsoftmax後の確率はPower80寄りになりますが、Power40も確率が0でない限り選ばれ得ます。もし片方の手が使えない状態なら、その手のlogitを非常に小さくして確率ほぼ0にします。Phase 1では通常どちらも使えるので、実質的にはpolicy softmaxからのサンプリングです。
5. 評価/通常推論でもpolicy確率に従ってactionをサンプリングする。
>User: 評価/通常推論でも普通にpolicy確率に従って選びたい
>Codex: 了解です。argmax方針は取り下げます。学習・評価・通常推論のいずれもpolicy確率に従って選びます。決定的な評価が欲しい場合だけ、将来オプションでargmaxを追加する余地を残します。
6. Rust executorへ返す形式に変換する。

返却形式は既存と合わせる。

```json
{
  "vec": [
    {
      "game_id": 0,
      "player": "P1",
      "policy": [0.0, 1.0],
      "win_rate": 0.5
    }
  ]
}
```

`policy`はRust側の現実装ではMove選択に使われる。PPO更新には、Python側で
サンプリングしたaction、log_prob、valueを別途trajectory照合用に保持する。

## trajectoryと学習データ

Rustから返るtrajectoryには、各stepのobservationとinferenceが含まれる。
ただしPPOには以下も必要。

- action
- old_log_prob
- value estimate
- reward
- done

action/log_prob/valueはPython推論時に保存し、Rustから返ったtrajectoryと
`game_id`, `player`, step順で対応付ける。

勝敗報酬はtrajectory終端で付与する。

- winnerと同じplayer: `+1.0`
- loser: `-1.0`
- draw/unknown: `0.0`
- 非終端step: `0.0`

## PPO + GAE

最小構成は以下。

- advantage: GAE(lambda)
- return: advantage + value
- policy loss: clipped PPO objective
- value loss: MSE
- entropy bonus: 小さく入れる
- gradient clipping: 入れる

初期ハイパーパラメータ案:

```text
gamma = 1.0
gae_lambda = 0.95
clip_epsilon = 0.2
value_coef = 0.5
entropy_coef = 0.01
max_grad_norm = 1.0
learning_rate = 3e-4
ppo_epochs = 4
minibatch_size = 64
```

Phase 1は短い決定的ゲームなので、まずは学習が動くことを優先する。
勝率やPower80選択率が伸びない場合に、ハイパーパラメータを調整する。

## checkpoint

デフォルトではcheckpointを読まず、新規モデルとして学習する。
checkpointから再開したい場合だけ、CLIで`--checkpoint-path`を明示する。
指定した場合は、そのファイルを読み込み、更新後も同じファイルへ保存する。

```text
data/poke-ai3/checkpoints/phase1_latest.pt
```

保存内容:

- model state dict
- optimizer state dict
- training step/epoch
- 設定値
- 直近の簡単な統計

`--checkpoint-path`未指定なら、既存checkpointがあっても読み込まないし保存もしない。
`--checkpoint-path`指定時にファイルが存在すれば読み込む。存在しなければランダム初期化し、
学習後にそのパスへ保存する。

## ログ

最低限、以下を標準出力または`data/poke-ai3/logs/`に記録する。

- epoch
- processed trajectories
- mean return
- win rate estimate
- Power80 selection rate
- policy loss
- value loss
- entropy

実験として残す必要がある長めの実行では、別途`experiments/poke-ai3/`に
目的、コマンド、結果を書く。

## FlashAttention必須チェック

学習開始時に以下を確認する。

- CUDAが利用可能
- `flash_attn`をimportできる
- モデルがFlashAttention経路を使う設定になっている
- 必要dtypeを満たす

満たせない場合は例外で停止する。eager attentionやCPUへのフォールバックはしない。

## 検証計画

実装後は以下を確認する。

1. `uv run phase1-dummy-loop --num-games 2 --chunk-threshold 4 --trajectories-threshold 2 --max-epochs 1`
   相当の新ループが起動する。
2. checkpointなしでランダム初期化から開始できる。
3. Rust executorとのJSON送受信形式が崩れていない。
4. 1回以上PPO更新が走る。
5. `--checkpoint-path`指定時、短い実行で`phase1_latest.pt`が保存される。
6. ある程度長く回すとPower80選択率が上がる。

リポジトリ全体の確認として、Rust側を変更した場合のみ以下も実行する。

```bash
cargo check --workspace
cargo test --workspace
```

Python側は`uv`で実行する。`pip`は使わない。

## 初回実装の非目標

- Phase 2以降の特徴量追加
- Showdownサーバー接続
- 事前学習済みModernBERT重みの流用
- 汎用Tokenizer対応
- CPU fallback
- 複数checkpoint履歴保存
- 大規模分散学習

## 実装順序案

1. `pyproject.toml`に依存を追加し、lockを更新する。
2. ModernBERT改造版をvendorする。
3. Phase 1のslot encodingを実装する。
4. policy/value model wrapperを実装する。
5. inference stateを持つagentを実装する。
6. trajectoryと推論時メタデータの対応付けを実装する。
7. PPO + GAE更新を実装する。
8. `phase1_loop.py`をdummyからagent呼び出しへ差し替える。
9. 短いコマンドで起動確認する。
10. 学習がPower80へ寄るか確認する。
