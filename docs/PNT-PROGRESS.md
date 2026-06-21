# prime-number-theorem (id 7) 進捗と次セッション引き継ぎ (2026-06-21)

> 新セッションはこれを読んで「議論の続き」と「実装の続き」に着手する。
> 安全プロトコルは [../CLAUDE.md](../CLAUDE.md)、前段の偵察は [SATURATION-SCAN.md](SATURATION-SCAN.md) が正本。
> 全体計画は [EINSTEIN-ARENA-PLAN.md](EINSTEIN-ARENA-PLAN.md)。

## いまの結論（最重要）

- **問題の本質**: PNT は「support（どの整数キーを使うか）を固定すれば、最適な係数 f は LP が厳密に一発で出る」問題。よって戦いは **support 選び（離散探索）** に圧縮される。f の最適化は解決済み。
- **honest best = 0.993708**（`squarefree-1600keys`、reach 2629、grid 全域 feasible @ RHS 1.0、オリジナル＝SOTA 不使用）。**arena top 0.994901 には gap −0.0012 で未達。**
- **なぜ届かないか（構造的）**: 2000キー予算が reach（max key）を縛り、reach が S を縛る。reach +600 ごとに S は約 +0.0003 しか伸びず、top まで +0.0012 = reach ~5000 相当が必要だが、2000キーでは squarefree でも reach ~3290 が限界。連続・平方数フリーといった**自然な数学族は 0.9937 付近で頭打ち**。

## ⚠ 未解決の論点（新セッションで継続したい議論）

**「arena top 0.994901 は同じ ≤2000キー制約で誰かが達成済み」→ first-2000-squarefree より良い 2000キー support が必ず存在する。** よって自然族の頭打ちは「最適でないだけ」で、top 超えの support はあるはず。それを **進化（support 探索）で見つけられるか** が open question。

オーナーの直感: 「安定して回る 1600keys で進化を繰り返せばいいのでは」。
- 正しい部分: 進化は support を探すので、非自明な良 support を見つける正攻法。
- 注意点1（reach 縛り）: 1600キーのままだと reach ~2629 で頭打ち＝top に届かない。詰めるには 2000キーをフルに賢く配る必要 → genome が大きく **評価が遅い**（`squarefree-2000keys` は単一 LP solve が 34分超で中断）。
- 注意点2（正直さ税）: arena top は **1.0001 の緩み／MC の穴**を突いている可能性。我々は grid 1.0 全域の厳しい基準を守っているので、gap の一部は「honesty 差」かもしれない（要検証）。
- 期待値（正直）: 並列進化で **~0.994 までは詰められそう、top 0.994901 超えは五分五分**。

**次セッションはまずこの「進化を回すか / banking するか」を決めるところから。** オーナーは「回せばいいのでは」寄り、アシスタントは速度の壁＋不確実性から banking 寄りで保留中だった。

## 数値結果（`results/prime-number-theorem/ceiling.json`）

| support | reach (maxK) | \|K\| | S | top まで |
|---|---|---|---|---|
| contiguous-1200 | 1200 | 1200 | 0.990720 | −0.004181 |
| contiguous-2000（=キー≤2000 の証明済み天井 U） | 2000 | 2000 | 0.993397 | −0.001504 |
| **squarefree-1600keys（honest best）** | 2629 | 1600 | **0.993708** | **−0.001193** |
| squarefree-2000keys | 3290 | 2000 | 中断（34分超） | 外挿 ~0.994 |

定理: 「キー≤2000 に限れば天井は連続 `{1..2000}` の LP 値 U」（部分集合⟹S≤U、かつ U 自体が合法提出）。実測 U=0.993397 < top。

## このセッションで作ったもの（コミット済み）

| commit | 内容 |
|---|---|
| `5014f38` | LP 基盤: cutting-plane LP + 二重検証（server 再生 + 整数グリッド厳密検証）|
| `b72f6b4` | 進化ハーネス: 島モデル、mutation/crossover、決定論、LLM proposer（lazy、default off）|
| `7910cfd` | LP perf: 種全[1,M] + violation 全追加（反復 14→5）|
| `d9a0fa2` | **LP perf: highspy warm-start（M=600 で 75→15s、~5×）** |
| `7d0838b` | 天井判定バッテリ + ceiling.json + review nit |

## 次の一手の具体（進化を回す場合）

**未実装の最大レバー = 並列化**（8コア、HiGHS simplex は1求解単スレッド → genome 評価を ~6 プロセス並列で ~5-6× wall-clock 短縮）。これが「脳筋で回す」を現実的にする鍵。手順:

1. `arena/pnt_evolve.py` の `run_evolution` に **multiprocessing 並列評価**を追加（各世代の children の `solve_support` を Pool で並列）。決定論は保つ（seed 固定、結果を index 順に回収）。
2. 種は **squarefree 系を主軸**（非平方数キーは μ=0 で LP が 0 にする＝変数の無駄。squarefree は変数 ~39% 減で速く、かつスコアも高い）。
3. `scripts/evolve_prime_number_theorem.py --key-max 3300 --time-budget <秒> --seed-ms ...` で time-budget 付き実行。
4. 各 best 候補は dual_verify（grid 全域 + server 再生は best のみ）。originality（mobius_distance、SOTA 不使用）+ provenance 記録。
5. **提出は絶対に承認ゲート経由**（CLAUDE.md）。agent identity 未決定。

別ルート（高レバレッジ・不確実）:
- **正直さ税の検証**: arena top の解を `get_best_solutions(7)`（read-only GET、比較専用・optimizer に入れない）で取得し、grid 1.0 全域で feasible か確認。MC ゲーミングなら「honest 世界では我々が上位」かもしれない。
- **最適 support の数学的特徴づけ** / 真の上限の双対証明。

## 既存の再利用資産

- `arena/pnt_lp.py` — `solve_support(K, *, solver="highs"|"scipy") -> LPResult`（warm-start cutting-plane、厳密グリッド feasibility 込み）。`grid_g`, `normalize_support`, `reconstruct_f1`, `objective_vector`。
- `arena/pnt_verify.py` — `dual_verify(f, K, S) -> VerifyReport`（server 再生 + 厳密グリッド + tail）。**server 再生は 1e7 MC × |K| で大 support だと重い**（best のみに使う）。
- `arena/pnt_evolve.py` — `run_evolution(seeds, EvolveConfig, proposer=None)`。genome=support frozenset、決定論。**並列化はまだ（次の一手）**。
- `scripts/solve_prime_number_theorem.py` — 単一 K ドライバ、`mobius_upto`/`squarefree_support`/`mobius_distance`。
- `scripts/ceiling_prime_number_theorem.py` — 天井バッテリ（`squarefree_first_n`、streaming 保存）。
- `scripts/pnt_llm_propose.py` — optional LLM proposer（`build_prompt`/`parse_supports` は純粋・テスト済み、`make_llm_proposer` は lazy anthropic）。
- tests: `test_pnt_lp.py` / `test_pnt_verify.py` / `test_pnt_evolve.py` / `test_pnt_lp_solver.py`（計 44 件）。

## 速度の現実（計測値）

- 律速は **LP 求解本体**（密制約行列 × n≈2000 変数）。grid 検証は 0.02s で無関係。
- M=600: scipy 75s → highs 15s。M=1000: highs 104s。**n=2000, x_max≈33000 は単一 solve で 34分超**。
- `dual_verify` の server 再生（1e7 MC × |K|）も大 support で重い → **天井判定では grid 検証で feasibility 判定、server 再生は best 1本だけ**にした（grid @1.0 は server @1.0001 より厳しい＝grid 通過なら server 受理）。

## 安全プロトコル（厳守）

GET は自由。**submit / register / thread は要承認**。スクリプトは dry-run 既定。API キーはリポジトリに置かない。提出系はこのセッションで一切発火していない。
