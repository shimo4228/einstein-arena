# prime-number-theorem (id 7) 進捗と次セッション引き継ぎ

> 新セッションはこれを読んで「議論の続き」と「実装の続き」に着手する。
> 安全プロトコルは [../CLAUDE.md](../CLAUDE.md)、全体計画は [EINSTEIN-ARENA-PLAN.md](EINSTEIN-ARENA-PLAN.md)。
> 設計の詳細は plan `~/.claude/plans/replicated-singing-sprout.md`（+ agent 版）。

## いまの結論（最重要・2026-06-21 セッション2 で更新）

- **問題の本質**: support（どの整数キーを使うか）を固定すれば、最適な係数 f は LP が厳密に一発で出る。戦いは **support 選び**に圧縮される。LP 部分は解決済み資産。
- **arena top 0.994901（OrganonAgent）は別フレームではない**: read-only GET で取得・解剖した結果 = **2000キー・全 squarefree・reach 3498・LP 的に最適化された f**（μ とは 0/2000 一致）。チームと同じフレーム。
- **gap の正体**: チーム honest best 0.993708 → top 0.994901（+0.001193）の **92% は reach/予算、8%（+0.0001）は正直さ税**（top は RHS 1.0001 の緩みを使用、g_max=1.0000990）。チームが 1600キーで止まったのは数学的限界でなく **2000キー LP が34分超で解けない計算問題**。
- **効くレバーは「キー物量」でなく「support 構造」**: 同じキー予算で **multiscale（密な小キー + 幾何的に疎な大キーで reach 延伸）が squarefree-first を上回る**（下表）。これがオリジナリティの本命。
- **colgen（双対誘導 rc 貪欲）は multiscale に勝てない**（実証済み・棚上げ）。rc は reach 延伸の便益を見えず、近視眼的に小キーを優先するため。→ **本命は multiscale 構造そのものの最適化**。

## 数値結果（honest = RHS 1.0、`solve_support`）

| support | reach | \|K\| | honest S | top まで |
|---|---|---|---|---|
| squarefree-first-600 | 986 | 600 | 0.988901 | − |
| **multiscale-600** | 2501 | 600 | **0.991717** | − |
| squarefree-first-1200 | 1977 | 1200 | 0.992884 | − |
| **multiscale-1200** | 3502 | 1200 | **0.994153** | **−0.00075** |
| squarefree-1600（旧 honest best） | 2629 | 1600 | 0.993708 | −0.001193 |
| multiscale-2000（reach 4501） | 4501 | 2000 | （計算中） | 期待 ~0.9948-0.9952 |
| **arena top（OrganonAgent, RHS 1.0001）** | 3498 | 2000 | 0.994901 | 0 |

**multiscale-1200 はわずか1200キーで honest 0.994153 ── 旧 honest best（1600キー 0.993708）を超え、top まで honest であと 0.00075。** N=2000 で top 超えが射程。

## このセッション（2）で作ったもの

| モジュール | 役割 | テスト |
|---|---|---|
| `arena/pnt_warm.py` | **WarmLP**: 永続 warm-start HiGHS、双対抽出・キー add/drop。`solve_support` を完全再現（無回帰で内部置換済み） | 10 |
| `arena/pnt_seeds.py` | `multiscale_support` / `candidate_pool` / `squarefree_first_n` / `arena_distance` | 6 |
| `arena/pnt_colgen.py` | 双対誘導 grow colgen（**棚上げ**: multiscale に劣後） | 6 |
| `scripts/fetch_pnt_arena_compare.py` | arena top 取得（read-only、originality 比較用） | — |
| `scripts/pnt_colgen_probe.py` | プロトタイプ（go/no-go 比較。multiscale vs squarefree） | — |

**全66テスト緑・ruff clean。** WarmLP は LP 高速化（warm-start）の土台で、次の multiscale 最適化に再利用する。

## 次の一手（open work）

**multiscale 構造の最適化で honest 0.994901 を超える**:

1. **multiscale-2000 の reach パラメータ最適化**: target_reach（と dense_upto）を変えて honest S を最大化。各 cold solve は重い（n=2000, x_max 数万で30-90分）。
2. **WarmLP で高速化**: 1回 cold solve した後、tail キーの add/drop（warm）で近傍 reach を高速評価 → reach スイープを現実的に。`arena/pnt_warm.py` の `add_key`/`drop_keys` 済み。これが「34分問題」を回避する本筋。
3. **検証**: best 候補は `arena/pnt_verify.py::dual_verify`（grid@1.0 全域 + server 再生、local==server）で honest 確認。
4. **originality**: `arena_distance`（取得済み OrganonAgent 解との support Jaccard + 共有キー L-inf）+ `mobius_distance`。multiscale は OrganonAgent の全密 squarefree とは構造が違うので originality は確保しやすい。
5. **提出は絶対に承認ゲート経由**（CLAUDE.md）。agent identity 未決。submit 系はこのセッションで一切発火していない。

棚上げ案（低優先）: colgen の swap フェーズ（multiscale-2000 を seed に個別キー swap で微改善）。grow が劣後したので投機的。

## 速度の現実（計測値）

- 律速は **LP 求解本体**（密制約行列 × n≈2000）。grid 検証は無関係。
- M=600: highs 15s。M=1000: 104s。**n=2000, reach 3290（x_max 32900）で34分超**。reach 4501（x_max 45000）はさらに重い。
- 対策: WarmLP warm-start（support 編集をまたいで再構築しない）+ cut-drop（slack 行除去、未実装）。

## 🏆 提出 → #1 達成（2026-06-21 セッション2、人間の明示承認 "go" 済み）

- **登録**: 永続エージェント **`Agent-Knowledge-Cycle`** を作成（PoW difficulty 25）。api_key は `~/.config/einsteinarena/credentials.json`（repo には置かない）。
- **提出**: problem 7 に honest **0.9955806**（検証済み record）を投稿。
  - id 2359: error（120秒タイムアウト、variance）。
  - **id 2360（再投）: evaluated, score 0.9955806006360862 → leaderboard #1**（2位 OrganonAgent 0.994901 を honest で +0.00068 上回る）。
- **サーバ採点知見**: verifier 120秒タイムアウト（E2B sandbox）。2000キー×1e7 MC は縁で variance あり（1回目落ち2回目通った、採点開始まで queue ~8.5分）。
- **thread（学びの共有）は未投稿**（後日検討、要承認）。arena の本質は thread の洞察（skill.md）。

## 安全プロトコル（厳守）

GET は自由。**submit / register / thread は要承認**。スクリプトは dry-run 既定（`scripts/submit_pnt_record.py` は `--i-have-approval` でのみ発火）。API キーはリポジトリに置かない。**今回の登録・提出は人間の明示承認 "go" を得てから発火した。** thread はまだ発火していない。

## 🏆 奪還キャンペーン → #1 再達成（2026-07-03/04、承認 "go" 済み提出 id 2383）

**背景**: 前回 #1（0.9955806）が #4 に後退。解剖（read-only GET）の結果 — CHRONOS 0.9965177307
（連続整数 2..1501 + 幾何 tail、reach 24000、緩み 1.0001 使用。honest 2位解 0.9962211@16k も保有）、
MAOJIASONG 0.9963197（squarefree 系 1675 キー、reach 16000、緩み使用）、Hyra 0.9956802
（**我々の解の丸コピー ×1.0001**、jaccard=1.000、linf vs ours×1.0001 = 2.7e-9）。
同 reach では構造細部の差がほぼ消える（広いプラトー）→ **支配レバーは reach** のまま。

**deficit 則**: 実測から 1−S ≈ c/ln(10·reach)。incumbent 解では c≈0.0444–0.0474 だが、
我々の multiscale では **c = 0.0392 (16k) → 0.0370 (24k) → 0.0362 (32k) と単調低下**
（= multiscale の構造優位は reach とともに拡大する）。

**エンジン改修**（このキャンペーンで実装、全テスト 115 本緑）:
- `grid_g_fast`（約数増分 + cumsum、O(x_max·Σ1/k)）— cut 候補探索専用。**収束 certify は必ず exact `grid_g`**（fast の丸めは最終結果に混入しない設計）
- cut-drop（`drop_slack_rows`: slack>tol・min_age 経過・非 pin 行を削除。再入 x は pin → 有限終了保証）+ `LedgerDriftError`（帳簿乖離の型付き例外）
- **層化 cut 選択**（`cut_window`: x 軸窓ごと worst-1 本）— 同 S で行数 3668→1035、wall 半減
- **大差分遷移は warm 編集しない**: セル間で 2000 列中 25–40% が入れ替わり、HiGHS の basis が壊れて "warm" 初回 solve が病的化（12 分超スタック実測）→ >15% swap で fresh モデル再構築 + **cut 行だけ移植**
- `near_binding_rows`: ledger の解 f から g(x)>1−ε の x を復元 → プロセス再起動を跨ぐ cut 知識の移送
- `scripts/pnt_reach_sweep.py`: JSONL 追記 + fsync のレジューム型スイープドライバ（success is True のみスキップ）
- 実験で棄却: IPM per-solve（warm 不可で 6.6 倍遅い）、HiGHS parallel オプション（即時 fail）

**reach ラダー実測（D\*=1800、honest RHS=1.0、全セル exact grid 検証済み）**:

| reach | honest S | coeff_c | 対 CHRONOS 0.9965177 |
|---|---|---|---|
| 16000 | 0.9967279 | 0.0392 | +0.00021 |
| 24000 | 0.9970172 | 0.0370 | +0.00050 |
| 32000 | **0.9971452** | 0.0362 | **+0.00063** |

D 選定: D=2300@16k は LP 上界（iter9 で 0.9967122 < D=1800 の確定値 0.9967279）で敗退確定 →
完走不要で D\*=1800。D 依存はごく小さく（≥1.6e-5）、reach が支配的。

**提出**: record v2（`multiscale_record_v2.json`、dual_verify: server 原典再採点一致 + 整数グリッド
[1,320010) 全数 g≤1.0、n_violations=0）→ dry-run → 人間承認 "go" → **id 2383: evaluated,
score 0.9971452043881762（ローカル検証値と一致）→ leaderboard #1**（1 回で通過、queue ~10 分）。
originality: support_jaccard 0.31 (CHRONOS) / 0.35 (MAOJIASONG) / 0.47 (Hyra)。

**レビュー**（並列 3 本、全指摘修正済み）: python-reviewer HIGH=budget-hours の時刻比較フレーク
（time.monotonic + <=0 特別扱いで修正）、security-reviewer MEDIUM=NaN/Infinity スコアによる
ゲート無力化（`math.isfinite` フィルタ 3 箇所 + check_gate 防御）、codex-review（cross-model）が
NaN を独立検出 + record 不在時 UX / resume 厳格化。LP 5 不変条件（exact certify / 帳簿 /
三重ゲート / resume / RHS=1.0 非漏洩）は python-reviewer が個別検証で健全確認。

**open work**:
1. **48k セル**（S≈0.99723+ 見込み、coeff_c 低下継続ならさらに上）— マシン低負荷時間帯に
   detached（nohup/setsid + caffeinate）で実行 → 上振れしたら**再承認を取って**再提出。
   セッション紐付きバックグラウンドは外部停止が 3 回発生したため detached 推奨。
2. cap=1900 保険セル（サーバ 120s タイムアウト対策。今回は 1 発通過したので低優先）。
3. thread 更新（reach スケーリング則 c の単調低下 = 構造優位の拡大、の共有）— **要承認**。
4. fine-tune D±300 は縮小判断（D 依存 1.6e-5 ≪ margin 6.3e-4 のため費用対効果低）。
