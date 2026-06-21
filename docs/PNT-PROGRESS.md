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
