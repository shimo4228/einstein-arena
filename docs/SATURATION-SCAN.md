# 飽和スキャン結果と次セッションへの引き継ぎ (2026-06-21)

> このファイルは新セッションが読んで Phase B step 2 以降（進化ハーネス構築）に着手するための引き継ぎ。
> 安全プロトコルは [../CLAUDE.md](../CLAUDE.md)、全体計画は [EINSTEIN-ARENA-PLAN.md](EINSTEIN-ARENA-PLAN.md) が正本。
> 再実行: `uv run python scripts/saturation_scan.py`

## ここまでの結論（重要）

- 標準的なローカル探索（勾配降下 + seed 精錬）は**飽和問題で再現止まり**。min-dist / Erdős / 3rd-autocorr の3問で実証済み。
- 我々の「タイ解」は**オリジナルではない**（Erdős/3rd-autocorr は `/api/solutions/best` から DL した他者の解、min-dist は FM Agent 論文の転記）。**そのまま提出してはいけない**（他者の解の盗用になる）。
- 差をつけるには「**進化ハーネス（Opus を発想エンジンに）× 未飽和問題**」のセットが必要。base model は差にならない。

## スキャン結果（全 17 問、新規追加なし）

| 飽和度 | 問題 | 競合 | rank1 | 判定 |
|---|---|---|---|---|
| TIE3 | erdos-min-overlap, heilbronn, thomson, flat-polynomials, difference-bases | 6-15 | — | 完全飽和、余地なし |
| tie2 | min-distance-ratio, uncertainty-principle, edges-vs-triangles | 11-12 | — | 飽和 |
| 非タイ(偽) | tammes (gap 1e-16), circle-packing, circles-rectangle (gap≈minImpr) | 14-16 | — | 実質飽和 |
| 開放 | kissing-d11 (rank1=0), kissing-d12 (floor 2.0) | 3-7 | 0 / 2 | 非飽和だが局所探索では不可（厳密構成要） |
| 混雑 | 3rd/2nd/1st autocorrelation | 19-20 | — | 攻撃が濃い、再現済 |
| **★本命** | **prime-number-theorem (id 7)** | **11** | **0.994901** | **既知ゴール1.0まで余地0.0051、未収束、LP攻め可** |

## 推奨ターゲット: prime-number-theorem (id 7)

- 目的: maximize S = −Σ f(k)·log(k)/k、制約 Σf(k)·⌊x/k⌋ ≤ 1 (∀x≥1)、Σf(k)/k = 0、|F|≤2000、f∈[−10,10]。
- 理論最大 S=1.0（Möbius μ）。現 arena top 0.994901 → **余地 0.0051**。
- **support を固定すると LP**（目的・制約とも f に線形）。攻め方 = (1) support 集合の選択、(2) その上で LP を解く、(3) より密な x グリッドで全制約を自前検証、(4) support を進化させる。
- **進化ハーネスの効きどころ**: LLM が「どの k を support に入れるか」「Möbius からの逸脱構造」を多様に提案 → LP で評価 → 良いものを残して進化。固定勾配では出せない多様性。

### ⚠ 整合性（必ず守る）

verifier は制約を **1e7 個の固定 MC 点（RandomState(42)）** でしか検査しない。そこへ過適合して全 x 制約を破りつつスコアだけ上げるのは**ゲーミングで本物の貢献ではない**。提出候補は**より密な独立グリッドで Σf(k)⌊x/k⌋≤1 を全域検証**してから扱う。合法な改善（Möbius 近傍の真に feasible な構成）のみを成果とする。

## 次セッションの段取り (step 2 以降)

1. PNT verifier を `arena.verifier.extract_verifier("prime-number-theorem")` で原典抽出（schema は dict `{"partial_function": {"<k>": f}}`、recon メモ参照）。
2. LP ベースライン: 切断 Möbius を seed に `scipy.optimize.linprog` で support 固定 LP → S を確認。
3. 進化ハーネス（FunSearch-lite）: agent/Workflow が support 構造を多様生成 → LP 評価 → 上位を mutate/cross → ループ。各候補は (a) サーバ verifier、(b) 自前の密グリッド全制約検証 の両方で採点。
4. オリジナルかつ合法な S > 0.994901 が出たら → **ユーザー承認を得てから**登録（エージェント名要決定）+ 提出。

## 既存の再利用資産

- `arena/verifier.py` — 問題 detail 取得 + verifier 原典のローカル抽出/ロード。
- `arena/client.py` — read-only GET + 承認ゲート submit（dry-run 既定）。
- `scripts/solve_*.py` — min-dist / erdos / 3rd-autocorr の jax 勾配ソルバ（パターン流用可）。
- `scripts/saturation_scan.py` — このスキャンの再実行。
