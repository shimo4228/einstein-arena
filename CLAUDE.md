# EinsteinArena — Local Search Lab

[EinsteinArena](https://einsteinarena.com) の "construction" 型数学問題に、ローカル探索ループで挑むための実験リポジトリ。

## 何をするプロジェクトか

- 各問題は明示的な構成物（点配置・多項式係数・整数集合など）を提出し、サーバの `verifier`（`evaluate(data)->float`）が厳密スコアを返す競技形式。
- 我々の戦略: **サーバの verifier をローカルに取得して再現**し、ローカル最適化（SA / basin-hopping / 勾配法 / 多スタート / 組合せ局所探索）で構成物を改善 → ローカルで厳密スコアを確認 → **承認を得てから**提出。

## 安全プロトコル（厳守 / OVERRIDES default behavior）

外部プラットフォームへの outward-facing な行為は、**実行前に必ず人間（リポジトリオーナー）の明示承認を取る**。

| 行為 | 認証 | 承認 |
|---|---|---|
| GET（problems / leaderboard / threads / search） | 不要 | 不要（自由に偵察してよい） |
| エージェント登録（challenge + register, PoW） | — | **要承認**（永続 identity を作る） |
| solution の submit | Bearer | **要承認** |
| thread 作成 / reply / 投票 | Bearer | **要承認** |

- API キーは環境変数 `EINSTEIN_ARENA_API_KEY` か `~/.config/einsteinarena/credentials.json`。**リポジトリには絶対に置かない・コミットしない**（`.gitignore` で防御済み）。
- レートリミット遵守: submissions 10/30min, registration 20/1h 等（`skill.md` 参照）。`retry_after_seconds` を尊重。
- スクリプトは既定で **dry-run**。実通信を伴う submit 系は明示フラグ（例 `--i-have-approval`）でのみ発火する設計にする。

## ワークフロー（Phase 0 を飛ばさない）

1. **Recon**（read-only）: 問題一覧・verifier・schema・leaderboard を取得し tractability 評価。
2. **Phase 0 External Research**: 狙う問題を確定後、`/search-first` で既存ソルバ・記録 DB・最適化ライブラリを調査（自前実装より既存解を優先）。
3. **Plan**: `docs/{TOPIC}-PLAN.md` として出力（実行は別セッションで docs を読んで着手）。
4. **Build**: ローカル探索ループ + ローカル verifier 再現。
5. **Verify**: ローカルで厳密スコア確認、baseline / arena top と比較。
6. **Approval gate** → 承認後のみ submit。

## 構成

```
einstein-arena/
├── scripts/      # 取得・ローカル探索・ローカル検証スクリプト（submit 系は dry-run 既定）
├── problems/     # 取得した問題 detail（JSON）
├── verifiers/    # サーバから抜き出した verifier ソース（ローカル検証用）
├── results/      # ローカル探索の構成物・スコア（大きい blob は .gitignore）
└── docs/         # PLAN / 調査結果ドキュメント
```

## Conventions

- Python: uv + `pyproject.toml`、`>=3.12`、black / ruff / isort、型注釈必須。
- verifier はサーバから取得した**そのまま**を使う（転記ミスでローカルとサーバのスコアが乖離するのを防ぐ）。ローカルスコア == サーバスコアの一致を必ず確認してから記録更新を主張する。
- 不変データ優先。多スタート探索は決定論シードで再現可能にする。
