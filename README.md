# einstein-arena

[EinsteinArena](https://einsteinarena.com) の construction 型数学問題に、ローカル探索ループで挑む実験リポジトリ。

サーバの verifier をローカルで再現し、局所探索で構成物を改善 → ローカルで厳密スコアを確認 → **人間の承認を得てから**提出する、という慎重運用を前提にしている。

- 詳細な運用ルールと安全プロトコル: [CLAUDE.md](CLAUDE.md)
- 計画・調査ドキュメント: [docs/](docs/)

## セットアップ

```bash
uv venv
uv sync
```

## 安全上の注意

- API キーはリポジトリに置かない（環境変数 `EINSTEIN_ARENA_API_KEY`）。
- 登録 / submit / 投稿はすべて事前承認制。スクリプトの submit 系は dry-run 既定。
