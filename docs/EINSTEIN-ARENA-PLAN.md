# EinsteinArena 攻略プラン（段階的）

> 新セッションはこの doc を読んで着手する。安全プロトコルは [../CLAUDE.md](../CLAUDE.md) が正本。

## 目的（ユーザー確定）

**段階的**: まず安全な1問で「取得→ローカル探索→ローカル verifier 一致→（承認後）提出」ループを検証し、型が固まってから *新記録の余地がある問題* へ pivot する。prototype-before-scale。

## 横断的事実（recon + Phase 0 で判明）

- このアリーナは **arXiv:2511.02864（AlphaEvolve "Mathematical exploration and discovery at scale"）のベンチマーク再現競争**。プラットフォーム自体の論文は arXiv:2606.10402。
- **解きやすい問題ほどリーダーボードが既知最適で飽和**（複数エージェントがビット一致）。tractable = 余地なし。
- 公開 SOTA repo が存在する: `togethercomputer/EinsteinArena-new-SOTA`, `vinid/einstein-arena`（プラットフォーム本体＝verifier 原典）。

## Phase 0 External Research — Verdict

| 対象 | Verdict | 採用物 |
|---|---|---|
| Arena API クライアント | **Build** | `requests` 薄ラッパ（read-only + 承認ゲート submit）。SDK は存在しない |
| 最適化コア | **Adopt** | `scipy.optimize` + `cma`（導入済）。`jax[cpu]` は ≥100 次元の滑らか目的が出たら追加 |
| 幾何系の既知最適 | **Adopt/転記** | min-dist: FM Agent arXiv:2510.26144 §5.3.3 / Tammes n=50: neilsloane pack.3.50.txt |
| 解析系 headroom | **Extend/Compose** | `zkli-math/autoconvolutionHolder`(coeffBL.txt), `test-time-training/discover`, `togethercomputer/erdos-minimum-overlap` |

## Phase A — ループ検証（対象: min-distance-ratio-2d, id=5）

選定理由: verifier が最小・-inf 崖なし・相似不変。soft-min/max + L-BFGS multistart が数秒で既知最適 12.8892299 に到達 → **「ローカル==サーバ」を高精度で実証**できる。

成功基準:
1. 保存した verifier（サーバ原典）でローカル採点が動く。
2. 自前 multistart が **R ≤ 12.88923**（arena top 12.889229907717521 と同等以下）に到達。
3. 不正入力（形状違い・点重複）が verifier で正しく弾かれる。
4. submit がコードレベルで承認ゲートされ、未承認では送信されない（dry-run）。
5. 上記すべて **提出なし** で完了。

Phase A はタイ狙い（新記録ではない）。ループ機構と verifier 忠実性の検証が目的。

## Phase B — pivot 候補（新記録の余地、ローカル探索で現実的）

| 問題 | 方向 | Arena top | 余地・参照資産 |
|---|---|---|---|
| **3rd autocorrelation** | min | 1.4523043 | *人間の先行研究ほぼ無い AI 定義領域*。ゼロから勾配降下で改善が注目に値する。最有力 |
| **Erdős min overlap** | min | 0.3808703 | bracket [0.379005, 0.380871]。600 区間非対称ステップ + FFT 勾配 + equioscillation。参照: ttt-discover, togethercomputer/erdos-minimum-overlap |
| **2nd autocorrelation** | max | 0.9626433 | Boyer–Li coeffBL.txt(559/2399 区間)から upsampling で 0.9626→1.0 へ。手順文献化済 |
| **1st autocorrelation** | min | 1.5028506 | minImprovement 1e-7（僅差でも登録）。30000 区間 + FFT 勾配 |
| **PNT** | max | 0.9949010 | 切断 Möbius + 有限グリッド LP。⚠ 半無限制約を 1e7 MC(RandomState(42))でサンプル→seed 過適合の罠 |

pivot 時は対象問題を1つ確定 → focused deep-dive + 参照コード精読 → 自前探索（公開 config を seed にする場合は disclose）。

## 安全ゲート（再掲）

- GET は自由。**登録・submit・投稿は実行前に人間の明示承認**。
- API キーは `EINSTEIN_ARENA_API_KEY`、repo にコミットしない。
- submit 系スクリプトは dry-run 既定、`approved=True` でのみ実送信。

## ファイル構成

```
arena/client.py        # read-only GET + 承認ゲート submit（dry-run 既定）
arena/verifier.py      # 問題 detail 取得 → verifier 原典抽出 → ローカル load
verifiers/<slug>.py    # サーバ verifier のバイト単位コピー
scripts/solve_*.py     # 問題別ソルバ（exact verifier で採点）
problems/<slug>.json   # 取得した問題 detail
results/<slug>/        # 探索結果（構成物 + スコア）
```
