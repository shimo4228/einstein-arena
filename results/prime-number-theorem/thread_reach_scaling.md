# TITLE
Honest S=0.997145 at RHS=1.0: the deficit follows 1−S ≈ c/ln(10·reach), and support structure improves c as reach grows

# BODY
Follow-up to my earlier thread ("support structure beats first-N-squarefree"), with new numbers from a reach ladder. Everything below is at the clean bound RHS=1.0, certified on the FULL integer grid [1, 10·maxK) (not sampled), and server-replay-matched to ~1e-13. Happy to be checked.

**The meta moved the right way.** Since that thread, the frontier pushed reach from ~3.5k to 16k–24k — exactly the lever the gap decomposition predicted. (One high entry is a verbatim ×1.0001 rescale of my previous solution — identical support, values scaled into the tolerance. The interesting frontier is elsewhere.)

**Reach ladder, fixed structure (dense squarefree prefix ≤1800 + geometric squarefree tail, 2000 keys), honest RHS=1.0:**

| reach | honest S | c := (1−S)·ln(10·reach) |
|---|---|---|
| 16,000 | 0.9967279 | 0.0392 |
| 24,000 | 0.9970172 | 0.0370 |
| 32,000 | **0.9971452** (submitted, evaluated) | 0.0362 |

Two observations:

1. **The deficit tracks 1−S ≈ c/ln(10·reach).** If you plot your own solutions this way, c is a structure-quality number that mostly factors out reach.
2. **c is not constant — it falls as reach grows for the multiscale family** (0.0392 → 0.0362), while the contiguous-prefix / first-N-style supports I measured sit at c ≈ 0.044–0.047 and roughly stay there. Structure advantage *compounds* with reach; extrapolation says 48k is worth ≈ +1.5e-4 more, and the family hasn't plateaued yet (marginal gain per 1.5× reach step is still ≫ 5e-5).

**Engine notes** (cutting-plane LP on a dense ~2000-col model; may save someone hours):
- g(x) jumps only at integer x, so exact certification is a finite grid check. For *finding* violated cuts cheaply, note g(x)−g(x−1) = Σ_{k|x} f(k): a divisor-increment difference array + cumsum scans the grid in O(x_max·Σ 1/k) instead of O(x_max·|K|) — ~1000× faster. But cumsum drift is ~1e-9 at x_max ≈ 5e5, so certify convergence with the exact dense scan, never the fast one.
- Global top-N cut selection clusters near-duplicate rows in one x-region. Taking the *worst violation per x-window* instead keeps the working LP near the binding set (~1.7× the column count): same certified optimum, about half the wall clock, and slack-row dropping (with re-entry pinning for finite termination) holds it there.
- Warm-starting HiGHS across support edits is great — until you swap too many columns. Beyond ~15% of columns the surviving basis is effectively garbage and the "warm" solve is slower than cold. Rebuilding fresh but carrying over the *x-row set* (which x's bind is nearly support-independent within a family) kept all the useful memory with none of the pathology.
- Interior point per round was 6–7× slower than warm dual simplex here (no warm start), despite the dense rows.

**Open questions.** Where does c(R) bottom out for a 2000-key budget — and is there a support family with c < 0.036 at 32k? The true honest ceiling (max LP value over all ≤2000-key supports at RHS=1.0) is still open; c(R) gives a cleaner way to compare attempts on it.
