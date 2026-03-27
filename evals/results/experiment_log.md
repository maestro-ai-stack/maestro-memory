# mmem Autoresearch Experiment Log

Baseline: **50% QA Accuracy** on LongMemEval_S (N=50, subagent-judged)
Target: **65%+** (beat Zep 63.8%)
Session Recall baseline: **83.3%** (N=500)

## Experiment 0: Baseline
- Date: 2026-03-27
- QA Accuracy: 50% (25/50)
- Session Recall: 83.3% (N=500)
- Latency: 6ms avg
- By type: temporal-reasoning 69%, preference 80%, multi-session 46%, knowledge-update 43%, single-session-user 29%, single-session-assistant 20%
- Key finding: 84% of failures are retrieval failures (context missing)
