# Dev question set (our own, for tuning)

`dev_questions.json` holds 23 questions written by the team against `kb.json`: 17 in scope, each with one gold passage, and 6 out of scope (`gold_ids: []`, correct behaviour = abstain).

- **Why it exists:** the lab09 honor code forbids hand-labelling the 22 leaderboard questions in `questions.json`. We tune chunking, k, τ and the reranker on this dev set **only**. The leaderboard questions are the sealed test set and get used once, at the end.
- **The trap it targets:** the questions are paraphrased away from the passage wording ("thyroid hormone" vs "levothyroxine", "ramipril + valsartan" vs "ACE inhibitor + ARB", "puffer" vs "metered-dose inhaler"). Without that, BM25 looks unrealistically good.
- **In-domain abstention traps:** D19 (HbA1c *frequency*: the KB has targets, not frequency), D21 (insulin initiation) and D22 (AF ablation). These share vocabulary with real passages but are not answerable from them.
- **Clinical lead: review every gold id before the final run.**
