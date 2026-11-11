# Error analysis: findings (test split)

Manual reading of the seeded samples in [error_analysis_test.md](error_analysis_test.md) (seed 0): 30 questions where R3 beats R2, 30 where R3 loses to R2, and all 11 test questions (0.5%) that every configuration misses in its top-20. Category counts are judgements by one annotator; a question can have more than one cause.

## 1. Segment breakdown: hypotheses checked with numbers

| Hypothesis (stated before the test run) | Evidence on test | Verdict |
|---|---|---|
| Dense is hurt when the answer article is truncated at 2,048 tokens (EDA, protocol threats) | On the 178 questions whose answer is truncated, R2's Hit@10 drops to **0.871 = R1's**, versus 0.890 vs 0.852 overall. R3 recovers to 0.921. | **Confirmed.** Truncation erases dense's recall advantage; fusion compensates. |
| BM25 is relatively stronger on questions with numbers / document codes | On the 193 questions with digits, BM25 nDCG@10 is above its average (R0 0.677 vs 0.645), dense is not (0.725 vs 0.727), and R3 gains most over R2 (Hit@10 0.938 vs 0.891). | **Confirmed.** |
| Short questions favour dense (few terms for BM25 to match) | ≤ 12 syllables: R1 Hit@10 0.759 vs R2 0.853 (+9.4). > 20 syllables: 0.902 vs 0.916 (+1.4). | **Confirmed.** The dense advantage shrinks as questions get longer. |

**R4 (reranker) has the same weakness, but worse.** The cross-encoder reads only 512 tokens shared between question and article. On the 178 long-answer questions, its nDCG@10 falls to **0.613, versus 0.681 for R3**, while it beats R3 in every other segment (e.g. short questions: 0.759 vs 0.686). The reranker sees only the start of long articles, so it confidently promotes shorter, more focused articles.

Truncation matters more than the EDA suggested: 2% of *labelled articles* exceed 2,048 tokens, but 8% of *test questions* (178/2,237) point to one. Long articles, such as the traffic-fine articles of Decree 100/2019, are asked about far more often than short ones.

## 2. Why R3 beats R2 (30 sampled questions)

| Cause | Count | Example |
|---|---|---|
| Exact rare term, phrase or number that BM25 matches and dense blurs | 12 | "dán tem *VIET NAM DUTY NOT PAID*", "xe ô tô không có *đèn báo hãm* theo nghị định 100", "xếp hàng chờ lớn hơn *150 xe đến 200 xe*" |
| Dense confuses a sibling article (same law, similar title); BM25's distinctive words pick the right one | 9 | Motorbike fines: R2 ranks Article 8 (*bicycles*) above Article 6 (*motorbikes*); bưu gửi Article 9 vs 11; securities licence Article 74 vs 259 |
| Neither system ranks the answer first, but both rank it moderately; fusion's consensus lifts it | 9 | "Kiểm soát viên chính thị trường là ai?" R1 = 5, R2 = 4 → R3 = 2 |

## 3. Why R3 loses to R2 (30 sampled questions)

| Cause | Count | Example |
|---|---|---|
| Paraphrased or colloquial question: dense has the answer at rank 1–2, BM25's top ranks are lexical distractors, and with `k = 10` a distractor at BM25 rank 1 outranks it | 20 | "Làm đơn tố cáo mà *không ghi tên mình*?" (article says "*không rõ họ tên*"): R2 = 1, R1 = 19, R3 = 2 |
| Long question with generic filler words that BM25 latches onto | 5 | "…được quy định ra sao trong *văn bản mới nhất do Ngân hàng nhà nước vừa ban hành*?" R1 misses the top-100, R2 = 1, R3 = 9 |
| Questionable or ambiguous label | 4 | "Giáo viên có được bảo lưu phụ cấp ưu đãi…" labelled *Article 1, scope*, while *Article 2, "Mức phụ cấp ưu đãi được bảo lưu"* answers it more directly |
| Other (multi-relevant question; both find the first answer at rank 1, R2 ranks the second answer higher) | 1 | |

Most losses are small (the answer drops from rank 1 to 2–3), while many wins are large (rank 8+ → 1–2). This asymmetry is why the averages favour R3 even though R3 loses 328 head-to-head nDCG comparisons on test.

## 4. Questions every configuration misses (all 11)

| Cause | Count | Example |
|---|---|---|
| Answer is a long glossary article (*"Giải thích từ ngữ"*) that defines dozens of terms; one definition is diluted in a single vector and in BM25's length normalisation | 4 | "Khái niệm quỹ đại chúng", "Theo quy định cũ chứng từ kế toán được hiểu như thế nào?" |
| Answer requires reasoning or world knowledge not in the text | 2 | "Sinh năm 2000 có còn được dự tuyển trường quân đội?" (age arithmetic); "Gọi điện thoại ở cây xăng phạt bao nhiêu?" (phone = fire-risk source) |
| A near-identical article from another document ranks first (likely an unlabelled correct answer) | 2 | "lựa chọn nhà thầu … nạo vét luồng hàng hải": several retrievers return an *"Điều 17. Tổ chức lựa chọn nhà thầu"* from a different circular; "Nguồn kinh phí … vùng khó khăn": R1 returns *"Điều 2. Nguồn kinh phí thực hiện Chương trình"* |
| Indirect label (the labelled article only implies the answer) | 2 | "Ban quản lý dự án có được tư vấn giám sát thi công?" labelled with the board's general duties |
| Answer sits in one clause of a very long fines article (Decree 100/2019, Article 15) | 1 | "Mức phạt khi làm sai lệch biển báo hiệu, đèn tín hiệu giao thông" |

## Takeaways

1. **Fusion works because the two systems fail differently.** BM25 handles exact terms, numbers and sibling disambiguation. Dense handles paraphrase and short questions.
2. **The main cost of `k = 10` is visible:** a strong lexical distractor at BM25 rank 1 can push dense's correct top-1 down one or two places. This is the precision loss that R4's reranker targets.
3. **Long articles are the hardest case for every model-based method.** R2 (2,048 tokens) loses its edge there, and R4 (512 tokens) falls below R3. Chunking articles and scoring the best chunk (rejected in the EDA for simplicity) is the most promising next step for both.
4. **Part of the remaining error is label noise** (near-duplicate articles across documents, indirect labels), so absolute scores understate quality. This matches the threat noted in the protocol.
