# Final model evaluation

Both saved models were evaluated without fitting on the same 40,429 validation rows in original order. Test data was hashed only, not evaluated.
All six recalculated metrics match saved artifacts to absolute tolerance 1e-7; confusion matrices match exactly.

| Metric | TF-IDF | Sentence-BERT | Absolute improvement |
|---|---:|---:|---:|
| accuracy | 0.778426 | 0.813253 | +0.034826 |
| precision | 0.669799 | 0.701639 | +0.031839 |
| recall | 0.785273 | 0.857296 | +0.072024 |
| f1 | 0.722954 | 0.771696 | +0.048742 |
| macro_f1 | 0.769172 | 0.806853 | +0.037681 |
| roc_auc | 0.868986 | 0.899528 | +0.030541 |

Improvements are differences on the 0–1 scale; multiply by 100 for percentage points (ROC-AUC is normally reported in AUC units).

| Model | TN | FP | FN | TP |
|---|---:|---:|---:|---:|
| TF-IDF | 19783 | 5762 | 3196 | 11688 |
| Sentence-BERT | 20119 | 5426 | 2124 | 12760 |

## Paired outcomes

- both_correct: 27,988
- only_tfidf_correct: 3,483
- only_sbert_correct: 4,891
- both_incorrect: 4,067
- Exact two-sided McNemar test (binomial test on 8,374 discordant pairs): p=1.3629405e-53.
This is nominal evidence of different error rates on these pairs. Repeated/shared questions can violate pair independence, so this p-value does not establish significance for independent unseen-question populations.

## Error analysis

Sentence-BERT made 7,550 validation errors. The CSV selects the first 10 false positives and first 10 false negatives in original row order, plus explicitly reviewed examples. It is a purposive illustration, not a representative frequency estimate.
Word overlap is lowercase word-set Jaccard, defined in Phase 1. Lexical thresholds (high >=0.75, low <=0.20) describe overlap, not semantic truth. Semantic labels in the CSV are the dataset's ground truth and may be ambiguous.

- Validation row 4: **High lexical overlap; changed scope**. Baby names starting with D versus D or H changes the allowed initial letters.
- Validation row 59: **Low lexical overlap; similar meaning**. Both ask for advice on outlining a novel; word-set Jaccard is 0.1333.
- Validation row 208: **High lexical overlap; changed year**. Best NBA player in 2016 versus 2013 asks about different years.
- Validation row 409: **High lexical overlap; changed year**. Mobile technology trends in 2016 versus 2015 asks about different years.
- Validation row 596: **Negation / opposite desired action**. Turning off Chrome password-save prompts versus fixing Chrome not offering to save passwords asks for opposite behavior.
- Validation row 615: **Negation / incompatible premise**. How Telugu and Tamil are related versus why there is no relation has incompatible premises; this annotation does not assert which premise is true.
- Validation row 716: **Negation**. Regretting things done versus things not done changes the question's meaning.
- Validation row 997: **Low lexical overlap; similar meaning**. Both seek a suitable laptop under a 30,000 budget; the second emphasizes specifications. Jaccard is 0.1176.
- Validation row 1282: **Low lexical overlap; similar meaning**. Both request an explanation of the Unified Payment Interface, with additional institutional context in the second. Jaccard is 0.1765.
- Validation row 1580: **Subject-object reversal**. Vendors forcing customers to sign non-competes versus customers forcing vendors reverses the actor and recipient.
- Validation row 6194: **Subject-object reversal**. America's view of the world versus the world's view of America reverses observer and object.
- Validation row 19996: **Possible annotation/context issue**. The two visible texts are identical ('How would you rephrase this sentence?') but the dataset label is 0. Unseen referents could differ; the label is retained, not corrected.

## Six semantic challenges

Intended labels are manually specified, not training data. Predictions are unchanged saved-model outputs; no thresholds were tuned. See semantic_challenge_tests.csv for full texts, probabilities and Sentence-BERT cosine scores.

| Pair | TF-IDF label / P(paraphrase) / correct | SBERT label / P(paraphrase) / correct | SBERT cosine |
|---|---|---|---:|
| How can I learn Python? / What is the best way to study Python? | Paraphrase / 0.781614 / True | Paraphrase / 0.876354 / True | 0.831580 |
| What causes rain? / How do I bake a chocolate cake? | Not Paraphrase / 0.173091 / True | Not Paraphrase / 0.000868 / True | 0.131890 |
| The dog chased the cat. / The cat chased the dog. | Paraphrase / 0.871439 / False | Paraphrase / 0.979012 / False | 0.991334 |
| He bought a car. / He purchased an automobile. | Not Paraphrase / 0.028918 / False | Paraphrase / 0.717514 / True | 0.915947 |
| I like coffee. / I do not like coffee. | Not Paraphrase / 0.491031 / True | Paraphrase / 0.581283 / False | 0.802463 |
| The meeting was postponed. / The meeting was moved to a later date. | Not Paraphrase / 0.255130 / False | Not Paraphrase / 0.325757 / False | 0.703822 |

## Reliability and limits

- Validation contains 25,545 negatives and 14,884 positives. Report positive-class F1, macro F1 and ROC-AUC alongside accuracy.
- Both classifiers use balanced class weights and uncalibrated probabilities. A large probability is not a measured guarantee of correctness.
- The specified challenge tests confirm role-reversal errors; negation and other errors are reported per model in the table. These six examples do not measure general accuracy.
- MiniLM's model card lists Quora Question Triplets among pretraining sources. Actual overlap with these validation rows is unverified; exposure is possible, not proven. Source: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2 .
- Phase 1 found 29,851 shared exact individual questions between the original train and validation splits. Removing two overlapping pairs does not remove question-level dependence.
- Frozen independent sentence embeddings can lose details of negation and argument roles. This is a plausible explanation, not a causal finding from this evaluation. Long inputs are truncated at 256 word pieces.
- Error categories were assigned only after reading the selected original texts. Unknown categories remain generic false positives/negatives. Absence of a confirmed category in the reviewed subset is not proof it is absent from the dataset.

## Recommended next steps

Keep this evaluation fixed for reporting. For future work, use a separate tuning/calibration split, consider grouping by shared questions, review ambiguous annotations, and test a pairwise cross-encoder or fine-tuning only after approval. Do not tune on the six challenges or this validation report.

## Verification

All 81 original dataset/model/cache/evaluation files had matching before/after SHA-256 hashes. No models were fitted or downloaded. Cache identity and sample fingerprint matched Phase 3. Both models predicted the same original validation rows; selected errors are indexed directly from those predictions.
