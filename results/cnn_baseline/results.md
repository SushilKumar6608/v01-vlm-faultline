# CNN baseline (from scratch, trained on clean clips, CPU)

Wilson 95 % intervals. All data are synthetic. Sheet versus single frame: exact McNemar test on paired clips; p-values are descriptive and not corrected for multiple comparisons.

## Summary

| model | input | condition | split | n | accuracy (95% CI) | fault detected (95% CI) | false alarm (95% CI) | macro_recall | invalid_rate |
|---|---|---|---|---|---|---|---|---|---|
| cnn_s0 | sheet | clean | heldout | 200 | 20% [15, 26] | 100% [98, 100] | 100% [91, 100] | 20% | 0% |
| cnn_s0 | sheet | clean | test | 500 | 100% [99, 100] | 100% [99, 100] | 0% [0, 4] | 100% | 0% |
| cnn_s0 | sheet | clean | val | 250 | 100% [98, 100] | 100% [97, 100] | 0% [0, 7] | 100% | 0% |
| cnn_s0 | sheet | degraded | heldout | 200 | 20% [15, 26] | 100% [98, 100] | 100% [91, 100] | 20% | 0% |
| cnn_s0 | sheet | degraded | test | 500 | 31% [27, 35] | 100% [99, 100] | 100% [96, 100] | 31% | 0% |
| cnn_s0 | sheet | degraded | val | 250 | 30% [25, 36] | 100% [98, 100] | 100% [93, 100] | 30% | 0% |
| cnn_s0 | single | clean | heldout | 200 | 20% [15, 27] | 100% [98, 100] | 100% [91, 100] | 20% | 0% |
| cnn_s0 | single | clean | test | 500 | 97% [95, 98] | 98% [97, 99] | 8% [4, 15] | 97% | 0% |
| cnn_s0 | single | clean | val | 250 | 94% [91, 97] | 94% [90, 97] | 4% [1, 13] | 94% | 0% |
| cnn_s0 | single | degraded | heldout | 200 | 20% [15, 26] | 100% [98, 100] | 100% [91, 100] | 20% | 0% |
| cnn_s0 | single | degraded | test | 500 | 60% [55, 64] | 100% [99, 100] | 96% [90, 98] | 60% | 0% |
| cnn_s0 | single | degraded | val | 250 | 55% [49, 61] | 100% [98, 100] | 100% [93, 100] | 55% | 0% |
| cnn_s1 | sheet | clean | heldout | 200 | 30% [24, 36] | 100% [98, 100] | 100% [91, 100] | 30% | 0% |
| cnn_s1 | sheet | clean | test | 500 | 100% [99, 100] | 100% [99, 100] | 0% [0, 4] | 100% | 0% |
| cnn_s1 | sheet | clean | val | 250 | 100% [98, 100] | 100% [97, 100] | 0% [0, 7] | 100% | 0% |
| cnn_s1 | sheet | degraded | heldout | 200 | 20% [15, 27] | 100% [98, 100] | 100% [91, 100] | 20% | 0% |
| cnn_s1 | sheet | degraded | test | 500 | 81% [77, 84] | 100% [99, 100] | 73% [64, 81] | 81% | 0% |
| cnn_s1 | sheet | degraded | val | 250 | 79% [73, 83] | 100% [98, 100] | 70% [56, 81] | 79% | 0% |
| cnn_s1 | single | clean | heldout | 200 | 20% [15, 27] | 100% [98, 100] | 100% [91, 100] | 20% | 0% |
| cnn_s1 | single | clean | test | 500 | 97% [95, 98] | 99% [97, 100] | 11% [6, 19] | 97% | 0% |
| cnn_s1 | single | clean | val | 250 | 96% [92, 98] | 97% [94, 99] | 10% [4, 21] | 96% | 0% |
| cnn_s1 | single | degraded | heldout | 200 | 20% [15, 26] | 100% [98, 100] | 100% [91, 100] | 20% | 0% |
| cnn_s1 | single | degraded | test | 500 | 76% [72, 79] | 100% [99, 100] | 86% [78, 91] | 76% | 0% |
| cnn_s1 | single | degraded | val | 250 | 73% [67, 78] | 100% [97, 100] | 82% [69, 90] | 73% | 0% |
| cnn_s2 | sheet | clean | heldout | 200 | 20% [15, 26] | 100% [98, 100] | 100% [91, 100] | 20% | 0% |
| cnn_s2 | sheet | clean | test | 500 | 100% [99, 100] | 100% [99, 100] | 0% [0, 4] | 100% | 0% |
| cnn_s2 | sheet | clean | val | 250 | 100% [98, 100] | 100% [97, 100] | 0% [0, 7] | 100% | 0% |
| cnn_s2 | sheet | degraded | heldout | 200 | 20% [15, 26] | 100% [98, 100] | 100% [91, 100] | 20% | 0% |
| cnn_s2 | sheet | degraded | test | 500 | 74% [70, 78] | 100% [99, 100] | 81% [72, 87] | 74% | 0% |
| cnn_s2 | sheet | degraded | val | 250 | 70% [64, 76] | 100% [98, 100] | 80% [67, 89] | 70% | 0% |
| cnn_s2 | single | clean | heldout | 200 | 20% [15, 26] | 100% [98, 100] | 100% [91, 100] | 20% | 0% |
| cnn_s2 | single | clean | test | 500 | 96% [94, 97] | 97% [95, 98] | 7% [3, 14] | 96% | 0% |
| cnn_s2 | single | clean | val | 250 | 95% [91, 97] | 94% [89, 96] | 0% [0, 7] | 95% | 0% |
| cnn_s2 | single | degraded | heldout | 200 | 20% [15, 26] | 100% [98, 100] | 100% [91, 100] | 20% | 0% |
| cnn_s2 | single | degraded | test | 500 | 36% [32, 40] | 100% [99, 100] | 99% [95, 100] | 36% | 0% |
| cnn_s2 | single | degraded | val | 250 | 34% [28, 40] | 100% [98, 100] | 100% [93, 100] | 34% | 0% |


## Recall per class

| model | input | condition | split | normal | stuck_item | gate_open | spray | blocked_lens |
|---|---|---|---|---|---|---|---|---|
| cnn_s0 | sheet | clean | heldout | 0% | 0% | 0% | 0% | 100% |
| cnn_s0 | sheet | clean | test | 100% | 100% | 100% | 100% | 100% |
| cnn_s0 | sheet | clean | val | 100% | 98% | 100% | 100% | 100% |
| cnn_s0 | sheet | degraded | heldout | 0% | 0% | 0% | 0% | 100% |
| cnn_s0 | sheet | degraded | test | 0% | 7% | 18% | 31% | 100% |
| cnn_s0 | sheet | degraded | val | 0% | 6% | 20% | 24% | 100% |
| cnn_s0 | single | clean | heldout | 0% | 0% | 2% | 0% | 100% |
| cnn_s0 | single | clean | test | 92% | 95% | 100% | 100% | 99% |
| cnn_s0 | single | clean | val | 96% | 76% | 100% | 100% | 100% |
| cnn_s0 | single | degraded | heldout | 0% | 0% | 0% | 0% | 100% |
| cnn_s0 | single | degraded | test | 4% | 38% | 94% | 62% | 100% |
| cnn_s0 | single | degraded | val | 0% | 26% | 92% | 56% | 100% |
| cnn_s1 | sheet | clean | heldout | 0% | 12% | 35% | 0% | 100% |
| cnn_s1 | sheet | clean | test | 100% | 99% | 100% | 100% | 100% |
| cnn_s1 | sheet | clean | val | 100% | 98% | 100% | 100% | 100% |
| cnn_s1 | sheet | degraded | heldout | 0% | 0% | 2% | 0% | 100% |
| cnn_s1 | sheet | degraded | test | 27% | 88% | 99% | 91% | 100% |
| cnn_s1 | sheet | degraded | val | 30% | 74% | 100% | 90% | 100% |
| cnn_s1 | single | clean | heldout | 0% | 2% | 0% | 0% | 100% |
| cnn_s1 | single | clean | test | 89% | 98% | 100% | 100% | 97% |
| cnn_s1 | single | clean | val | 90% | 88% | 100% | 100% | 100% |
| cnn_s1 | single | degraded | heldout | 0% | 0% | 0% | 0% | 100% |
| cnn_s1 | single | degraded | test | 14% | 95% | 96% | 73% | 100% |
| cnn_s1 | single | degraded | val | 18% | 90% | 98% | 58% | 100% |
| cnn_s2 | sheet | clean | heldout | 0% | 0% | 0% | 0% | 100% |
| cnn_s2 | sheet | clean | test | 100% | 100% | 100% | 100% | 100% |
| cnn_s2 | sheet | clean | val | 100% | 98% | 100% | 100% | 100% |
| cnn_s2 | sheet | degraded | heldout | 0% | 0% | 0% | 0% | 100% |
| cnn_s2 | sheet | degraded | test | 19% | 65% | 96% | 91% | 100% |
| cnn_s2 | sheet | degraded | val | 20% | 58% | 92% | 82% | 100% |
| cnn_s2 | single | clean | heldout | 0% | 0% | 0% | 0% | 100% |
| cnn_s2 | single | clean | test | 93% | 88% | 100% | 100% | 99% |
| cnn_s2 | single | clean | val | 100% | 74% | 100% | 100% | 100% |
| cnn_s2 | single | degraded | heldout | 0% | 0% | 0% | 0% | 100% |
| cnn_s2 | single | degraded | test | 1% | 16% | 30% | 32% | 100% |
| cnn_s2 | single | degraded | val | 0% | 8% | 30% | 32% | 100% |


## Sheet versus single frame (paired)

| model | condition | split | scope | n | acc_single | acc_sheet | delta | only_single_right | only_sheet_right | p_mcnemar |
|---|---|---|---|---|---|---|---|---|---|---|
| cnn_s0 | clean | heldout | all | 200 | 20% | 20% | -0% | 1 | 0 | 1.00 |
| cnn_s0 | clean | heldout | normal | 40 | 0% | 0% | 0% | 0 | 0 | 1.00 |
| cnn_s0 | clean | heldout | stuck_item | 40 | 0% | 0% | 0% | 0 | 0 | 1.00 |
| cnn_s0 | clean | heldout | gate_open | 40 | 2% | 0% | -2% | 1 | 0 | 1.00 |
| cnn_s0 | clean | heldout | spray | 40 | 0% | 0% | 0% | 0 | 0 | 1.00 |
| cnn_s0 | clean | heldout | blocked_lens | 40 | 100% | 100% | 0% | 0 | 0 | 1.00 |
| cnn_s0 | clean | test | all | 500 | 97% | 100% | 3% | 0 | 14 | 0.00 |
| cnn_s0 | clean | test | normal | 100 | 92% | 100% | 8% | 0 | 8 | 0.01 |
| cnn_s0 | clean | test | stuck_item | 100 | 95% | 100% | 5% | 0 | 5 | 0.06 |
| cnn_s0 | clean | test | gate_open | 100 | 100% | 100% | 0% | 0 | 0 | 1.00 |
| cnn_s0 | clean | test | spray | 100 | 100% | 100% | 0% | 0 | 0 | 1.00 |
| cnn_s0 | clean | test | blocked_lens | 100 | 99% | 100% | 1% | 0 | 1 | 1.00 |
| cnn_s0 | clean | val | all | 250 | 94% | 100% | 5% | 0 | 13 | 0.00 |
| cnn_s0 | clean | val | normal | 50 | 96% | 100% | 4% | 0 | 2 | 0.50 |
| cnn_s0 | clean | val | stuck_item | 50 | 76% | 98% | 22% | 0 | 11 | 0.00 |
| cnn_s0 | clean | val | gate_open | 50 | 100% | 100% | 0% | 0 | 0 | 1.00 |
| cnn_s0 | clean | val | spray | 50 | 100% | 100% | 0% | 0 | 0 | 1.00 |
| cnn_s0 | clean | val | blocked_lens | 50 | 100% | 100% | 0% | 0 | 0 | 1.00 |
| cnn_s0 | degraded | heldout | all | 200 | 20% | 20% | 0% | 0 | 0 | 1.00 |
| cnn_s0 | degraded | heldout | normal | 40 | 0% | 0% | 0% | 0 | 0 | 1.00 |
| cnn_s0 | degraded | heldout | stuck_item | 40 | 0% | 0% | 0% | 0 | 0 | 1.00 |
| cnn_s0 | degraded | heldout | gate_open | 40 | 0% | 0% | 0% | 0 | 0 | 1.00 |
| cnn_s0 | degraded | heldout | spray | 40 | 0% | 0% | 0% | 0 | 0 | 1.00 |
| cnn_s0 | degraded | heldout | blocked_lens | 40 | 100% | 100% | 0% | 0 | 0 | 1.00 |
| cnn_s0 | degraded | test | all | 500 | 60% | 31% | -28% | 152 | 10 | 0.00 |
| cnn_s0 | degraded | test | normal | 100 | 4% | 0% | -4% | 4 | 0 | 0.12 |
| cnn_s0 | degraded | test | stuck_item | 100 | 38% | 7% | -31% | 33 | 2 | 0.00 |
| cnn_s0 | degraded | test | gate_open | 100 | 94% | 18% | -76% | 76 | 0 | 0.00 |
| cnn_s0 | degraded | test | spray | 100 | 62% | 31% | -31% | 39 | 8 | 0.00 |
| cnn_s0 | degraded | test | blocked_lens | 100 | 100% | 100% | 0% | 0 | 0 | 1.00 |
| cnn_s0 | degraded | val | all | 250 | 55% | 30% | -25% | 66 | 4 | 0.00 |
| cnn_s0 | degraded | val | normal | 50 | 0% | 0% | 0% | 0 | 0 | 1.00 |
| cnn_s0 | degraded | val | stuck_item | 50 | 26% | 6% | -20% | 12 | 2 | 0.01 |
| cnn_s0 | degraded | val | gate_open | 50 | 92% | 20% | -72% | 36 | 0 | 0.00 |
| cnn_s0 | degraded | val | spray | 50 | 56% | 24% | -32% | 18 | 2 | 0.00 |
| cnn_s0 | degraded | val | blocked_lens | 50 | 100% | 100% | 0% | 0 | 0 | 1.00 |
| cnn_s1 | clean | heldout | all | 200 | 20% | 30% | 9% | 0 | 18 | 0.00 |
| cnn_s1 | clean | heldout | normal | 40 | 0% | 0% | 0% | 0 | 0 | 1.00 |
| cnn_s1 | clean | heldout | stuck_item | 40 | 2% | 12% | 10% | 0 | 4 | 0.12 |
| cnn_s1 | clean | heldout | gate_open | 40 | 0% | 35% | 35% | 0 | 14 | 0.00 |
| cnn_s1 | clean | heldout | spray | 40 | 0% | 0% | 0% | 0 | 0 | 1.00 |
| cnn_s1 | clean | heldout | blocked_lens | 40 | 100% | 100% | 0% | 0 | 0 | 1.00 |
| cnn_s1 | clean | test | all | 500 | 97% | 100% | 3% | 0 | 15 | 0.00 |
| cnn_s1 | clean | test | normal | 100 | 89% | 100% | 11% | 0 | 11 | 0.00 |
| cnn_s1 | clean | test | stuck_item | 100 | 98% | 99% | 1% | 0 | 1 | 1.00 |
| cnn_s1 | clean | test | gate_open | 100 | 100% | 100% | 0% | 0 | 0 | 1.00 |
| cnn_s1 | clean | test | spray | 100 | 100% | 100% | 0% | 0 | 0 | 1.00 |
| cnn_s1 | clean | test | blocked_lens | 100 | 97% | 100% | 3% | 0 | 3 | 0.25 |
| cnn_s1 | clean | val | all | 250 | 96% | 100% | 4% | 0 | 10 | 0.00 |
| cnn_s1 | clean | val | normal | 50 | 90% | 100% | 10% | 0 | 5 | 0.06 |
| cnn_s1 | clean | val | stuck_item | 50 | 88% | 98% | 10% | 0 | 5 | 0.06 |
| cnn_s1 | clean | val | gate_open | 50 | 100% | 100% | 0% | 0 | 0 | 1.00 |
| cnn_s1 | clean | val | spray | 50 | 100% | 100% | 0% | 0 | 0 | 1.00 |
| cnn_s1 | clean | val | blocked_lens | 50 | 100% | 100% | 0% | 0 | 0 | 1.00 |
| cnn_s1 | degraded | heldout | all | 200 | 20% | 20% | 0% | 0 | 1 | 1.00 |
| cnn_s1 | degraded | heldout | normal | 40 | 0% | 0% | 0% | 0 | 0 | 1.00 |
| cnn_s1 | degraded | heldout | stuck_item | 40 | 0% | 0% | 0% | 0 | 0 | 1.00 |
| cnn_s1 | degraded | heldout | gate_open | 40 | 0% | 2% | 2% | 0 | 1 | 1.00 |
| cnn_s1 | degraded | heldout | spray | 40 | 0% | 0% | 0% | 0 | 0 | 1.00 |
| cnn_s1 | degraded | heldout | blocked_lens | 40 | 100% | 100% | 0% | 0 | 0 | 1.00 |
| cnn_s1 | degraded | test | all | 500 | 76% | 81% | 5% | 14 | 41 | 0.00 |
| cnn_s1 | degraded | test | normal | 100 | 14% | 27% | 13% | 3 | 16 | 0.00 |
| cnn_s1 | degraded | test | stuck_item | 100 | 95% | 88% | -7% | 8 | 1 | 0.04 |
| cnn_s1 | degraded | test | gate_open | 100 | 96% | 99% | 3% | 1 | 4 | 0.38 |
| cnn_s1 | degraded | test | spray | 100 | 73% | 91% | 18% | 2 | 20 | 0.00 |
| cnn_s1 | degraded | test | blocked_lens | 100 | 100% | 100% | 0% | 0 | 0 | 1.00 |
| cnn_s1 | degraded | val | all | 250 | 73% | 79% | 6% | 13 | 28 | 0.03 |
| cnn_s1 | degraded | val | normal | 50 | 18% | 30% | 12% | 1 | 7 | 0.07 |
| cnn_s1 | degraded | val | stuck_item | 50 | 90% | 74% | -16% | 11 | 3 | 0.06 |
| cnn_s1 | degraded | val | gate_open | 50 | 98% | 100% | 2% | 0 | 1 | 1.00 |
| cnn_s1 | degraded | val | spray | 50 | 58% | 90% | 32% | 1 | 17 | 0.00 |
| cnn_s1 | degraded | val | blocked_lens | 50 | 100% | 100% | 0% | 0 | 0 | 1.00 |
| cnn_s2 | clean | heldout | all | 200 | 20% | 20% | 0% | 0 | 0 | 1.00 |
| cnn_s2 | clean | heldout | normal | 40 | 0% | 0% | 0% | 0 | 0 | 1.00 |
| cnn_s2 | clean | heldout | stuck_item | 40 | 0% | 0% | 0% | 0 | 0 | 1.00 |
| cnn_s2 | clean | heldout | gate_open | 40 | 0% | 0% | 0% | 0 | 0 | 1.00 |
| cnn_s2 | clean | heldout | spray | 40 | 0% | 0% | 0% | 0 | 0 | 1.00 |
| cnn_s2 | clean | heldout | blocked_lens | 40 | 100% | 100% | 0% | 0 | 0 | 1.00 |
| cnn_s2 | clean | test | all | 500 | 96% | 100% | 4% | 0 | 20 | 0.00 |
| cnn_s2 | clean | test | normal | 100 | 93% | 100% | 7% | 0 | 7 | 0.02 |
| cnn_s2 | clean | test | stuck_item | 100 | 88% | 100% | 12% | 0 | 12 | 0.00 |
| cnn_s2 | clean | test | gate_open | 100 | 100% | 100% | 0% | 0 | 0 | 1.00 |
| cnn_s2 | clean | test | spray | 100 | 100% | 100% | 0% | 0 | 0 | 1.00 |
| cnn_s2 | clean | test | blocked_lens | 100 | 99% | 100% | 1% | 0 | 1 | 1.00 |
| cnn_s2 | clean | val | all | 250 | 95% | 100% | 5% | 0 | 12 | 0.00 |
| cnn_s2 | clean | val | normal | 50 | 100% | 100% | 0% | 0 | 0 | 1.00 |
| cnn_s2 | clean | val | stuck_item | 50 | 74% | 98% | 24% | 0 | 12 | 0.00 |
| cnn_s2 | clean | val | gate_open | 50 | 100% | 100% | 0% | 0 | 0 | 1.00 |
| cnn_s2 | clean | val | spray | 50 | 100% | 100% | 0% | 0 | 0 | 1.00 |
| cnn_s2 | clean | val | blocked_lens | 50 | 100% | 100% | 0% | 0 | 0 | 1.00 |
| cnn_s2 | degraded | heldout | all | 200 | 20% | 20% | 0% | 0 | 0 | 1.00 |
| cnn_s2 | degraded | heldout | normal | 40 | 0% | 0% | 0% | 0 | 0 | 1.00 |
| cnn_s2 | degraded | heldout | stuck_item | 40 | 0% | 0% | 0% | 0 | 0 | 1.00 |
| cnn_s2 | degraded | heldout | gate_open | 40 | 0% | 0% | 0% | 0 | 0 | 1.00 |
| cnn_s2 | degraded | heldout | spray | 40 | 0% | 0% | 0% | 0 | 0 | 1.00 |
| cnn_s2 | degraded | heldout | blocked_lens | 40 | 100% | 100% | 0% | 0 | 0 | 1.00 |
| cnn_s2 | degraded | test | all | 500 | 36% | 74% | 38% | 5 | 197 | 0.00 |
| cnn_s2 | degraded | test | normal | 100 | 1% | 19% | 18% | 0 | 18 | 0.00 |
| cnn_s2 | degraded | test | stuck_item | 100 | 16% | 65% | 49% | 2 | 51 | 0.00 |
| cnn_s2 | degraded | test | gate_open | 100 | 30% | 96% | 66% | 2 | 68 | 0.00 |
| cnn_s2 | degraded | test | spray | 100 | 32% | 91% | 59% | 1 | 60 | 0.00 |
| cnn_s2 | degraded | test | blocked_lens | 100 | 100% | 100% | 0% | 0 | 0 | 1.00 |
| cnn_s2 | degraded | val | all | 250 | 34% | 70% | 36% | 0 | 91 | 0.00 |
| cnn_s2 | degraded | val | normal | 50 | 0% | 20% | 20% | 0 | 10 | 0.00 |
| cnn_s2 | degraded | val | stuck_item | 50 | 8% | 58% | 50% | 0 | 25 | 0.00 |
| cnn_s2 | degraded | val | gate_open | 50 | 30% | 92% | 62% | 0 | 31 | 0.00 |
| cnn_s2 | degraded | val | spray | 50 | 32% | 82% | 50% | 0 | 25 | 0.00 |
| cnn_s2 | degraded | val | blocked_lens | 50 | 100% | 100% | 0% | 0 | 0 | 1.00 |


See the CSV files in this folder for severity bins, scenes and confusion matrices. Held-out D1 has 40 clips per class (about 13 per severity bin), so its intervals are wide.
