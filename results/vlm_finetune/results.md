# Vision-language models: zero-shot and fine-tuned

Accuracy with Wilson 95 % interval; chance is 20 %. False alarm = normal clips called a fault.

## test

| model | input | condition | n | accuracy (95 % CI) | detect recall | false alarm |
|---|---|---|---|---|---|---|
| Qwen3-VL-4B-Instruct | sheet | clean | 500 | 51.2% (47%–56%) | 39% | 0% |
| Qwen3-VL-4B-Instruct | single | clean | 500 | 57.2% (53%–61%) | 100% | 100% |
| Qwen3-VL-4B-Instruct | sheet | degraded | 500 | 33.6% (30%–38%) | 17% | 0% |
| Qwen3-VL-4B-Instruct | single | degraded | 500 | 35.6% (32%–40%) | 100% | 100% |
| Qwen3-VL-4B-Instruct+adapter_sheet | sheet | clean | 250 | 97.2% (94%–99%) | 100% | 13% |
| Qwen3-VL-4B-Instruct+adapter_single | single | clean | 500 | 94.8% (92%–96%) | 94% | 2% |

## heldout

| model | input | condition | n | accuracy (95 % CI) | detect recall | false alarm |
|---|---|---|---|---|---|---|
| Qwen3-VL-4B-Instruct | sheet | clean | 200 | 43.0% (36%–50%) | 29% | 0% |
| Qwen3-VL-4B-Instruct | single | clean | 200 | 52.5% (46%–59%) | 100% | 100% |
| Qwen3-VL-4B-Instruct | sheet | degraded | 200 | 29.0% (23%–36%) | 16% | 2% |
| Qwen3-VL-4B-Instruct | single | degraded | 200 | 33.5% (27%–40%) | 89% | 72% |

## Per-class recall (test)

| model | input | condition | normal | stuck_item | gate_open | spray | blocked_lens |
|---|---|---|---|---|---|---|---|
| Qwen3-VL-4B-Instruct | sheet | clean | 100% | 0% | 0% | 97% | 59% |
| Qwen3-VL-4B-Instruct | single | clean | 0% | 0% | 100% | 100% | 86% |
| Qwen3-VL-4B-Instruct | sheet | degraded | 100% | 0% | 0% | 58% | 10% |
| Qwen3-VL-4B-Instruct | single | degraded | 0% | 0% | 100% | 67% | 11% |
| Qwen3-VL-4B-Instruct+adapter_sheet | sheet | clean | 87% | 100% | 100% | 100% | 100% |
| Qwen3-VL-4B-Instruct+adapter_single | single | clean | 98% | 78% | 100% | 100% | 98% |

## Sheet adapter on sheets against single-frame adapter on single frames (paired, exact McNemar)

| condition | split | n | only sheet right | only single right | p |
|---|---|---|---|---|---|
| clean | test | 250 | 8 | 7 | 1 |

