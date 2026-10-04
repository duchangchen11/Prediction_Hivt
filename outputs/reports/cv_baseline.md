# Constant velocity diagnostic

30 windows; Th=5, Tf=12; actual timestamps; distances in metres.

Mini scene split: 6 train / 2 val / 2 test, seed 42. This is a custom smoke-test split.

Pooled values include all three partitions and are diagnostic, not benchmark performance. Actor-windows are averaged equally; overlapping windows are not independent.

ADE uses valid future points. FDE uses the last valid future point. Fixed-horizon FDE separately requires the final requested frame. Histories with fewer than two observations are excluded.

## Pooled diagnostic

| Type | ADE (m) | FDE last valid (m) | FDE final frame (m) | Evaluated / current actors |
|---|---:|---:|---:|---:|
| vehicle | 1.156408 | 2.570848 | 2.793824 | 674 / 718 |
| pedestrian | 0.511357 | 1.053188 | 1.304765 | 395 / 411 |
| bicycle | 0.424552 | 0.852467 | 0.978731 | 18 / 18 |
| overall | 0.909887 | 1.990897 | 2.217906 | 1087 / 1147 |

## train

| Type | ADE (m) | FDE last valid (m) | FDE final frame (m) | Evaluated / current actors |
|---|---:|---:|---:|---:|
| vehicle | 1.419382 | 3.117457 | 3.739275 | 402 / 429 |
| pedestrian | 0.548579 | 1.119694 | 1.308817 | 270 / 281 |
| bicycle | 0.317035 | 0.670577 | 0.531515 | 13 / 13 |
| overall | 1.055225 | 2.283580 | 2.628437 | 685 / 723 |

## val

| Type | ADE (m) | FDE last valid (m) | FDE final frame (m) | Evaluated / current actors |
|---|---:|---:|---:|---:|
| vehicle | 1.232228 | 2.882167 | 2.433136 | 103 / 113 |
| pedestrian | 0.492186 | 1.025102 | 1.351243 | 86 / 90 |
| bicycle | 0.528064 | 1.005485 | 1.005485 | 1 / 1 |
| overall | 0.893555 | 2.031723 | 1.942681 | 190 / 204 |

## test

| Type | ADE (m) | FDE last valid (m) | FDE final frame (m) | Evaluated / current actors |
|---|---:|---:|---:|---:|
| vehicle | 0.484664 | 1.080890 | 1.152576 | 169 / 176 |
| pedestrian | 0.295939 | 0.654698 | 1.125281 | 39 / 40 |
| bicycle | 0.748107 | 1.405353 | 4.529702 | 4 / 4 |
| overall | 0.454916 | 1.008609 | 1.172806 | 212 / 220 |
