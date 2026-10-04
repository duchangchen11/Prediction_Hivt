# Stage3A official multi-type data audit

700 train / 150 val scenes; overlap0; test unused. Natural class distribution; no oversampling or class-balanced loss.

train: vehicle/pedestrian/bicycle target actor-windows=[315257, 109934, 5390].

val: vehicle/pedestrian/bicycle target actor-windows=[62981, 20740, 1306].

Scene shards=850; failed0; NaN0; Inf0. All anchors retained, zero-supervision anchors indexed separately. Exact old vehicle input/target pairing=378238.

All generated windows use the existing instance-chain and ego-coordinate checks. Ten random examples per class also compare stored positions/headings/map coordinates directly against source annotations and map discretizations.
