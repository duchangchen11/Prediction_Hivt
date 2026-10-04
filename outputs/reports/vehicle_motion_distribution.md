# Vehicle motion distribution

Source: nuScenes mini, existing training scenes only. Counts are actor-windows, not unique vehicles.

All current vehicles: 3623; loss-eligible: 3417; full 12-step future: 2158.

Actual attribute names: cycle.with_rider, cycle.without_rider, pedestrian.moving, pedestrian.sitting_lying_down, pedestrian.standing, vehicle.moving, vehicle.parked, vehicle.stopped

## Real t0 attribute

| State | Count | Fraction |
|---|---:|---:|
| moving | 1166 | 32.18% |
| parked | 1434 | 39.58% |
| stopped | 903 | 24.92% |
| unknown | 120 | 3.31% |

## All actor-windows

Valid future denominator=3529; no future=94.

| Endpoint bin | Count | Fraction of valid future |
|---|---:|---:|
| 0–1m | 2338 | 66.25% |
| 1–2m | 48 | 1.36% |
| 2–5m | 55 | 1.56% |
| 5–10m | 87 | 2.47% |
| 10–20m | 196 | 5.55% |
| 20–40m | 357 | 10.12% |
| >40m | 448 | 12.69% |

## Loss-eligible targets

Valid future denominator=3417; no future=0.

| Endpoint bin | Count | Fraction of valid future |
|---|---:|---:|
| 0–1m | 2269 | 66.40% |
| 1–2m | 47 | 1.38% |
| 2–5m | 54 | 1.58% |
| 5–10m | 87 | 2.55% |
| 10–20m | 196 | 5.74% |
| 20–40m | 350 | 10.24% |
| >40m | 414 | 12.12% |

## Full horizon only

Valid future denominator=2158; no future=0.

| Endpoint bin | Count | Fraction of valid future |
|---|---:|---:|
| 0–1m | 1518 | 70.34% |
| 1–2m | 34 | 1.58% |
| 2–5m | 19 | 0.88% |
| 5–10m | 24 | 1.11% |
| 10–20m | 77 | 3.57% |
| 20–40m | 169 | 7.83% |
| >40m | 317 | 14.69% |

## Definitions and limits

- state: t0 real annotation attribute; absent/other is unknown; low speed never implies parked
- endpoint: distance from t0 to last valid future, approximately 6s only when full 12 steps
- GT_ADE_displacement: mean norm of valid future minus t0; descriptive motion quantity, not prediction ADE
- path_length: sum segments from t0 through available future points; gaps bridged and valid length retained
- history_speed: observed history path length / elapsed time; current speed uses last two past observations
- bins: [0,1], (1,2], (2,5], (5,10], (10,20], (20,40], (40,inf) meters

Real stopped and parked labels remain separate. Full-horizon bins provide the approximate 6s comparison; partial-horizon endpoints must not be described as 6s displacement.

Detailed descriptive speeds, path lengths and valid lengths are in the JSON and vehicle_actor_windows.csv.
