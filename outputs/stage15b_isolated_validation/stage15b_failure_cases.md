# Stage15B failure cases

Cases are selected after all checkpoints and evaluation outputs freeze. They illustrate extreme observed harms and do not estimate their prevalence. Exact GT/candidate CSVs are local-only and intentionally absent from GitHub; the linked small case metadata remains versioned. All degradation and tie counts appear in `09_statistics/stage15b_switch_harm.csv`. Shared-candidate oracle metrics remain unchanged.

## case1: G-C-NG-C / MovingVehicle

Fold 3; scene `8c7dfcee70754286959b80c6cfe2d246`, sample `ad6d4f310f6143459f3e35303d45c20a`, instance `e44d9b6cccb64b7ba0a866e5fd565921`. Delta Top1FDE=25.088585 m; shared oracle minFDE6=4.702390 m.

Full selected modes, errors and predictor SHA: [case metadata](08_evaluation/stage15b_cases/case1.json). Exact GT and all six candidate coordinates/times: local CSV `08_evaluation/stage15b_cases/case1_trajectories.csv`. These cases show ranking errors under common geometry; they do not establish a causal mechanism.

## case2: G-C-NG-C / Pedestrian

Fold 3; scene `bed8426a524d45afab05b19cf02386b2`, sample `12a087e40d874fc19f7282c153c15789`, instance `dabc3da6de3045779670d0ab98293554`. Delta Top1FDE=5.705440 m; shared oracle minFDE6=0.679795 m.

Full selected modes, errors and predictor SHA: [case metadata](08_evaluation/stage15b_cases/case2.json). Exact GT and all six candidate coordinates/times: local CSV `08_evaluation/stage15b_cases/case2_trajectories.csv`. These cases show ranking errors under common geometry; they do not establish a causal mechanism.

## case3: G-C-Matched-NG-C / Overall

Fold 2; scene `1a819a84e3494177b47bea8c7f0770ec`, sample `cbea9b2886a3432c8788b1533c125381`, instance `349d672353f242e2b337b6f79ad495dd`. Delta Top1FDE=19.906831 m; shared oracle minFDE6=1.063567 m.

Full selected modes, errors and predictor SHA: [case metadata](08_evaluation/stage15b_cases/case3.json). Exact GT and all six candidate coordinates/times: local CSV `08_evaluation/stage15b_cases/case3_trajectories.csv`. These cases show ranking errors under common geometry; they do not establish a causal mechanism.

## case4: NG-C-NG-A / Vehicle

Fold 3; scene `30ae9c1092f6404a9e6aa0589e809780`, sample `63673387306e4c89a6a7b82cd6004c59`, instance `a6e18ad0ef09473caef85cee9ea782ad`. Delta Top1FDE=31.017097 m; shared oracle minFDE6=6.395051 m.

Full selected modes, errors and predictor SHA: [case metadata](08_evaluation/stage15b_cases/case4.json). Exact GT and all six candidate coordinates/times: local CSV `08_evaluation/stage15b_cases/case4_trajectories.csv`. These cases show ranking errors under common geometry; they do not establish a causal mechanism.

