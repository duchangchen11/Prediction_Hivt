# Trainval readiness

TRAJECTORY_METADATA_READY=YES
HD_MAP_READY=YES
LIDAR_REQUIRED_FOR_STAGE2=NO

Readiness=PASS for the trajectory/map task. No raw LiDAR file was opened or downloaded.

Source: /media/lrj/54926A1D926A0438/nuscenes-trainval; metadata cache: /home/lrj/Prediction_Hivt/outputs/stage2/trainval/cache.

This 16GB host uses streaming JSON to extract five complete scenes (seed 42), then loads them through the official devkit. All source rows in the large tables were scanned; only selected records are materialized. This is a sampled task audit, not a full sensor-data or full trainval integrity claim.

| Scene | Samples | Location | Annotation links checked |
|---|---:|---|---:|
| scene-0316 | 39 | singapore-queenstown | 756 |
| scene-0428 | 41 | singapore-queenstown | 1378 |
| scene-0653 | 41 | boston-seaport | 3888 |
| scene-0666 | 41 | boston-seaport | 3986 |
| scene-0927 | 40 | singapore-queenstown | 1532 |

All four HD maps loaded and sample lane centerlines discretized successfully.
