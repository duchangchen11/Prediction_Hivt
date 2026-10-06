# Run from /home/lrj/Prediction_Hivt with the existing ped_intent environment.
# Register exactly once; later commands reuse the stored scene split and cache.
export PYTHONDONTWRITEBYTECODE=1
/home/lrj/anaconda3/envs/ped_intent/bin/python outputs/stage6a_future_interaction_reliability/00_manifest/stage6a_register.py
/home/lrj/anaconda3/envs/ped_intent/bin/python outputs/stage6a_future_interaction_reliability/00_manifest/stage6a_tests.py
/home/lrj/anaconda3/envs/ped_intent/bin/python outputs/stage6a_future_interaction_reliability/01_cache/stage6a_cache.py
/home/lrj/anaconda3/envs/ped_intent/bin/python outputs/stage6a_future_interaction_reliability/01_cache/stage6a_cache.py --audit
/home/lrj/anaconda3/envs/ped_intent/bin/python outputs/stage6a_future_interaction_reliability/02_features/stage6a_build_features.py
/home/lrj/anaconda3/envs/ped_intent/bin/python outputs/stage6a_future_interaction_reliability/03_training/stage6a_train.py
/home/lrj/anaconda3/envs/ped_intent/bin/python outputs/stage6a_future_interaction_reliability/04_evaluation/stage6a_evaluate.py
/home/lrj/anaconda3/envs/ped_intent/bin/python outputs/stage6a_future_interaction_reliability/04_evaluation/stage6a_headdev_importance.py
/home/lrj/anaconda3/envs/ped_intent/bin/python outputs/stage6a_future_interaction_reliability/04_evaluation/stage6a_efficiency.py
/home/lrj/anaconda3/envs/ped_intent/bin/python outputs/stage6a_future_interaction_reliability/05_figures/stage6a_figures.py
/home/lrj/anaconda3/envs/ped_intent/bin/python outputs/stage6a_future_interaction_reliability/05_figures/stage6a_cases.py
/home/lrj/anaconda3/envs/ped_intent/bin/python outputs/stage6a_future_interaction_reliability/05_figures/stage6a_export_QA.py
/home/lrj/anaconda3/envs/ped_intent/bin/python outputs/stage6a_future_interaction_reliability/09_reports/stage6a_finalize.py
