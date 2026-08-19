# Ablation: Consensus CBS v2 Configuration Sweep

| env            | config             |   f1_mean |   f1_std |   GRS_wj |   GRS_ta |   BRA |   worst_action_recall_mean |   n_rules_mean |   group_size_mean |
|:---------------|:-------------------|----------:|---------:|---------:|---------:|------:|---------------------------:|---------------:|------------------:|
| MountainCar-v0 | lB0.0_smhard_sgoff |     0.760 |    0.019 |    0.879 |    0.841 | 0.937 |                      0.286 |         11.000 |             4.818 |
| MountainCar-v0 | lB0.0_smhard_sgon  |     0.760 |    0.019 |    0.879 |    0.841 | 0.937 |                      0.286 |         11.000 |             4.818 |
| MountainCar-v0 | lB0.0_smsoft_sgoff |     0.725 |    0.005 |    0.904 |    0.859 | 0.983 |                      0.000 |         18.200 |             3.615 |
| MountainCar-v0 | lB0.0_smsoft_sgon  |     0.725 |    0.005 |    0.903 |    0.836 | 0.983 |                      0.000 |         18.600 |             3.645 |
| MountainCar-v0 | lB0.1_smhard_sgoff |     0.760 |    0.019 |    0.879 |    0.841 | 0.937 |                      0.286 |         11.000 |             4.818 |
| MountainCar-v0 | lB0.1_smhard_sgon  |     0.760 |    0.019 |    0.879 |    0.841 | 0.937 |                      0.286 |         11.000 |             4.818 |
| MountainCar-v0 | lB0.1_smsoft_sgoff |     0.725 |    0.005 |    0.904 |    0.859 | 0.983 |                      0.000 |         18.200 |             3.615 |
| MountainCar-v0 | lB0.1_smsoft_sgon  |     0.725 |    0.005 |    0.903 |    0.836 | 0.983 |                      0.000 |         18.600 |             3.645 |
| MountainCar-v0 | lB0.2_smhard_sgoff |     0.760 |    0.019 |    0.879 |    0.841 | 0.937 |                      0.286 |         11.000 |             4.818 |
| MountainCar-v0 | lB0.2_smhard_sgon  |     0.760 |    0.019 |    0.879 |    0.841 | 0.937 |                      0.286 |         11.000 |             4.818 |
| MountainCar-v0 | lB0.2_smsoft_sgoff |     0.728 |    0.006 |    0.900 |    0.834 | 0.983 |                      0.000 |         17.400 |             3.736 |
| MountainCar-v0 | lB0.2_smsoft_sgon  |     0.728 |    0.006 |    0.900 |    0.810 | 0.983 |                      0.000 |         17.800 |             3.730 |
