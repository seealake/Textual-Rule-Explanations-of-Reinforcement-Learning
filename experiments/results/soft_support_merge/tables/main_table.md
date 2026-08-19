# Main comparison: CBS vs default merge vs rule-set voting vs soft support

| env            | method       | config             |   f1_mean |   f1_std |   GRS_wj |   GRS_ta |   BRA |   worst_action_recall_mean |   n_rules_mean |
|:---------------|:-------------|:-------------------|----------:|---------:|---------:|---------:|------:|---------------------------:|---------------:|
| MountainCar-v0 | CBS          | baseline           |     0.782 |    0.025 |    0.823 |    0.845 | 0.985 |                      0.229 |         13.400 |
| MountainCar-v0 | B3_consensus | baseline           |     0.760 |    0.019 |    0.879 |    0.841 | 0.937 |                      0.286 |         11.000 |
| MountainCar-v0 | B3_vote      | baseline           |     0.792 |    0.005 |  nan     |  nan     | 0.988 |                      0.286 |         13.000 |
| MountainCar-v0 | soft_support | lB0.0_smhard_sgoff |     0.760 |    0.019 |    0.879 |    0.841 | 0.937 |                      0.286 |         11.000 |
