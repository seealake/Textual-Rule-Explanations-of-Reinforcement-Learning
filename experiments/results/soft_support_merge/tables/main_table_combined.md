# Main comparison: CBS vs default merge vs rule-set voting vs soft support (Both Envs)

| env            | method       | config             |   f1_mean |   f1_std |   GRS_wj |   GRS_ta |   BRA |   worst_action_recall_mean |   n_rules_mean |
|:---------------|:-------------|:-------------------|----------:|---------:|---------:|---------:|------:|---------------------------:|---------------:|
| CartPole-v1    | CBS          | baseline           |     0.800 |    0.034 |    0.316 |    0.392 | 0.783 |                      0.680 |         12.600 |
| CartPole-v1    | B3_consensus | baseline           |     0.568 |    0.115 |    0.158 |    0.225 | 0.448 |                      0.227 |          1.400 |
| CartPole-v1    | B3_vote      | baseline           |     0.822 |    0.044 |  nan     |  nan     | 0.854 |                      0.718 |         12.800 |
| CartPole-v1    | soft_support | lB0.1_smsoft_sgoff |     0.804 |    0.053 |    0.226 |    0.438 | 0.808 |                      0.693 |         42.000 |
| LunarLander-v3 | CBS          | baseline           |     0.662 |    0.010 |    0.299 |    0.644 | 0.897 |                      0.269 |         30.600 |
| LunarLander-v3 | B3_consensus | baseline           |     0.518 |    0.014 |    0.341 |    0.622 | 0.735 |                      0.277 |         17.800 |
| LunarLander-v3 | B3_vote      | baseline           |     0.667 |    0.008 |  nan     |  nan     | 0.907 |                      0.289 |         30.200 |
| LunarLander-v3 | soft_support | lB0.2_smsoft_sgoff |     0.653 |    0.022 |    0.265 |    0.575 | 0.825 |                      0.345 |         52.400 |
