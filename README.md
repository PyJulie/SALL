# SALL
Official implementation for IEEE JBHI paper 'Synergic Adversarial Label Learning for Grading Retinal Diseases via Knowledge Distillation and Multi-task Learning'

Please cite:

>@article{ju2021synergic,<br>
  title={Synergic Adversarial Label Learning for Grading Retinal Diseases via Knowledge Distillation and Multi-task Learning},<br>
  author={Ju, Lie and Wang, Xin and Zhao, Xin and Lu, Huimin and Mahapatra, Dwarikanath and Bonnington, Paul and Ge, Zongyuan},<br>
  journal={IEEE Journal of Biomedical and Health Informatics},<br>
  year={2021},<br>
  publisher={IEEE}<br>
}

This work uses a private datasets. You can find some useful dataset [here](https://github.com/nkicsl/Fundus_Review).

Also, you can try a cifar-10 (5/5) dataset as a toy experiment. Our methods can also achieve improvments on those classes with similar features.

|             | Task A (1-5) | Task B (6-10) |
|-------------|--------------|---------------|
| Single Task | 91.80        | 95.84         |
| Ours        | 92.70        | 96.60         |

## To Do

Pytorch implementation.
