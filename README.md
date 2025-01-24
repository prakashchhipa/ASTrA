# ASTrA: Adversarial Self-Supervised Training with Adaptive Attacks

_Chhipa, P.C.*_, _Vashishtha, G.*_, _Settur, J.*_, _Saini, R._, _Shah, M._, _Liwicki, M._ _ASTrA: Adversarial Self-Supervised Training with Adaptive Attacks. International Conference on Learning Representations. (ICLR 2025)_

\*Indicates equal contributions.
  

Visit Project Website - https://prakashchhipa.github.io/projects/ASTrA

ASTrA addresses the limitations of conventional self-supervised adversarial training (Self-AT) methods, which rely on static heuristic attacks that fail to adapt to diverse data distributions and model dynamics. By introducing a learnable, reinforcement learning-based adversarial attack strategy, ASTrA generates adaptive perturbations that optimize self-supervised representations. This approach improves robustness and generalization across multiple datasets, enabling more effective and sustainable training for adversarially robust models. ASTrA also incorporates mixed contrastive loss to further enhance representation quality, making it a state-of-the-art solution for robust self-supervised learning.

![Main Framework](images/MainFigure.png)


Install the requisite conda environment using ASTra_environment.yml
## Training 
To run the code:
Run it using the bash file
bash run.sh 
run.sh file has all the hyperparameters tuned accordingly

## Evaluation

### SLF & ALF

    python test_LF.py --experiment EXP_PATH --checkpoint PATH_TO_PRETRAINED_MODEL --dataset DATASET_NAME --data PATH_TO_DATASET --cvt_state_dict --bnNameCnt 1 --evaluation_mode EVALUATION_MODE

### AFF

    python test_AFF.py --experiment EXP_PATH --checkpoint PATH_TO_PRETRAINED_MODEL --dataset DATASET_NAME --data PATH_TO_DATASET


## Acknowledgements

Some of our code are borrowed from [DynACL](https://github.com/PKU-ML/DYNACL/). Thanks for their great work!
