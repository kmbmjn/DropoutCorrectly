# DropoutCorrectly
This repository contains Python source code for ``How to Use Dropout Correctly on Residual Networks with Batch Normalization''

Works with Python 3.6+ and PyTorch 1.7+.

# predropout_postdropout.py
Used for Figure 1 for the empirical validation on proposition 2. Simply run ``python predropout_postdropout.py''

# residual_nonresidual.py
Used for Figure 2 for the empirical validations on propositions 3 and 4.

# preresnetdropout_cifar.py
Used for CIFAR experiments.

# train.py and my_model.py
Used for Pet and Caltech experiments. The code requires hyperparameter arguments, such as the following:
``python train.py --model_name my_resnetv2_50_p6_f --mode train --data_name pet --hp_opt sgd --hp_lr 1e-1 --hp_wd 5e-3 --hp_bs 128 --hp_ep 200 --hp_id 0''

# densenet_h4.py
Used for head experiments.
