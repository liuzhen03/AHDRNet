# AHDRNet
This repository is a PyTorch implementation of CVPR 2019 "Attention-guided Network for Ghost-free High Dynamic Range Imaging"  ![AHDRNet](https://github.com/qingsenyangit/AHDRNet)

## Results
Todo

## Prerequisites
This codebase was developed and tested with pytorch 1.1.1 and Python 3.6.9


* [runx](https://github.com/Pea-Shooter/runx) for experiments management (you should install `runx` via source code)
* [TensorboardX](https://github.com/lanpa/tensorboardX) for training visualization

## Training
### Data
* ![HDR_Dynamic_Scenes_SIGGRAPH2017](https://cseweb.ucsd.edu/~viscomp/projects/SIG17HDR/) which includes 74 samples for training and 15 samples for testing.



### Training
To train from terminal, run:
```shell script
python -m runx.runx experiments/test.yml
```

Or just run the shell script:
```shell script
bash run.sh
```

Run the following commmand for help / more options.
```shell script
python train -h
```

To get visualization of the training, you can run tensorboard from the `LOGROOT` directory using the command:
```shell script
tensorboard --logdir /data/log/nnhdr --port 6006
```

## Evaluation
Todo
