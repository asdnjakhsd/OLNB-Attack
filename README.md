# Optimal Low-Frequency Noise Black-box Attack for Visual Object Tracking

We provide the complete attack code against SiamRPN++, which runs on the `pysot` framework. So for use you can basically refer to pysot's guide, you may have to install `scipy` additionally if needed.

In order to run the code, you need to do the following things:

```txt
1. Install the environment needed for pysot, e.g. using conda.
2. Download the required models from Model Zoo.
3. Download the json file used to record the labels of the dataset.
4. Unzip and place the dataset in /pysot/experiments
```

These steps are exactly the same as the required sessions for deploying pysot, it is recommended to refer to the pysot guide for more specific instructions and to ensure that scipy is installed, which is the only additional thing that needs to be done.

## run in VOT2018

Test Tracker:

```sh
cd experiments/siamrpn_r50_l234_dwxcorr
python -u ../../tools/test_OLNBA.py   \
       --snapshot model.pth     \   # model path
       --dataset VOT2018        \   # dataset name
       --config config.yaml         # config file
```

Eval Tracker:

```sh
python ../../tools/eval.py     \
      --tracker_path ./results \ # result path
      --dataset VOT2018        \ # dataset name
      --num 1                  \ # number thread to eval
      --tracker_prefix 'model'   # tracker_name
```

Ensure that a tracking test is completed for a successful evaluation

## License

Licensed under an MIT license.
