# business-area
## Introduction
This repository contains the project component for course CSCI-GA 3033-091 (Introduction to Deep Learning Systems) in New York University. The authors are **Qingyang Li** and **Jiahao Chen**.

This project aims at discovering the boundaries of business areas in cities and predicting its future variations using trajectory data of taxicabs. More detailed information can be found in the Project Report.

## Overall Structure
The `codes` directory contains source code of this project. The whole procedure is compressed into a Jupyter Notebook named `pipeline.ipynb`. The steps include: data preprocessing; ConvLSTM model construction, training and testing; comparison with other methods; visualization of results.

In addition, `convolution_lstm.py` includes the definition of the network structure of ConvLSTM, which is the deep learning model we applied to this task. `evaluate.py` includes python scripts to compute the metrics (Precision, Recall and F1-score) to evaluate the performance of various models.

The `orders` directory contains trajectory data (plain text) of the original form. The `data_paired` directory contains trajectory data (plain text) after pre-processing. Please refer to the Project Report for more details.

The `count_map` directory contains transition cuboids of 20 days, in `npy` format (numpy matrices). These data serve as the input data of our model. The `heat_map` directory contains heat maps of the city, also in `npy` format. These data serve as the label of our model.

The `model` directory contains saved model parameters. You may use the `load_state_dict` method provided by PyTorch to load a pretrained model.

## How to run it
You may just follow the steps clearly stated in `pipeline.ipynb` to run our model. Feel free to replace the `orders` directory by trajectory data of other cities.