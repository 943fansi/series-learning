# Time Series 
[SOLETE Dataset](https://data.dtu.dk/articles/dataset/TheSOLETEdataset/17040767)

[SWaT WADI EPIC Dataset](https://itrust.sutd.edu.sg/itrust-labs_datasets/)

[Forecasting Methods and Practice Books Online](https://otexts.com/fppcn/)

[torchscale_demo](https://github.com/kashif/pytorch-transformer-ts/blob/main/torchscale/module.py)

[Time series distances](https://github.com/wannesm/dtaidistance)

[asymmetric loss](https://github.com/datatrigger/asymmetric_loss)
**<font color="blue">MLP-based</font>**
|  DLinear | STID | NHits | Nbeatsx | Nbeats |
|----------|----------|----------|----------|----------|
| [pdf](https://arxiv.org/abs/2205.13504)  | [pdf](https://arxiv.org/abs/2208.05233) | [pdf](https://arxiv.org/abs/2201.12886) |[pdf](https://arxiv.org/abs/2104.05522) |[pdf](https://arxiv.org/abs/1905.10437) |
| [code](https://github.com/cure-lab/LTSF-Linear) | [code](https://github.com/zezhishao/stid) | [code](https://github.com/Nixtla/neuralforecast) |[code](https://github.com/cchallu/nbeatsx) |[code](https://github.com/philipperemy/n-beats) |
#
**<font color="blue">RNN-based</font>**
| LSTNet | DeepAR | SRU |
|----------|----------|----------|
| [pdf](https://arxiv.org/abs/1703.07015) | [pdf](https://arxiv.org/abs/1704.04110) | [pdf](https://arxiv.org/abs/1709.02755)|
| [code](https://github.com/laiguokun/LSTNet) | [code](https://github.com/jdb78/pytorch-forecasting) | [code](https://github.com/asappresearch/sru) |
# 
**<font color="blue">CNN-based</font>**
| SCINet | OmniScale | TCN | mWDN |  
|----------|----------|----------|----------|
| [pdf](https://arxiv.org/abs/2106.09305) | [pdf](https://arxiv.org/abs/1803.01271) |[pdf](https://arxiv.org/abs/1803.01271) |[pdf](https://dl.acm.org/doi/abs/10.1145/3219819.3220060) |
| [code](https://github.com/cure-lab/SCINet) | [code](https://github.com/timeseriesAI/tsai) | [code](https://github.com/ForestsKing/TCN) | [code](https://github.com/timeseriesAI/tsai) |
#
**<font color="blue">Transformer-based</font>**
| PatchTST | ETSformer | FEDformer | Stationary | Scaleformer | Pyraformer | Preformer |
|----------|----------|----------|----------|----------|----------|----------|
| [pdf](https://arxiv.org/abs/2211.14730)   | [pdf](https://arxiv.org/abs/2202.01381)   | [pdf](https://arxiv.org/abs/2201.12740) | [pdf](https://arxiv.org/abs/2205.14415)   | [pdf](https://arxiv.org/abs/2206.04038)  | [pdf](https://openreview.net/pdf?id=0EXmFzUn5I) | [pdf](https://arxiv.org/abs/2202.11356) |
| [code](https://github.com/yuqinie98/PatchTST)| [code](https://github.com/salesforce/ETSformer)   | [code](https://github.com/MAZiqing/FEDformer)   | [code](https://github.com/thuml/Nonstationary_Transformers)   | [code](https://github.com/Scaleformer/Scaleformer) |  [code](https://github.com/alipay/Pyraformer) |[code](https://github.com/ddz16/Preformer) |

| TFT | TCCT | Autoformer | Informer | LogTrans | Reformer |
|----------|----------|----------|----------|----------|----------|
| [pdf](https://arxiv.org/abs/1912.09363)   | [pdf](https://arxiv.org/abs/2108.12784)   | [pdf](https://arxiv.org/abs/2106.13008)   | [pdf](https://arxiv.org/abs/2012.07436) | [pdf](https://arxiv.org/abs/1907.00235)   | [pdf](https://arxiv.org/abs/2001.04451) | 
| [code](https://github.com/unit8co/darts)   | [code](https://github.com/OrigamiSL/TCCT2021-Neurocomputing-)   | [code](https://github.com/thuml/autoformer)   | [code](https://github.com/zhouhaoyi/Informer2020)   | [code](https://github.com/AIStream-Peelout/flow-forecast)   | [code](https://github.com/google/trax/tree/master/trax/models/reformer) | 
#

### wind speed forecast
[Multistep short-term wind speed prediction using nonlinear auto-regressive neural network with exogenous variable selection](https://www.sciencedirect.com/science/article/pii/S1110016820305627?via%3Dihub)

[code](https://github.com/MNFuad/Multivariate-Multistep-Wind-Speed-Prediction)

**<font color="green">Loss function</font>**
* [TILDE-Q: A Transformation Invariant Loss Function for Time-Series Forecasting]()
* [A Comprehensive Survey of Regression Based Loss Functions for Time Series Forecasting]()
* [Deep Time Series Forecasting with Shape and Temporal Criteria]()
* [Shape and Time Distortion Loss for Training Deep Time Series Forecasting Models]()

**<font color="pink">Tricks</font>**
* [Reversible Instance Normalization for Accurate Time-Series Forecasting against Distribution Shift]()
* [ST-Norm: Spatial and Temporal Normalization for Multi-variate Time Series Forecasting]()
* [Time Series FeatuRe Extraction on basis of Scalable Hypothesis tests (tsfresh – A Python package)]()
* [Forecasting at Scale](https://github.com/facebook/prophet)
* [NeuralProphet: Explainable Forecasting at Scale](https://github.com/ourownstory/neural_prophet)
* [PyDMD: Python Dynamic Mode Decomposition](https://github.com/mathLab/PyDMD)
* [DON’T OVERFIT THE HISTORY -RECURSIVE TIME SERIES DATA AUGMENTATION](https://arxiv.org/abs/2207.02891)
* [Learning Fast and Slow for Online Time Series Forecasting](https://arxiv.org/abs/2202.11672) [fsnet](https://github.com/salesforce/fsnet)

* [A Hybrid System Based on Dynamic Selection for Time Series Forecasting](https://ieeexplore.ieee.org/document/9502692)
[code](https://github.com/EraylsonGaldino/Dref)

* [Improving forecast stability using deep learning](https://www.sciencedirect.com/science/article/pii/S016920702200098X)
[code](https://github.com/KU-Leuven-LIRIS/n-beats-s)
>> loss function (RMSSE for forecast error and RMSSC for forecast instability) instead of three different loss functions (MAPE, sMAPE, and MASE)

# Anomaly Detection
[Rethinking Graph Neural Networks for Anomaly Detection](https://proceedings.mlr.press/v162/tang22b.html)

Code [https://github.com/squareRoot3/Rethinking-Anomaly-Detection](https://github.com/squareRoot3/Rethinking-Anomaly-Detection)

[Deep Variational Graph Convolutional Recurrent Network for Multivariate Time Series Anomaly Detection](https://proceedings.mlr.press/v162/chen22x.html)

Code [https://github.com/SigmaLab01/DVGCRN](https://github.com/SigmaLab01/DVGCRN)

[Latent Outlier Exposure for Anomaly Detection with Contaminated Data](https://proceedings.mlr.press/v162/qiu22b.html)

Code [https://github.com/boschresearch/LatentOE-AD](https://github.com/boschresearch/LatentOE-AD)

[Time Series Anomaly Detection for Cyber-physical Systems via Neural System Identification and Bayesian Filtering](https://doi.org/10.1145/3447548.3467137)

Code [https://github.com/NSIBF/NSIBF](https://github.com/NSIBF/NSIBF)

# CNKI
[THU](https://navi.cnki.net/knavi/degreeunits/GQHAU/detail) Number of articles ***14893***

[PKU](https://navi.cnki.net/knavi/degreeunits/GBEJU/detail) Number of articles ***1548***

[SJTU](https://navi.cnki.net/knavi/degreeunits/GSJTU/detail) Number of articles ***65746***

[ZJU](https://navi.cnki.net/knavi/degreeunits/GZJDX/detail) Number of articles ***94398*** 

[WHU](https://navi.cnki.net/knavi/degreeunits/GWHDU/detail) Number of articles ***26703***  

[NJU](https://navi.cnki.net/knavi/degreeunits/GNJIU/detail) Number of articles ***33715***

[FDU](https://navi.cnki.net/knavi/degreeunits/GFUDU/detail) Number of articles ***35699***

[USTC](https://navi.cnki.net/knavi/degreeunits/GZKJU/detail) Number of articles ***23874***

[HUST](https://navi.cnki.net/knavi/degreeunits/GHZKU/detail) Number of articles ***89436***

[RUC](https://navi.cnki.net/knavi/degreeunits/GZRMU/detail) Number of articles ***666***

# Strange
> Reading articles written by Indians is not as good as reading articles written by Chinese

>> Hybrid wind speed forecasting using ICEEMDAN and transformer model with novel loss function[https://doi.org/10.1016/j.energy.2022.126383]

>> **(transformer network (TRA) with a novel KMSE loss function (NLF) for the WSF)**

>> A novel loss function of deep learning in wind speed forecasting[https://doi.org/10.1016/j.energy.2021.121808]

>> **(propose a kernel MSE loss function)**

> `There is no citation relationship in the above two articles`



