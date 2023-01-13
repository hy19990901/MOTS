# Test Input Selection for Deep Neural Network Enhancement Based on Multiple-Objective Optimization

## Introduction

Deep Neural Networks (DNNs) have been applied in many domains, such as autonomous driving and image recognition. However, due to the lower-than-expected performance of the DNN models in the real application, researchers are committed to sampling a test subset from test data with the limited labeling effort to retrain the DNN models for enhancement. Existing test input selection methods aim at selecting the test inputs according to the probability that is classified incorrectly by the DNN model. However, the test inputs selected by using existing methods might have similar features, making the DNN model unable to learn more diverse features when retraining. To address this limitation, this paper proposes Multiple-Objective Optimization-Based Test Input Selection (MOTS) to select more effective test subset to retrain the DNN model for enhancement. Different from existing works, this work not only considers the uncertainty of the test input but also takes the diversity of the test subset into account. Then MOTS uses a multiple-objective optimization algorithm NSGA-II to solve this problem, which ensures the test subset has more diverse features and is more helpful for retraining DNN models. This paper conducts the experiment on two popular DNN models and three widely-used datasets. The experiment results indicate that MOTS outperforms other baseline methods regarding the effectiveness of retraining DNN models.

## Required Libraries

* Python 3.7
* Numpy 1.18.1
* Tensorflow 1.15.6
* Keras 2.1.6
* geatpy 2.6.0

## Main Experimental Codes

```model_train``` : This folder contains the code related to model training.  The model training code of this work refers to [MCP](https://github.com/actionabletest/MCP). Please refer to their work if you need more information.

```adv``` : This folder contains part of the generated adversarial examples. Because some of datasets exceed GitHub's file size limit of 100.00 MB, we can only upload part of our datasets. Moreover, we used the framework by [Ma et al.](https://github.com/xingjunm/lid_adversarial_subspace_detection) to generate various adversarial examples (FGSM, BIM-A, BIM-B, JSMA, and C&W). Please refer to [craft_adv_samples.py](https://github.com/xingjunm/lid_adversarial_subspace_detection/blob/master/craft_adv_examples.py) in the above repository of Ma et al. 

```run.py``` : This file guides the execution of NSGA-II.

```EA.py``` : This file describes the basic configuration and the structure of NSGA-II.

```objectives.py``` : This file defines two objectives in NSGA-II.

```retrain.py``` : This file describes the details for model retraining.

## Baseline Methods

To evaluate the performance of MODS, we selected the following four approaches for comparison.

* MCP : [Shen et al.](https://github.com/actionabletest/MCP) proposed MCP (Multiple-Boundary Clustering and Prioritization), which is a technique to cluster the samples into the boundary areas of multiple boundaries for DL models and specify the priority to select samples evenly from all boundary areas, to make sure enough useful samples for each boundary reconstruction. The results of the experiments demonstrate that MCP is very effective in retraining DL model.

* DeepGini : [Feng et al.](https://github.com/853108389/deepgini) proposed DeepGini that prioritizes the test data and selects the most informative data that are more likely to be classified incorrectly by the DNN model. Its authors have also demonstrated that DeepGini is useful to guide model retraining.

* LSA/DSA : [Kim et al.](https://github.com/coinse/sadl) proposed a test adequacy criterion SADL, which includes LSA (Likelihood-based Surprise Adequacy) and DSA (Distance-based Surprise Adequacy). And SA can estimate the difference between a single test sample and the training set. The experimental results show that SA can guide selection of inputs for more effective retraining of DL systems. Therefore, we introduce LSA/DSA as the baseline approach.

* SRS : To evaluate the effect of randomly selected test inputs on model retraining, the Simple Random Sampling (SRS) is naturally taken as a baseline approach.  


