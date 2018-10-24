===================================================================
============    DeepNeuralNet_QSAR Documentation     ==============
===================================================================

Authors: Yuting Xu, Junshui Ma. 

Contact: yuting.xu@merck.com, junshui_ma@merck.com.

Affiliation: Merck Biometrics Research, Merck Sharp & Dohme Corp. a subsidiary of Merck & Co., Inc., Kenilworth, NJ, USA.

Date: 02/07/2017

Acknowledgement: 
	This set of codes were developed based on George Dahl's Kaggle codes in Dec. 2012.

If you use the DeepNeuralNet_QSAR for scientific work that gets published, you should include in that publication a citation of the paper below:

Xu, Yuting, Junshui Ma, Andy Liaw, Robert P. Sheridan, and Vladimir Svetnik. "Demystifying Multitask Deep Neural Networks for Quantitative Structure–Activity Relationships." Journal of chemical information and modeling 57, no. 10 (2017): 2490-2504.


===================================================================
Basic info.
===================================================================

System requirements:
* Python 2.7+
* Required Python Modules: 
  - Python Modules installed by default: sys, os, argparse, itertools, gzip, time
  - General Python Modules:	numpy, scipy.sparse 
  - Special Python Modules: gnumpy, cudamat (if use GPU) or npmat (if use multiplec-core CPU)
* CUDA toolkit: a prerequisite of cudamat Python Module.


Installation of Special Python Modules:
	* gnumpy: http://www.cs.toronto.edu/~tijmen/gnumpy.html
	* npmat: http://www.cs.toronto.edu/~ilya/npmat.py
	* cudamat: https://github.com/cudamat/cudamat

Note: 
  - Modules "gnumpy" and "npmat" are also provided in this distribution.
  - If you have not GPU card or have problem installing cudamat module, the npmat.py module will use multiplec-core CPU to simulate the GPU computing. 
  - Create a directory for this moduel of DeepNeuralNet_QSAR, and keep all the python scripts in that directory. 

Usage:
* Start a commandline-window (in windows) or a terminal (in linux), and run the python scripts. Please refer to details below.


===================================================================
Brief explaination of all python files
===================================================================
All the files are listed in alphabetical order, not ordered by importance.
Please find more detailed comments of all individual functions inside each python file.

[activationFunctions.py]
	Define several classes of common activiation functions, such as ReLU/Linear/Sigmoid, along with their derivation or error function (if used for ouput layer).
	Used by [dnn.py]

[counter.py]
	Utilize sys.stderr to produce progress bar for each training epoch.
	Include several different classes of progress bar, but only "Progress" and "DummyProgBar" are used.
	Used by [dnn.py]

[DeepNeuralNetPredict.py]
	For making predictions for new compound structure with a single-task/multi-task DNN, which is trained by DeepNeuralNetTrain.py or DeepNeuralNetTrain_dense.py. 

[DeepNeuralNetTrain.py]
	For training a multi-task/single-task DNN with sparse QSAR dataset(s), accepts raw csv datasets or processed npz datasets.

[DeepNeuralNetTrain_dense.py]
	For training a multi-task DNN with dense QSAR dataset(s), accepts raw csv datasets or processed npz datasets.
	
[dnn.py]
	Key components of a simple feed forward neural network.
	Used by [DeepNeuralNetTrain.py], [DeepNeuralNetPredict.py], [DeepNeuralNetTrain_multi.py] and [DeepNeuralNetPredict_multi.py]

[DNNSharedFunc.py]
	A group of assistant functions, such as calculating R-squared, writing predictions into file. 
	Used by many other files in the package.

[gnumpy.py]
	A simple python module for GPU computing, the "GPU-version" of numpy module. 

[npmat.py]
	A simple python module which is required by gnumpy.py for the simulation mode. 
	If failed to import cudamat, using npmat (CPU computing) instead. 

[processData_sparse.py], [processData_dense.py]	
	Pre-processing a group of raw csv QSAR data sets(either sparse or dense) to sparse-matrix python file format (save as *.npz), 
	to facilitate later use.
	Contains many data-manipulation functions used by other files in the package.
	
	
===================================================================
How to use - Example scripts
===================================================================
0) Prepare input datasets
	[sparse datasets]
	* Arrange all the datasets as examples in "data_sparse" folder.
	* Example #1 (It is a subset of three tasks from the 15 Kaggle datasets): 
		 - Folder name: data_sparse
		 - Contains several datasets, each has training set and test set: 
				METAB_training.csv METAB_test.csv   
				OX1_training.csv   OX1_test.csv   
				TDI_training.csv   TDI_test.csv  
	* Example #2 (It is a single task selected from Kaggle datasets): 
		 - Folder name: data_sparse_single
		 - Contains one pair of training set and test set:
				METAB_training.csv METAB_test.csv   

	[dense datasets]
	* Arrange all the datasets as examples in "data_dense_raw" folder.
	* Example (It is a subsample from CYP datasets, which has 3 tasks): 
		 - Folder name: data_dense
		 - Contains two datasets, one training set and one test set: 
				training.csv  test.csv  

1) Pre-process data (Optional, can be skipped.)
	* preprocess sparse format datasets: create a new folder "data_sparse" under the working directory to save processed data.
		python processData_sparse.py data_sparse data_sparse_processed

	* preprocess dense format datasets: create a new folder "data_dense" under the working directory to save processed data, need to tell how many tasks are there in the dense dataset, such as "3" in the example datasets. 
		python processData_dense.py data_dense data_dense_processed 3

2) Train a single-task DNN for one QSAR task

	Default transformation of inputs is log; activation function is ReLU, minibatch size 128....

	The key parameters that need to be specify by user: 
	 - seed: random seed for the program. It is optional but better to be given for reproducibility. 
	 - CV: (optional) proportation of cross-validation subset which randomly sampled from training set
	 - test: (optional) whether to use the corresponding external test set for checking performance on test set during training.
	 - hid: DNN structure, specify the number of nodes at each layer. 
	 - dropouts: the drop out probability for each layer, to prevent over-fitting. 
	 - epochs: number of epochs for training
	 - data: path to the folder which contains a single QSAR task data, could contain raw csv file or processed npz file
	 - the last argument: where you want to save the trained model, if the folder doesn't exists it'll be created automatically

	* Example: use .csv raw data to train a single-task DNN for METAB, each corresponding processed .npz files will be automatically save to input data path
		python DeepNeuralNetTrain.py --seed=0 --CV=0.4 --test --hid=2000 --hid=1000 --dropouts=0_0.25_0.1 --epochs=10 --data=data_sparse_single models/METAB_single

	* Example: use .npz processed data to train a single-task DNN for METAB (recommended, loading data faster than raw data)
	Parameters are the same as above. The processed datasets in folder "data_sparse_single" is created in last step.
		python DeepNeuralNetTrain.py --seed=0 --CV=0.4 --test --hid=2000 --hid=1000 --dropouts=0_0.25_0.1 --epochs=10 --data=data_sparse_single models/METAB_single

	* Example: Without the optional 'CV' and 'test' arguments.
		python DeepNeuralNetTrain.py --seed=0 --hid=2000 --hid=1000 --dropouts=0_0.25_0.1 --epochs=10 --data=data_sparse_single models/METAB_single

3) Prediction with a single-task DNN
	
	The key parameters that need to be specify by user: 
	 - model: the path to previous trained model folder, e.g. the "models/METAB_single" from step 2). 
	 - data: path to the folder which contains a single QSAR task data, could contain raw csv file or processed npz file
	 - label: whether the "test" dataset have true label. Default is 0, but in this example it has true label. 
	 - rep: (optional) number of dropout prediction rounds. Default is 0, means don't perform dropout prediction.
	 - seed: random seed for the program, useful for dropout prediction. Optional but better to be given for reproducibility. 
	 - result: (optional) specify where to save the prediction results. Default is the same as model folder.

	* Example: use the previous trained single DNN model for METAB to perform prediction for its test data
		python DeepNeuralNetPredict.py --seed=0 --label=1 --rep=10 --data=data_sparse_single --model=models/METAB_single --result=predictions/METAB_single

	* Example: Without the optional 'rep' and 'PredictResultPath':
		python DeepNeuralNetPredict.py --label=1 --data=data_sparse_single --model=models/METAB_single

4) Train a multi-task DNN for the sparse datasets
	Need to use the processed datasets but not raw datasets.
	Parameters that are different from single-task DNN:
	 - data: path to the data folder that stores all the QSAR datasets
	 (Below are optional)
	 - mbsz: the minibatch size, default is 20, but for multi-task it may be modified to achieve better results
	 - keep: the datasets to keep in the model, if don't want to include all datasets in the 'data' folder
	 - watch: if use internal cross-validation set or external test set, choose to monitor the MSE and R-squared for certain task
	 - reducelearnRateVis: sometimes reduce the learning rate of the first layer helps the training process to converge better

	* Example: a multi-task DNN to model all the three sparse datasets: METAB, OX1, TDI
		python DeepNeuralNetTrain.py --seed=0 --hid=2000 --hid=1000 --dropouts=0_0.25_0.1 --epochs=5 --data=data_sparse models/multi_sparse_1
	
	* Example: load the previous trained model and continue the training process for more epochs. 
		python DeepNeuralNetTrain.py --seed=0 --hid=2000 --hid=1000 --dropouts=0_0.25_0.1 --epochs=10 --data=data_sparse --loadModel=models/multi_sparse_1 models/multi_sparse_continue

	* Example: with more optional parameters, keep only METAB and OX1 tasks and monitor OX1 task performance
		python DeepNeuralNetTrain.py --seed=0 --CV=0.4 --test --mbsz=30 --keep=METAB --keep=OX1 --watch=OX1 --hid=2000 --hid=1000 --dropouts=0_0.25_0.1 --epochs=10 --data=data_sparse models/multi_sparse_2

5) Prediction with multi-task DNN for the sparse datasets
	The parameter settings are the same as single-task DNN for sparse dataset. See step 3).
	Only difference:
	- data: path to the data folder that stores all the processed datasets (including test datasets).

	* Example: prediction for all the three sparse datasets with the model trained in previous step, save results to model folder:
		python DeepNeuralNetPredict.py --label=1 --data=data_sparse --model=models/multi_sparse_1

	* Example: prediction with the model for METAB and OX1, trained in previous step, with dropout prediction, and save result to another folder.
		python DeepNeuralNetPredict.py --label=1 --seed=0 --rep=10 --data=data_sparse --model=models/multi_sparse_2 --result=predictions/multi_sparse_2

6) Train a multi-task DNN for the dense datasets
	Most of the parameter settings are the same as multi-task DNN for sparse datasets
	Difference: use integer parameters for the 'keep' and 'watch' arguments
	The key parameters that need to be specify by user: 
	 - numberOfOutputs: number of QSAR task output columns in the raw training set (.csv)

	* Example: keep only the first two output tasks and monitor the first output during training process, with internal cross-validation set and external test set, using raw data
		python DeepNeuralNetTrain_dense.py --numberOfOutputs=3 --CV=0.4 --test --keep=0_1 --watch=0 --hid=2000 --hid=1000 --dropouts=0_0.25_0.1 --epochs=10 --data=data_dense models/multi_dense_1

	* Example: Without the optional arguments, using pre-processed data
	Note: for processed data, don't need to specify "--numberOfOutputs=3"
		python DeepNeuralNetTrain_dense.py --hid=2000 --hid=1000 --dropouts=0_0.25_0.1 --epochs=10 --data=data_dense_processed models/multi_dense_2

7) Prediction with multi-task DNN for the dense datasets
	Parameter settings are the same as prediction for sparse datasets

	* Example: Prediction using trained DNN from previous step
		python DeepNeuralNetPredict.py --label=1 --dense --data=data_dense --model=models/multi_dense_1 --result=predictions/multi_dense_1
