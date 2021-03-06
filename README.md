# WideOpenThoughts

Wide Open Thoughts (WOTs) is almost same software as [DeeperThought](https://github.com/pepe78/DeeperThought), but instead of CUDA C++ uses OpenCL C++ so that AMD GPU cards can be used as well as NVIDIA GPU cards.

Wide Open Thoughts is general framework for training Deep Neural Networks (and convolutional as well) using OpenCL C++.

To install OpenCL on linux:

> sudo apt update

> sudo apt install ocl-icd-opencl-dev

To compile on linux:

> ./compile.sh

Or run provided mnist example:

> ./run_mnist_example.sh

To compile on Windows, open sln solution.

To run training using your train and test data from command line execute:

> ./wots configFile trainFile testFile batchSize(integer) paramFile/null saveEveryNEpochs(integer) square/log wheremax/netflix/none


## Input format of data:

expOut_1, ... , expOut_n, inp_1, ... , inp_m

For both trainFile and testFile (expOut - expected output, inp - input). One data point is one line.

## Good research publications on the topic:

* N. Srivastava, G. Hinton, A. Krizhevsky, I. Sutskever, R. Salakhutdinov: [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](http://www.jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf?utm_content=buffer79b43&utm_medium=social&utm_source=twitter.com&utm_campaign=buffer), JMLR, 2014

* J. Masci, U. Meier, D. Ciresan, and J. Schmidhuber: [Stacked Convolutional Auto-Encoders for
Hierarchical Feature Extraction](https://www.researchgate.net/profile/Jonathan_Masci/publication/221078713_Stacked_Convolutional_Auto-Encoders_for_Hierarchical_Feature_Extraction/links/0deec518b9c6ed4634000000/Stacked-Convolutional-Auto-Encoders-for-Hierarchical-Feature-Extraction.pdf), Artificial Neural Networks and Machine Learning – ICANN, 2011

* H. Xiao, K. Rasul, R. Vollgra: [Fashion-MNIST: a Novel Image Dataset for
Benchmarking Machine Learning Algorithms](https://arxiv.org/pdf/1708.07747.pdf), Arxiv, 2017

## Results for other datasets:

* [Fashion MNIST results](./README_fashion.md)
* [Adult dataset results](./README_adult.md)

## Results for MNIST dataset:

You can download [MNIST dataset here](https://www.dropbox.com/s/9eoiignb7tlrr2u/mnist.zip?dl=0). Mnist dataset can be found in other formats on [Yann Lecun's web page](http://yann.lecun.com/exdb/mnist/index.html) or [Kaggle's web page](https://www.kaggle.com/c/digit-recognizer).

### configBB.txt

#### batch size 4000, log loss error, auto step size

> matrix,784,130,0.5,-0.001

> sigmoid,130

> dropout,130,0.5

> matrix,130,10,0.5,-0.001

> sigmoid,10

![graphBB](./pics/graphBB.png)

Top accuracy on test data: 97.47 %

### configBBAnd.txt

#### batch size 4000, log loss error, auto step size

> and,784,130,0.5,-0.001

> sigmoid,130

> dropout,130,0.5

> and,130,10,0.5,-0.001

> softmax,10

![graphBBAnd](./pics/BBAnd.png)

Top accuracy on test data: 97.86 %

### configBB2.txt

#### batch size 4000, log loss error, auto step size

> matrix,784,500,0.5,-0.001

> sigmoid,500

> matrix,500,250,0.5,-0.001

> sigmoid,250

> dropout,250,0.5

> matrix,250,130,0.5,-0.001

> sigmoid,130

> matrix,130,10,0.5,-0.001

> softmax,10

![graphBB2](./pics/bb2.png)

Top accuracy on test data: 98.35 %

### configBB2OneAnd.txt

#### batch size 4000, log loss error, auto step size

> and,784,500,0.5,-0.001

> sigmoid,500

> matrix,500,250,0.5,-0.001

> sigmoid,250

> dropout,250,0.5

> matrix,250,130,0.5,-0.001

> sigmoid,130

> matrix,130,10,0.5,-0.001

> softmax,10

![graphBB2](./pics/BB2OneAnd.png)

Top accuracy on test data: 98.59 %

### configG.txt (convolutional neural network CNN)

#### batch size 4000, log loss error, auto step size

> convolution,1,28,28,200,8,8,0.5,-0.001

> max,200,21,21,7,7

> matrix,1800,130,0.5,-0.001

> sigmoid,130

> dropout,130,0.5

> matrix,130,10,0.5,-0.001

> softmax,10

![graphG](./pics/graphG4.png)

Top accuracy on test data: 99.43 %

### configX.txt ([random convolutional neural network RCNN](https://www.frisky.world/2020/07/when-you-dont-train-parameters-of.html))

#### batch size 3000, log loss error, auto step size

> convolution,1,28,28,400,8,8,0.5,0

> max,400,21,21,7,7

> matrix,3600,130,0.5,-0.001

> sigmoid,130

> dropout,130,0.5

> matrix,130,10,0.5,-0.001

> softmax,10

![graphX](./pics/graphX.png)

### configY.txt ([sufficient random convolutional neural network SRCNN](https://www.frisky.world/2020/07/when-you-realize-you-need-only-64.html))

#### batch size 3000, log loss error, auto step size

> convolution,1,28,28,64,8,8,0.5,0

> convolution,1,64,441,200,64,1,0.5,-0.001

> max,200,21,21,7,7

> matrix,1800,130,0.5,-0.001

> sigmoid,130

> dropout,130,0.5

> matrix,130,10,0.5,-0.001

> softmax,10

![graphY](./pics/graphY.png)

### configZ.txt

#### batch size 3500, log loss error, auto step size

> convolution,1,28,28,200,8,8,0.5,-0.001

> max,200,21,21,7,7

> matrix,1800,130,0.5,-0.001

> sigmoid,130

> noise,130,0.1

> matrix,130,10,0.5,-0.001

> softmax,10

![graphZ](./pics/graphZ.png)

### configW.txt

#### batch size 3500, log loss error, auto step size

> convolution,1,28,28,200,8,8,0.5,-0.001

> max,200,21,21,7,7

> [noise](https://www.frisky.world/2020/08/when-you-use-noise-layer-instead-of.html),1800,0.5

> matrix,1800,130,0.5,-0.001

> sigmoid,130

> matrix,130,10,0.5,-0.001

> softmax,10

![graphW](./pics/graphW.png)

Top accuracy on test data: 99.36 %

Some of these results were verified with [PyTorch & Tensorflow](https://github.com/pepe78/MNISTWideNetwork).

## What you need to download and install beforehand:

* [Ubuntu for developers](https://www.ubuntu.com/desktop/developers) or [elementary OS](https://elementary.io/) or [Zorin OS](https://zorinos.com/download/#core) - it was [verified](https://www.frisky.world/2020/02/testing-elementary-os.html) & [verified](https://www.frisky.world/2020/03/testing-zorin-os-core.html) & [verified](https://www.frisky.world/2020/04/testing-ubuntu-2004-daily-build-kernel.html). Now working on Windows with [Visual Studio Community](https://visualstudio.microsoft.com/vs/community/) as well.

* [OpenCL](https://www.khronos.org/opencl/)

* GPU card

## Other alternatives for training DNNs:

* [CNTK](https://github.com/Microsoft/CNTK) - Microsoft

* [DeeperThought](https://github.com/pepe78/DeeperThought/)

* [Pytorch](https://pytorch.org/) - Facebook

* [TensorFlow](https://www.tensorflow.org/) - Google

