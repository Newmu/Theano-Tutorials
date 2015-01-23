Theano-Tutorials
================

Bare bones introduction to machine learning from linear regression to convolutional neural networks using Theano.

***Dataset***
It's worth noting that this library assumes that the reader has access to the mnist dataset. This dataset is freely available and is accessible through Yann LeCun's [personal website](http://yann.lecun.com/exdb/mnist/).

If you want to automate the download of the dataset, there is an included file that will do this for you. Simply run the following:
`sudo ./download_mnist.sh`

***Known Issues***
`Library not loaded: /usr/local/opt/openssl/lib/libssl.1.0.0.dylib`
This results from a broken openssl installation on mac. It can be fixed by uninstalling and reinstalling openssl:
`sudo brew remove openssl`
`brew install openssl`
