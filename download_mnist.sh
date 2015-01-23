#!/bin/bash

mkdir -p /media/datasets/mnist

if ! [ -e /media/datasets/mnist/train-images-idx3-ubyte.gz ]
	then
		wget -P /media/datasets/mnist/ http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
fi
gzip -d /media/datasets/mnist/train-images-idx3-ubyte.gz

if ! [ -e /media/datasets/mnist/train-labels-idx1-ubyte.gz ]
	then
		wget -P /media/datasets/mnist/ http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
fi
gzip -d /media/datasets/mnist/train-labels-idx1-ubyte.gz

if ! [ -e /media/datasets/mnist/t10k-images-idx3-ubyte.gz ]
	then
		wget -P /media/datasets/mnist/ http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
fi
gzip -d /media/datasets/mnist/t10k-images-idx3-ubyte.gz

if ! [ -e /media/datasets/mnist/t10k-labels-idx1-ubyte.gz ]
	then
		wget -P /media/datasets/mnist/ http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
fi
gzip -d /media/datasets/mnist/t10k-labels-idx1-ubyte.gz
