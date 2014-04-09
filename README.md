The python code found in this repository is aimed at reproducing the 90/10 baseline result of [1] on PubFig83, a benchmark for familiar face recognition in the wild. This code has not been exhaustively tested.

* Install the python package [convnet-rfw](http:/github.com/giovanichiachia/convnet-rfw).

* Download and extract the [original](https://dl.dropboxusercontent.com/u/275083/pubfig83.zip) or the [aligned](https://www.dropbox.com/s/0ez5p9bpjxobrfv/pubfig83-aligned.tar.bz2) version of PubFig83.

* Clone/fork this repository and execute the code made available from its root path with:

```
python pf83_baseline.py <DATASET_PATH>
```

You should expect as output something like this:

```
accuracy...
```

Requirements:

* numpy>=1.6.1
* scipy>=0.10.1
* [convnet-rfw](http:/github.com/giovanichiachia/convnet-rfw)
* [scikit-learn](http://scikit-learn.org/)>=0.12
* 8GB of RAM

Make sure they are all available in your python environment.

#####[[1]](http://www.coxlab.org/pdfs/cvpr2011_ws_fb_manuscript.pdf) Nicolas Pinto, Zak Stone, Todd Zickler, and David D. Cox, “Scaling-up Biologically-Inspired Computer Vision: A Case-Study on Facebook,” in IEEE Computer Vision and Pattern Recognition, Workshop on Biologically Consistent Vision, 2011.

