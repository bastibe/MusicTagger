
Electronic Music Tagger
-----------------------

Electronic Dance Music (EDM) is usually built from electronic sound sources such as synthesizers or drum machines. In the last decade, more and more software products such as Digital Audio Workstations (DAW) and software instruments have been used as well. Still, a major part of all samples are still derived from small audio snippets that are mixed into the music.

These samples could be base elements such as single snare hits, or whole rhythmic loops, or even complex drum arrangements. In this project we assume that electronic dance music is comprised of atomic, identifiable elements. Thus, complex musical productions should be representable as superpositions of many short samples. We liken these samples to words in speech.

During the production of electronic dance music it is important to use samples that are similar to each other. The process of choosing these samples can be very complex. We try to create an algorithm to aid in this selection process. This should happen in three distinct steps: First, an existing collection of samples should be divided into different classes. Secondly, short parts of a musical track are classified according to these classes. For typical tracks, there will be no clear classification, but more likely a superposition of different probabilities for each short part. Lastly, these classifications can be used to match the musical track back to the samples, thus re-synthesizing the music.

In order for this to work, the classes need not correspond to musically relevant classes such as bass drums or string instruments.

#### Block Processing

All the existing samples are cut up in blocks of 20 ms with 50% overlap. These blocks are used to calculate a set of features. The features are then used for classification. You can use the script *extract_features.py* to calculate a Pandas DataFrame (saved as HDF file) containing all feature data from all files in the sample base. Since this calculation is very time consuming, an already calculated database like this is available as *feature_data.hd5*. You can use the script *preprocess.py* to preprocess one sample for testing purposes.

#### Features

Several different features have been analyzed for their suitability for our purpose. We looked at classical features such as RMS, peak, crest factor as well as several spectral features such as the centroid, log centroid, variance, skewness, flatness, brightness and a slope mean. These features have been reduced to five abstract features using a principal component analysis (PCA). An already calculated and pickled PCA object is available as *pca.pickle*.

More complex features such as full spectra have been tried as well, but turned out to be no better than the simple features mentioned above. The file *features.py* contains all the features calculated.

#### Classification of the Samples

Each sample is now a collection of several blocks of features. The sample bank is organized into musically relevant classes. We used the Dynamic Time Warping (DTW) algorithm to calculate the difference between two samples. You can use *dynamic_time_warping.py* to calculate the DTW distance between two samples. The DTW algorithm was initially implemented in Python, but found to be too slow. Now, *dynamic_time_warping.py* by default uses a C implementation (*dtw.c*), which should be compiled to a shared library called *dtw.dll* or *dtw.so*. This provides this critical algorithm with a performance boost of about two orders of magnitude (100x). The C library is called using the CFFI.

This algorithm enables us to use a *k*-NN ("k-nearest-neighbors") algorithm to classify new test samples into these classes. You can use *knn_classify.py* to classify one sample using this algorithm.

Additionally, a modified [Fast *k*-Means][0] algorithm is used to algorithmically cluster the sample base into other classes. You can use the script *compare_all_data.py* to calculate a Pandas DataFrame (saved as HDF file) that contains all distances of all samples to all samples. Since this computation is very time consuming, an already calculated database is available as *distances.hd5*. 

The DTW distance can then be used to classify new samples into one of these classes. (This is equivalent to K-Nearest-Neighbors with K=1). You can use *k_means_classify.py* to calculate cluster centroids and classify a sample to one of those centroids.

[0]: http://www.eecs.tufts.edu/~dsculley/papers/fastkmeans.pdf

### Requirements

In order to run the programs in this project, you need to have the following Python packages installed:

- Numpy
- Scipy
- Pandas
- PyTables
- PySoundFile
- CFFI
- docopt
- sklearn

This has been developed using Python 3.3 on Windows and Mac

### Motivation

This was created as part of a homework project. The homework report can be downloaded [here](https://github.com/bastibe/MusicTagger/raw/master/Report/Main.pdf).
