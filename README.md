
Electronic Music Tagger
-----------------------

Electronic Dance Music (EDM) is usually built from electronic sound sources such as synthesizers or drum machines. In the last decade, more and more software products such as Digital Audio Workstations (DAW) and software instruments have been used as well. Still, a major part of all samples are still derived from small audio snippets that are mixed into the music.

These samples could be base elements such as single snare hits, or whole rythmic loops, or even complex drum arrangements. In this project we assume that electronic dance music is comprised of atomic, identifiable elements. Thus, complex musical productions should be representable as superpositions of many short samples. We liken these samples to words in speech.

During the production of electronic dance music it is important to use samples that are similar to each other. The process of chosing these samples can be very complex. We try to create an algorithm to aid in this selection process. This should happen in three distinct steps: First, an existing collection of samples should be divided into different classes. Secondly, short parts of a musical track are classified according to these classes. For typical tracks, there will be no clear classification, but more likely a superposition of different probablities for each short part. Lastly, these classifications can be used to match the musical track back to the samples, thus re-synthesizing the music.

In order for this to work, the classes need not correspond to musically relevant classes such as bass drums or string instruments.

#### Block Processing

All the existing samples are cut up in blocks of 20 ms with 50% overlap. These blocks are used to calculate a set of features. The features are then used for classification.

#### Features

Several different features have been analyzed for their suitability for our purpose. We looked at classical features such as RMS, peak, crest factor as well as several spectral features such as the centroid, log centroid, variance, skewness, flatness, brightness and a slope mean. These features have been reduced to five abstract features using a principal component analysis (PCA). More complex features such as full spectra have been tried as well, but turned out to be no better than the simple features mentioned above.

#### Classification of the Samples

Each sample is now a collection of several blocks of features. The sample bank is organized into musically relevant classes. We used the Dynamic Time Warping (DTW) algorithm to calculate the difference between two samples. This enables us to use a K-Nearest-Neighbors algorithm to classify new test samples into these classes. 

Additionally, a modified [Fast K-Means][0]  algorithm is used to algorithmically cluster the sample base into other classes. The DTW distance can then be used to classify new samples into one of these classes. (This is equivalent to K-Nearest-Neighbors with K=1)

[0]: http://www.eecs.tufts.edu/~dsculley/papers/fastkmeans.pdf
