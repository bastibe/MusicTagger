import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.decomposition import PCA
from sklearn.lda import LDA

feature_data = pd.read_hdf('feature_data.hdf', 'features')

np.random.seed()
idx = np.random.random_integers(len(feature_data), size=1000)
features = ['crest_factor', 'log_spectral_centroid', 'peak', 'rms',
            'spectral_abs_slope_mean', 'spectral_brightness', 'spectral_centroid',
            'spectral_flatness', 'spectral_skewness', 'spectral_variance']
selected_data = feature_data.iloc[idx]

pca = PCA(n_components=2)
pca.fit(feature_data[features])
transformed_data_pca = pca.transform(selected_data[features])
print("explained variance ratio: %s" % str(pca.explained_variance_))

lda = LDA(n_components=2)
lda.fit(feature_data[features], feature_data['tag'])
transformed_data_lda = lda.transform(selected_data[features])

tags = selected_data['tag'].unique()
colormap = mpl.cm.get_cmap("spectral")

fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(1,2,1)
ax1.set_title("Two most significant dimensions of PCA""")
for idx, tag in enumerate(tags):
    ax1.scatter(transformed_data_pca[selected_data['tag'] == tag, 0],
                transformed_data_pca[selected_data['tag'] == tag, 1],
                color=colormap(idx/len(tags)), label=tag)

ax2 = fig.add_subplot(1,2,2)
ax2.set_title("Two most significant dimensions of LDA")
for idx, tag in enumerate(tags):
    ax2.scatter(transformed_data_lda[selected_data['tag'] == tag, 0],
                transformed_data_lda[selected_data['tag'] == tag, 1],
                color=colormap(idx/len(tags)), label=tag)

# resize axes so there is room for the legend
for ax in [ax1, ax2]:
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width, box.height*0.8])
# collect all plots in one of the axes
plots = [p for p in ax1.get_children() if
         isinstance(p, mpl.collections.PathCollection)]
plt.figlegend(plots, tags, loc='upper center', ncol=4)
plt.show()
