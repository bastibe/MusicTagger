import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.decomposition import PCA
from sklearn.lda import LDA

psd_data = pd.read_hdf('feature_data.hdf', 'psd')

np.random.seed()
idx = np.random.random_integers(len(psd_data), size=1000)
selected_data = psd_data.iloc[idx]

pca = PCA(n_components=2)
pca.fit(list(psd_data['psd']))
transformed_data_pca = pca.transform(list(selected_data['psd']))
print("explained variance ratio: %s" % str(pca.explained_variance_))
with open('pca_psd.pickle', 'wb') as f:
    pickle.dump(pca, f)

lda = LDA(n_components=2)
lda.fit(list(psd_data['psd']), psd_data['tag'])
transformed_data_lda = lda.transform(list(selected_data['psd']))

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
