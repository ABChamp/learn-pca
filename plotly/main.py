import pandas as pd
import plotly
import plotly.plotly as py
from plotly.graph_objs import *
import plotly.tools as tls
import numpy as np
from sklearn.preprocessing import  StandardScaler

df = pd.read_csv(
    filepath_or_buffer="./data/iris.data",
    header=None,
    sep=','
)

df.columns=['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid', 'class']
df.dropna(how="all", inplace=True) # drop the empty line at file-end

X = df.ix[:,0:4].values
y = df.ix[:,4].values

""" traces = []

legend = {0:False, 1:False, 2:False, 3:True}

colors = {
    'Iris-setosa': 'rgb(31, 119, 180)',
    'Iris-versicolor': 'rgb(255, 127, 14)',
    'Iris-virginica': 'rgb(44, 160, 44)'
}

for col in range(4):
    for key in colors:
        traces.append(Histogram(x=X[y==key, col],
                        opacity=0.75,
                        xaxis='x%s' %(col+1),
                        marker=Marker(color=colors[key]),
                        name=key,
                        showlegend=legend[col]))

data = Data(traces)

layout = Layout(barmode='overlay',
                xaxis=XAxis(domain=[0, 0.25], title='sepal length (cm)'),
                xaxis2=XAxis(domain=[0.3, 0.5], title='sepal width (cm)'),
                xaxis3=XAxis(domain=[0.55, 0.75], title='petal length (cm)'),
                xaxis4=XAxis(domain=[0.8, 1], title='petal width (cm)'),
                yaxis=YAxis(title='count'),
                title='Distribution of the different Iris flower features')

fig = Figure(data=data, layout=layout)
plotly.offline.plot(fig) """

# Standardizing
# the covariance matrix depends on the measurement scales of the original features
# transformation of the data onto unit scale (mean=0 and variance=1)
X_std = StandardScaler().fit_transform(X)
mean_vec = np.mean(X_std, axis=0)
# cov_mat = (X_std - mean_vec).T.dot((X_std - mean_vec)) / (X_std.shape[0]-1)
cov_mat = np.cov(X_std.T)
eig_vals, eig_vecs = np.linalg.eig(cov_mat)

print('Eigenvectors \n%s' % eig_vecs)
print('Eigenvalues \n%s' % eig_vals)

for ev in eig_vecs:
    np.testing.assert_array_almost_equal(1.0, np.linalg.norm(ev))
