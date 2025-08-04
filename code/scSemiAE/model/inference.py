import numpy as np
import anndata
import scanpy as sc
from sklearn.cluster import KMeans

# evaluation metrics
def knn_infer(embd_space, lab_full, labeled_idx, unlabeled_idx):
	"""
	Predicts the labels of unlabeled data in the embedded space with KNN.
	Parameters
	----------
	embd_space : ndarray (n_samples, embedding_dim)
		Each sample is described by the features in the embedded space.
		Contains all samples, both labeled and unlabeled.
	labeled_idx : list
		Indices of the labeled samples (used for training the classifier).
	labeled_lab : ndarray (n_labeled_samples)
		Labels of the labeled samples.
	unlabeled_idx : list
		Indices of the unlabeled samples.
	Returns
	-------
	pred_lab : ndarray (n_unlabeled_samples)
		Inferred labels of the unlabeled samples.
	"""

	# obtain labeled data and unlabled data from indices
	labeled_samp = embd_space[labeled_idx, :]
	unlabeled_samp = embd_space[unlabeled_idx, :]

	from sklearn.neighbors import KNeighborsClassifier

	knn = KNeighborsClassifier(n_neighbors=10)

	knn.fit(labeled_samp, lab_full[labeled_idx])

	pred_lab = knn.predict(unlabeled_samp)
	return pred_lab

def log_infer(embd_space,lab_full, labeled_idx, unlabeled_idx):
	"""
	Infers the labels of unlabeled data in the embedded space with logistic regression.
	Same parameters as knn_infer.
	"""
	# obtain labeled data and unlabled data from indices
	labeled_samp = embd_space[labeled_idx, :]
	unlabeled_samp = embd_space[unlabeled_idx, :]

	from sklearn.linear_model import LogisticRegression
	log = LogisticRegression(penalty="l2", C=1, multi_class="multinomial", solver="lbfgs", max_iter=1e4)

	log.fit(labeled_samp, lab_full[labeled_idx])

	pred_lab = log.predict(unlabeled_samp)
	return pred_lab


def louvain_cluster_infer(embd_space, unlabeled_idx):
	"""
	Cluster unlabeled cells in the embedded space with louvain.
	Parameters
	----------
	embd_space : ndarray (n_samples, embedding_dim)
		Each sample is described by the features in the embedded space.
		Contains all samples, both labeled and unlabeled.
	unlabeled_idx : list
		Indices of the unlabeled samples.
	Returns
	-------
	pred : ndarray (n_unlabeled_samples)
		result of louvain algorithm. 
	"""
	em = anndata.AnnData(embd_space[unlabeled_idx])
	sc.pp.neighbors(em, n_neighbors=30, use_rep='X')
	sc.tl.louvain(em)
	pred = em.obs['louvain'].to_list()
	pred = list(map(int, pred))
	return np.array(pred)

def kmeans_cluster_infer(embd_space, unlabeled_idx, n_cls = 10):
	"""
	Cluster unlabeled cells in the embedded space with kmeans.
	Parameters
	----------
	embd_space : ndarray (n_samples, embedding_dim)
		Each sample is described by the features in the embedded space.
		Contains all samples, both labeled and unlabeled.
	unlabeled_idx : list
		Indices of the unlabeled samples.
	n_cls: int
		the number of clusters
	Returns
	-------
	pred : ndarray (n_unlabeled_samples)
		result of KMeans algorithm. 
	"""
	pred = KMeans(n_clusters=n_cls).fit_predict(embd_space[unlabeled_idx])
	return pred