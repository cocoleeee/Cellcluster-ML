import numpy as np
import random
import pandas as pd

class Data():

	def __init__(self, load_path, labeled_size=None, labeled_ratio=None, seed=0):
		"""
		Parameters
		----------
		load_path : string
			Data directory.
		labeled_ratio : float
			Ratio of samples to retain labels. The other samples' labels are all hidden.
		labeled_size : int
			Number of samples per class to retain labels.
			Choose to specify either labeled_ratio or labeled_size, but not both.
		seed : int
			Random seed for choosing labeled and unlabeled sets.
		"""
		self.seed = seed
		self.dataset = pd.read_csv(load_path+"expression_matrix.txt", sep="\t", index_col=0).sort_index()
		self.metadata = pd.read_csv(load_path+"metadata.txt", sep="\t", index_col=0).sort_index()
		self.labeled_size = labeled_size
		self.labeled_ratio = labeled_ratio

	def load_all(self, label_name="celltype", batch_name=None, number_of_hide=0):
		"""
		Load data
		
		Parameters
		----------
		label_name : string
			label name in the dataframe of metadata
		batch_name : string
			batch name in the dataframe of metadata
		number of hide : int
			simulate the different numbers of hidden cell labels which cannot be clutched 

		Returns
		-------
		expr : 
			expression matrix
		lab_full : list
			labels for all cells
		labeled_idx: list
			Indices of the labeled samples.
		unlabeled_idx: list
			Indices of the unlabeled samples.
		info: 
			some information used possibly
		"""
		print("loading all data...")
		expr = self.dataset.values.astype(float)

		# Convert label to numeric format
		if label_name in self.metadata.columns:		
			cell_label, lab_full = celllabel_to_numeric(self.metadata[label_name])
		else: 
			raise Exception("Invalid label name!")

		# manually hide labels
		if number_of_hide == 0:
			labeled_idx, unlabeled_idx = self.hide_labs(lab_full)
		else:
			num = len(set(cell_label)) - number_of_hide
			labeled_idx, unlabeled_idx = self.hide_some_labs(lab_full, num)

		# combine information not necessary
		info = {}
		info["cell_label"] = cell_label
		info["cell_id"] = self.dataset.index
		info["gene_names"] = self.dataset.columns

		# if there exist betch effects
		if batch_name in self.metadata.columns:
			tech_type, batch_full = techtype_to_numeric(self.metadata.tech)
			info['batch_name'] = tech_type
			info["batch"] = batch_full

		print("expression set dimensions:", expr.shape)
		return expr, lab_full, labeled_idx, unlabeled_idx, info

	def hide_labs(self, lab):
		"""
		Hide a portion of the labels to simulate semi-supervised learning.

		Parameters
		----------
		lab : ndarray (1, nsamples)
			Complete labels of all samples.

		Returns
		-------
		labeled_idx : list
			Indices of the labeled samples.
		unlabeled_idx : list
			Indices of the unlabeled samples.
		"""

		idx_byclass = gen_idx_byclass(lab)

		# keep track of a list of sample indices whose labels are retained
		labeled_idx = []
		unlabeled_idx = []

		for class_label in idx_byclass:
			print("class: {}, size: {}".format(class_label, len(idx_byclass[class_label])))
			# Process for this class:
			idx_thisclass = idx_byclass[class_label]
			# shuffle with seed
			# this ensures that the labeled set is always the same whenever you obtain data from this module
			random.Random(self.seed).shuffle(idx_thisclass)

			# append indices
			if self.labeled_ratio is not None:
				self.labeled_size = int(self.labeled_ratio * len(idx_thisclass))

			if self.labeled_size >= len(idx_thisclass):
				print("Specified labeled_size is greater than number of samples for class", class_label)
				print("Use all samples of this class instead.")
				labeled_idx += idx_thisclass
			else:
				labeled_idx += idx_thisclass[0:self.labeled_size]
				unlabeled_idx += idx_thisclass[self.labeled_size:]

		# print labeled samples indices
		print("seed:", self.seed)
		print("labeled sample idx:", labeled_idx)
		print("labeled set size:", len(labeled_idx))

		return labeled_idx, unlabeled_idx

	def hide_some_labs(self, lab, number):
		"""
		basic: Hide a portion of the labels to simulate semi-supervised learning.
		and: The labels only contains some of all labels with more than 50 cells, not complete

		Parameters
		----------
		lab : ndarray (1, nsamples)
			Complete labels of all samples.
		number: float (0， nsamples)
			The number of  labels types that are not all hided

		Returns
		-------
		labeled_idx : list
			Indices of the labeled samples.
		unlabeled_idx : list
			Indices of the unlabeled samples.
		"""
		idx_byclass = gen_idx_byclass(lab)

		# keep track of a list of sample indices whose labels are retained
		labeled_idx = []
		unlabeled_idx = []

		# if need to sort with the number of cells 
		# idx_byclass = dict(sorted(idx_byclass.items(), key=lambda e: len(e[1]), reverse=True))
		random.Random(self.seed).shuffle(idx_byclass)
		idx = 0
		for class_label in idx_byclass:
			if len(idx_byclass[class_label]) < 50:
				unlabeled_idx += idx_byclass[class_label]
				continue
			else:
				if idx < number:
					idx += 1
					print("class: {}, size: {}".format(class_label, len(idx_byclass[class_label])))

					# Process for this class:
					idx_thisclass = idx_byclass[class_label]
					# shuffle with seed
					# this ensures that the labeled set is always the same whenever you obtain data from this module
					random.Random(self.seed).shuffle(idx_thisclass)

					# append indices
					if self.labeled_ratio is not None:
						self.labeled_size = int(self.labeled_ratio * len(idx_thisclass))

					if self.labeled_size >= len(idx_thisclass):
						print("Specified labeled_size is greater than number of samples for class", class_label)
						print("Use all samples of this class instead.")
						labeled_idx += idx_thisclass
					else:
						labeled_idx += idx_thisclass[0:self.labeled_size]
						unlabeled_idx += idx_thisclass[self.labeled_size:]
				else:
					unlabeled_idx += idx_byclass[class_label]

		# print labeled samples indices
		print("seed:", self.seed)
		print("labeled sample idx:", labeled_idx)
		print("labeled set size:", len(labeled_idx))

		return labeled_idx, unlabeled_idx


def gen_idx_byclass(labels):
    """
    Neatly organize indices of labeled samples by their classes.

    Parameters
    ----------
    labels : list
        Note that labels should be a simple Python list instead of a tensor.

    Returns
    -------
    idx_byclass : dictionary {[class_label (int) : indices (list)]}
    """
    # print("in gen_idx_byclass...")
    from collections import Counter
    classes = Counter(labels).keys()  # obtain a list of classes
    idx_byclass = {}

    for class_label in classes:
        # Find samples of this class:
        class_idx = []  # indices for samples that belong to this class
        for idx in range(len(labels)):
            if labels[idx] == class_label:
                class_idx.append(idx)
        idx_byclass[class_label] = class_idx

    return idx_byclass


def celllabel_to_numeric(celllabel):
	"""
    convert cell label to numeric format

    Parameters
    ----------
    celllabel: list 

    Returns
    -------
    mapping: 
		mapping number to celllabel
	truth_label: 
		list of int
    """
	lab = celllabel.tolist()
	lab_set = sorted(set(lab))

	mapping = {lab:idx for idx,lab in enumerate(lab_set)}    
	truth_labels = [mapping[l] for l in lab]
	truth_labels = np.array(truth_labels).astype(int)
	print(mapping)
	mapping = {idx:lab for idx, lab in enumerate(lab_set)}
	return mapping, truth_labels


def techtype_to_numeric(batch_name):
	"""
    convert batch name to numeric format

    Parameters
    ----------
    batch_name: list 

    Returns
    -------
    mapping: 
		mapping number to batch name
	truth_batches: 
		list of int
    """
	batch = batch_name.tolist()
	bth_set = sorted(set(batch))

	mapping = {bth: idx for idx, bth in enumerate(bth_set)}

	truth_batches = [mapping[b] for b in batch]
	truth_batches = np.array(truth_batches).astype(int)

	mapping = {idx:bth for idx, bth in enumerate(bth_set)}
	return mapping, truth_batches