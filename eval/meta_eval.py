from __future__ import print_function

import numpy as np
from scipy.stats import t
from tqdm import tqdm

import torch
import torch.nn.functional as F
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import kneighbors_graph as k_graph
import scipy as sp
import faiss
from sklearn.semi_supervised import LabelSpreading
from sklearn.ensemble import RandomForestClassifier


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), sp.stats.sem(a)
    h = se * t._ppf((1+confidence)/2., n-1)
    return m, h


def normalize(x):
    norm = x.pow(2).sum(1, keepdim=True).pow(1. / 2)
    out = x.div(norm)
    return out


def meta_test(net, testloader, use_logit=True, is_norm=True, classifier='LR'):
    net = net.eval()
    acc = []

    with torch.no_grad():
        for idx, data in tqdm(enumerate(testloader)):
           # print("iteration: ", idx)
            support_xs, support_ys, query_xs, query_ys = data
            support_xs = support_xs.cuda()
            query_xs = query_xs.cuda()
            batch_size, _, height, width, channel = support_xs.size()
            support_xs = support_xs.view(-1, height, width, channel)
            query_xs = query_xs.view(-1, height, width, channel)
           # print(support_xs.shape)
            if use_logit:
                support_features = net(support_xs).view(support_xs.size(0), -1)
                query_features = net(query_xs).view(query_xs.size(0), -1)
               # print(query_features.shape)
            else:
                feat_support, _ = net(support_xs, is_feat=True)
                #print("Feature support: ",len(feat_support))
                support_features = feat_support[-1].view(support_xs.size(0), -1)
                #print("Support features: ", support_features.shape)
                feat_query, _ = net(query_xs, is_feat=True)
                query_features = feat_query[-1].view(query_xs.size(0), -1)

            if is_norm:
                support_features = normalize(support_features)
                query_features = normalize(query_features)

            support_features = support_features.detach().cpu().numpy()
            query_features = query_features.detach().cpu().numpy()

            support_ys = support_ys.view(-1).numpy()
            query_ys = query_ys.view(-1).numpy()

            if classifier == 'LR':
                clf = LogisticRegression(random_state=0, solver='lbfgs', max_iter=1000,
                                         multi_class='multinomial')
                clf.fit(support_features, support_ys)
                query_ys_pred = clf.predict(query_features)
            elif classifier == 'NN':
                query_ys_pred = NN(support_features, support_ys, query_features)
            elif classifier == 'Cosine':
                query_ys_pred = Cosine(support_features, support_ys, query_features)
            elif classifier =="propagation":
                query_ys_pred = LabelPropagation(support_features, support_ys, query_features)
            elif classifier =="my_avrithis":
                query_ys_pred = lp_deepssl(support_features, support_ys, query_features)
            elif classifier =="original_avrithis":
                query_ys_pred, _ = update_plabels(support_features, support_ys, query_features)
            elif classifier == "iterate_diffusion":
                query_ys_pred, _ = iterative_lp(support_features, support_ys, query_features)
            else:
                raise NotImplementedError('classifier not supported: {}'.format(classifier))

            acc.append(metrics.accuracy_score(query_ys, query_ys_pred))

    return mean_confidence_interval(acc)


def NN(support, support_ys, query):
    """nearest classifier"""
    support = np.expand_dims(support.transpose(), 0)
    query = np.expand_dims(query, 2)

    diff = np.multiply(query - support, query - support)
    distance = diff.sum(1)
    min_idx = np.argmin(distance, axis=1)
    pred = [support_ys[idx] for idx in min_idx]
    return pred


def Cosine(support, support_ys, query):
    """Cosine classifier"""
    support_norm = np.linalg.norm(support, axis=1, keepdims=True)
    support = support / support_norm
    query_norm = np.linalg.norm(query, axis=1, keepdims=True)
    query = query / query_norm

    cosine_distance = query @ support.transpose()
    max_idx = np.argmax(cosine_distance, axis=1)
    pred = [support_ys[idx] for idx in max_idx]
    return pred

def LabelPropagation(support, support_ys, query):
    alpha = 0.3
    k_neighbours = 38
    all_embeddings = np.concatenate((support, query), axis=0)
    #X = all_embeddings.cpu().detach().numpy()
    labels = np.full(all_embeddings.shape[0], -1.)
    labels[:support.shape[0]] = support_ys
    label_propagation = LabelSpreading(kernel='knn', alpha=alpha, n_neighbors=k_neighbours, tol=0.000001)
    label_propagation.fit(all_embeddings, labels)
    predicted_labels = label_propagation.transduction_
    query_prop = predicted_labels[support.shape[0]:]
    return query_prop

def update_plabels(support, support_ys, query):
    #print("enter")
    #sel_acc = train_data.update_plabels(feats, k=args.dfs_k, max_iter=20)
    max_iter = 20
    no_classes = support_ys.max() + 1
    k = 15
    alpha = 0.6
    X = np.concatenate((support, query), axis=0)
    labels = np.zeros(X.shape[0])
    labels[:support_ys.shape[0]]= support_ys
    #print(labels.shape)
    labeled_idx = np.arange(support.shape[0])
    unlabeled_idx = np.arange(query.shape[0]) + support.shape[0]
    #print(unlabeled_idx)

    # kNN search for the graph
    d = X.shape[1]
    res = faiss.StandardGpuResources()
    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.device = int(torch.cuda.device_count()) - 1
    index = faiss.GpuIndexFlatIP(res, d, flat_config)  # build the index

    faiss.normalize_L2(X)
    index.add(X)
    N = X.shape[0]
    Nidx = index.ntotal

   # c = time.time()
    D, I = index.search(X, k + 1)
   # elapsed = time.time() - c
   # print('kNN Search done in %d seconds' % elapsed)

        # Create the graph
    D = D[:, 1:] ** 3
    I = I[:, 1:]
    row_idx = np.arange(N)
    row_idx_rep = np.tile(row_idx, (k, 1)).T
    W = sp.sparse.csr_matrix((D.flatten('F'), (row_idx_rep.flatten('F'), I.flatten('F'))), shape=(N, N))
    W = W + W.T

        # Normalize the graph
    W = W - sp.sparse.diags(W.diagonal())
    S = W.sum(axis=1)
    S[S == 0] = 1
    D = np.array(1. / np.sqrt(S))
    D = sp.sparse.diags(D.reshape(-1))
    Wn = D * W * D

        # Initiliaze the y vector for each class (eq 5 from the paper, normalized with the class size) and apply label propagation
    Z = np.zeros((N, no_classes))
    A = sp.sparse.eye(Wn.shape[0]) - alpha * Wn
    for i in range(no_classes):
        cur_idx = labeled_idx[np.where(labels[labeled_idx] == i)]
        y = np.zeros((N,))
        y[cur_idx] = 1.0 #/ cur_idx.shape[0]
        f, _ = sp.sparse.linalg.cg(A, y, tol=1e-6, maxiter=max_iter)
        Z[:, i] = f

    # Handle numberical errors
    Z[Z < 0] = 0

        # Compute the weight for each instance based on the entropy (eq 11 from the paper)
    probs_l1 = F.normalize(torch.tensor(Z), 1).numpy()
    #print(np.sum(probs_l1[0]))
    probs_l1[probs_l1 < 0] = 0
    entropy = sp.stats.entropy(probs_l1.T)
    weights = 1 - entropy / np.log(no_classes)
    weights = weights / np.max(weights)
    p_labels = np.argmax(probs_l1, 1)
    p_probs = np.amax(probs_l1,1)
    p_rank =  sp.stats.rankdata( p_probs, method='ordinal')
    #print(p_probs)

        # Compute the accuracy of pseudolabels for statistical purposes
        #correct_idx = (p_labels == labels)
        #acc = correct_idx.mean()

    p_labels[labeled_idx] = labels[labeled_idx]
    weights[labeled_idx] = 1.0

        # self.p_weights = weights.tolist()
        # self.p_labels = p_labels
        #
        # # Compute the weight for each class
        # for i in range(len(self.classes)):
        #     cur_idx = np.where(np.asarray(self.p_labels) == i)[0]
        #     self.class_weights[i] = (float(labels.shape[0]) / len(self.classes)) / cur_idx.size
    return p_labels[support.shape[0]:], weights

def lp_deepssl(support, support_ys, query):
    #A. Iscen, G. Tolias, Y. Avrithis, O. Chum. "Label Propagation for Deep Semi-supervised Learning", CVPR 2019
    alpha = 0.99
    k_neighbours = 7
    #concatenate embeddings
    all_embeddings = np.concatenate((support, query), axis=0)
    #get the Y matrix
    support_hot = np.zeros((support_ys.shape[0], support_ys.max() + 1))
    support_hot[np.arange(support_ys.shape[0]), support_ys] = 1
    query_hot = np.zeros((query.shape[0], support_ys.max() + 1))
    Y = np.concatenate((support_hot, query_hot), axis=0)
    #get the W matrix
    A = k_graph(all_embeddings, k_neighbours, mode='connectivity', include_self=False)
    A = A.toarray()
    W = np.zeros((A.shape[0], A.shape[1]))
    cos_sim = cosine_similarity(all_embeddings, all_embeddings)
    cos_sim[cos_sim > 1] = 1
    W = np.multiply(cos_sim, A)
    W[W == -0] = 0
    W = W + W.T
    D = np.eye(W.shape[0]) * k_neighbours
    D12 = sp.linalg.sqrtm(np.linalg.matrix_power(D, -1))
    W_norm = D12.dot(W).dot(D12)
    F = np.eye(W_norm.shape[0]) - alpha * W_norm
    Z = np.linalg.matrix_power(F, -1).dot(Y)
    y_pred = np.argmax(Z, axis=1)
    query_prop = y_pred[support.shape[0]:]
    return query_prop
    #
    # # Normalize the graph
    # W = W - scipy.sparse.diags(W.diagonal())
    # S = W.sum(axis=1)
    # S[S == 0] = 1
    # D = np.array(1. / np.sqrt(S))
    # D = scipy.sparse.diags(D.reshape(-1))
    # Wn = D * W * D
    #
    # # Initiliaze the y vector for each class (eq 5 from the paper, normalized with the class size) and apply label propagation
    # Z = np.zeros((N, len(self.classes)))
    # A = scipy.sparse.eye(Wn.shape[0]) - alpha * Wn
    # for i in range(len(self.classes)):
    #     cur_idx = labeled_idx[np.where(labels[labeled_idx] == i)]
    #     y = np.zeros((N,))
    #     y[cur_idx] = 1.0 / cur_idx.shape[0]
    #      f, _ = scipy.sparse.linalg.cg(A, y, tol=1e-6, maxiter=max_iter)
    #     Z[:, i] = f
    #
    # # Handle numberical errors
    # Z[Z < 0] = 0
    #
    # # Compute the weight for each instance based on the entropy (eq 11 from the paper)
    # probs_l1 = F.normalize(torch.tensor(Z), 1).numpy()
    # probs_l1[probs_l1 < 0] = 0
    # entropy = scipy.stats.entropy(probs_l1.T)
    # weights = 1 - entropy / np.log(len(self.classes))
    #  weights = weights / np.max(weights)
    # p_labels = np.argmax(probs_l1, 1)

def  iterative_lp(support, support_ys, query):
    max_iter = 20
    no_classes = support_ys.max() + 1
    k = 15
    alpha = 0.6
    X = np.concatenate((support, query), axis=0)
    labels = np.zeros(X.shape[0])
    labels[:support_ys.shape[0]] = support_ys
    # print(labels.shape)
    labeled_idx = np.arange(support.shape[0])
    unlabeled_idx = np.arange(query.shape[0]) + support.shape[0]
    # print(unlabeled_idx)

    # kNN search for the graph
    d = X.shape[1]
    res = faiss.StandardGpuResources()
    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.device = int(torch.cuda.device_count()) - 1
    index = faiss.GpuIndexFlatIP(res, d, flat_config)  # build the index

    faiss.normalize_L2(X)
    index.add(X)
    N = X.shape[0]
    Nidx = index.ntotal

    # c = time.time()
    D, I = index.search(X, k + 1)
    # elapsed = time.time() - c
    # print('kNN Search done in %d seconds' % elapsed)

    # Create the graph
    D = D[:, 1:] ** 3
    I = I[:, 1:]
    row_idx = np.arange(N)
    row_idx_rep = np.tile(row_idx, (k, 1)).T
    W = sp.sparse.csr_matrix((D.flatten('F'), (row_idx_rep.flatten('F'), I.flatten('F'))), shape=(N, N))
    W = W + W.T

    # Normalize the graph
    W = W - sp.sparse.diags(W.diagonal())
    S = W.sum(axis=1)
    S[S == 0] = 1
    D = np.array(1. / np.sqrt(S))
    D = sp.sparse.diags(D.reshape(-1))
    Wn = D * W * D

    iterate = True
    while iterate == True:
    # Initiliaze the y vector for each class (eq 5 from the paper, normalized with the class size) and apply label propagation
        Z = np.zeros((N, no_classes))
        A = sp.sparse.eye(Wn.shape[0]) - alpha * Wn
        for i in range(no_classes):
            cur_idx = labeled_idx[np.where(labels[labeled_idx] == i)]
            y = np.zeros((N,))
            y[cur_idx] = 1.0  # / cur_idx.shape[0]
            f, _ = sp.sparse.linalg.cg(A, y, tol=1e-6, maxiter=max_iter)
            Z[:, i] = f

        # Handle numberical errors
        Z[Z < 0] = 0

        # Compute the weight for each instance based on the entropy (eq 11 from the paper)
        probs_l1 = F.normalize(torch.tensor(Z), 1).numpy()
        # print(np.sum(probs_l1[0]))
        probs_l1[probs_l1 < 0] = 0
        entropy = sp.stats.entropy(probs_l1.T)
        weights = 1 - entropy / np.log(no_classes)
        weights = weights / np.max(weights)
        p_labels = np.argmax(probs_l1, 1)
        p_probs = np.amax(probs_l1, 1)
        # print(p_probs)

        p_labels[labeled_idx] = labels[labeled_idx]
        weights[labeled_idx] = 1.0
        labels, labeled_idx = iteration_labels(probs_l1, labels, labeled_idx)
        count_unlabeled = len(labels) - len(labeled_idx)
        #print(count_unlabeled)
        if count_unlabeled<60:
            iterate = False

    return p_labels[support.shape[0]:], weights

def iteration_labels(probs, labels, labeled_idx):
    p_probs = np.amax(probs, 1)
    p_probs = p_probs*-1
    p_labels = np.argmax(probs, 1)
    p_probs[labeled_idx] = 10
    rank = sp.stats.rankdata(p_probs, method='ordinal')
    rank[rank>1] = 0
    indices = np.nonzero(rank)
    #print(labeled_idx.shape)
    labeled_idx = np.concatenate((labeled_idx, indices[0]), axis=0)
    #print("probabilities", p_probs)
    #print("indices: ", indices)
    labels[indices[0]] = p_labels[indices[0]]
    #print("after: ", labels)


    return labels, labeled_idx