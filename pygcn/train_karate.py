from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

from utils import load_data, accuracy, encode_onehot
from models2 import GCN

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden1', type=int, default=200,
                    help='Number of hidden units.')
parser.add_argument('--hidden2', type=int, default=50,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
adj, features, labels, idx_train, idx_val, idx_test = load_data("../data/karate/","karate")
# adj, features, labels, idx_train, idx_val, idx_test = load_data("../data/cora/","cora")

# Model and optimizer
model = GCN(nfeat=features.shape[1],
            nhid1=args.hidden1,
	    nhid2=args.hidden2,
            nclass=4,
            dropout=args.dropout)

optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()


criterion = nn.CrossEntropyLoss()
indices = torch.LongTensor([0, 16,18,24])

def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    #output2 = PCA(output.detach().numpy())[0]
    #ax = plt.scatter(output2[:,0],output2[:,1])
    #plt.show()
    loss = criterion(torch.index_select(output, 0, indices),torch.LongTensor([0,1,2,3]))
    loss.backward()
    optimizer.step()

    if not args.fastmode:
        model.eval()
        output = model(features, adj)

    print('Epoch: {:04d}'.format(epoch+1),
          'time: {:.4f}s'.format(time.time() - t))

for epoch in range(args.epochs):
    train(epoch)

def PCA(data, dims_rescaled_data=2):
    """
    returns: data transformed in 2 dims/columns + regenerated original data
    pass in: data as 2D NumPy array
    """
    import numpy as NP
    from scipy import linalg as LA
    m, n = data.shape
    # mean center the data
    data -= data.mean(axis=0)
    # calculate the covariance matrix
    R = NP.cov(data, rowvar=False)
    # calculate eigenvectors & eigenvalues of the covariance matrix
    # use 'eigh' rather than 'eig' since R is symmetric, 
    # the performance gain is substantial
    evals, evecs = LA.eigh(R)
    # sort eigenvalue in decreasing order
    idx = NP.argsort(evals)[::-1]
    evecs = evecs[:,idx]
    # sort eigenvectors according to same index
    evals = evals[idx]
    # select the first n eigenvectors (n is desired dimension
    # of rescaled data array, or dims_rescaled_data)
    evecs = evecs[:, :dims_rescaled_data]
    # carry out the transformation on the data using eigenvectors
    # and return the re-scaled data, eigenvalues, and eigenvectors
    return NP.dot(evecs.T, data.T).T, evals, evecs

output = model(features, adj)
print (output)
output = PCA(output.detach().numpy())[0]
print (output)

# draw
import matplotlib.pyplot as plt
plt.scatter(output[:,0],output[:,1])
plt.show()

# classify 
community = dict()
for index,i in enumerate(output):
    temp = [np.linalg.norm(i-output[0,:]),
             np.linalg.norm(i-output[16,:]),
             np.linalg.norm(i-output[18,:]),
             np.linalg.norm(i-output[24,:])]
    belongs = temp.index(min(temp))
    if belongs not in community.keys():
    	community[belongs]=[]
	community[belongs].append(index)
    else:
	community[belongs].append(index)
print (community)
#for i,_ in enumerate(output[:,0]):
#    plt.annotate(i,(output[:,0][i],output[:,1][i]))
