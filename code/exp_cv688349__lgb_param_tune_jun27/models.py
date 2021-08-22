from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
import numpy as np
import torch
import torch.nn as nn

def get_model_class(model_arch_name):
    model_zoo = {
        # 'mlp': MLPClassifier(solver='adam', alpha=1e-5,
        #                      hidden_layer_sizes=(512),  # (2048, 512),
        #                      # learning_rate=5e-4,
        #                      max_iter=cfg.mlp.n_epochs,
        #                      batch_size=1000,
        #                      validation_fraction=0.03,  # 0.05,
        #                      early_stopping=True,
        #                      verbose=True, random_state=cfg.seed_other),
        # 'randomforest': RandomForestClassifier(**cfg.randomforest),
        # 'svm': SVC(probability=True, verbose=True),
        # 'naive_bayes': BernoulliNB(),
        # 'logistic_regression': LogisticRegression(solver='sag',
        #                                           random_state=cfg.seed_other,
        #                                           multi_class='multinomial',
        #                                           verbose=1),
        'simple_dnn': SimpleDNN,
        
    }

    return model_zoo[model_arch_name]


def dnn_block_layer(dim_in, dim_out, drop_rate):
    dnn_block = nn.Sequential(
        nn.Linear(dim_in, dim_out),
        nn.BatchNorm1d(dim_out),
        nn.Dropout(p=drop_rate),
        nn.ReLU()
    )
    return dnn_block

class SimpleDNN(nn.Module):
    def __init__(self, input_dim: int, output_dim, layers_dims: list, dropouts: list):
        super(SimpleDNN, self).__init__()
        if len(layers_dims)>1:
            in_dims = [input_dim] + layers_dims[:-1]
            out_dims = layers_dims
        else:
            in_dims = [input_dim]
            out_dims = layers_dims

        if isinstance(dropouts, float):
            dropouts_adj = [dropouts]*len(in_dims)
        elif isinstance(dropouts, list):
            assert len(dropouts) == len(in_dims), "len(dropouts) wrong"
            dropouts_adj = dropouts
        self.dnn_hidden1 = dnn_block_layer(in_dims[0], out_dims[0], dropouts_adj[0])
        if len(layers_dims)>1: self.dnn_hidden2 = dnn_block_layer(in_dims[1], out_dims[1], dropouts_adj[1])
        self.head = nn.Linear(in_features=out_dims[-1], out_features=output_dim)

    def forward(self, x):
        # import pdb; pdb.set_trace()
        x = self.dnn_hidden1(x)
        # x = self.dnn_hidden2(x)
        x = self.head(x)
        return x


class SimpleDNN_ModuleList(nn.Module):
    def __init__(self, input_dim: int, output_dim, layers_dims: list, dropouts: list):
        super(SimpleDNN, self).__init__()
        if len(layers_dims)>1:
            in_dims = [input_dim] + layers_dims[:-1]
            out_dims = layers_dims
        else:
            in_dims = [input_dim]
            out_dims = layers_dims

        if isinstance(dropouts, float):
            dropouts_adj = [dropouts]*len(in_dims)
        elif isinstance(dropouts, list):
            assert len(dropouts) == len(in_dims), "len(dropouts) wrong"
            dropouts_adj = dropouts

        self.dnn_modulelist = nn.ModuleList([dnn_block_layer(dim_in, dim_out, drop)
                                             for (dim_in, dim_out, drop) in
                                             zip(in_dims, out_dims, dropouts_adj)])
        self.head = nn.Linear(in_features=out_dims[-1], out_features=output_dim)

    def forward(self, x):
        # import pdb; pdb.set_trace()
        for i,l in enumerate(self.dnn_modulelist):
            x = self.dnn_modulelist[i](x)
        x = self.head(x)
        return x


class SimpleDNNwithEmbeddings(nn.Module):
    # https://yashuseth.blog/2018/07/22/pytorch-neural-network-for-tabular-data-with-categorical-embeddings/
    def __init__(self, input_dim: int, layers_dims: list, dropouts: list):
        super(SimpleDNNwithEmbeddings, self).__init__()
        if len(layers_dims)>1:
            in_dims = [input_dim] + layers_dims[:-1]
            out_dims = layers_dims
        else:
            in_dims = [input_dim]
            out_dims = layers_dims

        if isinstance(dropouts, float):
            dropouts_adj = [dropouts]*len(in_dims)
        elif isinstance(dropouts, list):
            assert len(dropouts) == len(in_dims), "len(dropouts) wrong"
            dropouts_adj = dropouts

        self.dnn_modulelist = nn.ModuleList([dnn_block_layer(dim_in, dim_out, drop)
                                             for (dim_in, dim_out, drop) in
                                             zip(in_dims, out_dims, dropouts_adj)])
        self.head = nn.Linear(in_features=out_dims[-1], out_features=5)

    def forward(self, x):
        # import pdb; pdb.set_trace()
        for i,l in enumerate(self.dnn_modulelist):
            x = self.dnn_modulelist[i](x)
        x = self.head(x)
        return x


def extract_features(model, df, layer_name, loader, feat_name_prefix='extracted_'):
    # Source: https://www.kaggle.com/appian/implementing-image-feature-extraction-in-pytorch
    model = model.cuda()
    model.eval()
    for name, module in model.named_modules():
        print(f"{name}:\t{module}")

    # register hook to access to features in forward pass
    features_ls = []

    # def get_activation():
    def hook(module, input, output):
        #N, C, H, W = output.shape
        # bs, n_classes = output.shape
        # output = output.reshape(N, C, -1)
        # output = output.reshape(bs, n_classes)
        #features.append(output.mean(dim=2).cpu().detach().numpy())
        # import pdb; pdb.set_trace()
        features_ls.append(output.detach().cpu().numpy())
    # return hook

    handle = model._modules.get(layer_name).register_forward_hook(hook)

    # dataset = Dataset(df, size)
    # loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=4)

    # for i_batch, inputs in tqdm(enumerate(loader), total=len(loader)):
    for inputs in loader:
        _ = model(inputs[0].cuda())

    # import pdb; pdb.set_trace()
    features = np.concatenate(features_ls)
    handle.remove()
    del model

    return features



# # https://discuss.pytorch.org/t/how-to-extract-intermediate-feature-maps-from-u-net/27084/8
# activation = {}

def get_activation(name):
    # # https://discuss.pytorch.org/t/how-to-extract-intermediate-feature-maps-from-u-net/27084/8
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

#
# model.dnn_modellist.register_forward_hook(get_activation('block1'))
#
# x = torch.randn(1, 1, 224, 224)
# output = model(x)
# print(activation['block1'].shape)
# print(activation['block2'].shape)
# print(activation['block3'].shape)


# class MultiBranchDNN(nn.Module):
#     def __init__(self, input_dim: int, tabular_layers_dims: list, dropouts: list):
#         super(SimpleDNN, self).__init__()
#         if len(tabular_layers_dims)>1:
#             in_dims = [input_dim] + tabular_layers_dims[:-1]
#             out_dims = tabular_layers_dims
#         else:
#             in_dims = [input_dim]
#             out_dims = tabular_layers_dims
#
#         if isinstance(dropouts, float):
#             dropouts_adj = [dropouts]*len(in_dims)
#         elif isinstance(dropouts, list):
#             assert len(dropouts) == len(in_dims), "len(dropouts) wrong"
#             dropouts_adj = dropouts
#
#         self.dnn_modellist = nn.ModuleList([dnn_block_layer(dim_in, dim_out, drop)
#                                    for (dim_in, dim_out, drop) in
#                                    zip(in_dims, out_dims, dropouts_adj)])
#         self.head = nn.Linear(in_features=out_dims[-1], out_features=5)
#
#     def forward(self, x):
#         for i,l in enumerate(self.dnn_modellist):
#             x = self.dnn_modellist[i](x)
#         x = self.head(x)
#         return x
