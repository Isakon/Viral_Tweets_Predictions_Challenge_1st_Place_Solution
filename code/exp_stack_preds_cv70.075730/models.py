from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
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
        'branched_dnn': BranchedDNN,
        'branched_dnn_embed': BranchedDNNwithEmbeddings,
        
        'stack_linear': LogisticRegression(),
        'stack_mlp_1layer': MLPClassifier,
        'stack_mlp_2layers': MLPClassifier,
        'stack_mlp_3layers': MLPClassifier,
    }

    return model_zoo[model_arch_name]


def dnn_block_layer(dim_in, dim_out, drop_rate, act_func=nn.ReLU()):
    dnn_block = nn.Sequential(
        nn.Linear(dim_in, dim_out),
        nn.BatchNorm1d(dim_out),
        nn.Dropout(p=drop_rate),
        act_func  # nn.ReLU()
    )
    return dnn_block

class BranchedDNN(nn.Module):
    def __init__(self, input_dim: int, output_dim, layers_dims: list,
                 dropouts: list, feat_idxs: dict = {}, ready_feats_divisor=5):
        super(BranchedDNN, self).__init__()
        dropout_head = dropouts[-1]
        dropout_ready_feats = dropouts[-1]
        self.text_idxs = feat_idxs['text_idxs']
        self.media_img_idxs = feat_idxs['media_img_feats']
        self.user_img_idxs = feat_idxs['user_img_feats']
        self.user_des_idxs = feat_idxs['user_des_feats']
        self.cat_idxs = feat_idxs['category_feats']
        self.num_idxs = feat_idxs['numerical_feats']
        n_ready_feats = np.sum([len(feat_idxs[k]) for k in ('text_idxs', 'media_img_feats',
                                                           'user_img_feats','user_des_feats')])
        input_dim_num_cat = len(self.cat_idxs)+len(self.num_idxs)
        if len(layers_dims)>1:
            # in_dims = [input_dim] + layers_dims[:-1]
            in_dims = ([input_dim_num_cat] +
                       layers_dims[:-1])
            out_dims = layers_dims
        else:
            in_dims = [input_dim_num_cat]
            out_dims = layers_dims

        if isinstance(dropouts, float):
            dropouts_adj = [dropouts]*len(in_dims)
        elif isinstance(dropouts, list):
            assert len(dropouts) == len(in_dims), "len(dropouts) wrong"
            dropouts_adj = dropouts

        # self.dnn_hidden1 = dnn_block_layer(in_dims[0], out_dims[0], dropouts_adj[0])
        # if len(layers_dims)>1: self.dnn_hidden2 = dnn_block_layer(in_dims[1], out_dims[1], dropouts_adj[1])
        act_func_ls = [nn.ReLU(), nn.ELU(), nn.SELU()]
        self.dnn_modulelist = nn.ModuleList([dnn_block_layer(dim_in, dim_out, drop, a_func)
                                             for (dim_in, dim_out, drop, a_func) in
                                             zip(in_dims, out_dims, dropouts_adj, act_func_ls)])
        out_dim_ready_feats1 = n_ready_feats // ready_feats_divisor
        out_dim_ready_feats2 = out_dim_ready_feats1 // ready_feats_divisor
        # self.linear1 = nn.Linear(n_ready_feats, out_dim_ready_feats)
        self.dnn_ready_feats = dnn_block_layer(n_ready_feats, out_dim_ready_feats1,
                                               dropout_ready_feats)
        self.joint_dnn = dnn_block_layer(out_dims[-1] + out_dim_ready_feats1,
                                         out_dim_ready_feats2, dropout_ready_feats)
        self.head_input_dim = out_dim_ready_feats2
        # self.head = nn.Linear(in_features=self.head_input_dim, out_features=output_dim)
        self.head = dnn_block_layer(self.head_input_dim, output_dim, dropout_head,
                                    act_func=nn.ELU())

    def forward(self, x):
        # import pdb; pdb.set_trace()
        x_text = x[:, self.text_idxs]
        x_media_img = x[:, self.media_img_idxs]
        x_user_img = x[:, self.user_img_idxs]
        x_user_des = x[:, self.user_des_idxs]
        x_num = x[:, self.num_idxs]
        x_cat = x[:, self.cat_idxs]
        x_ready_feats = self.dnn_ready_feats(torch.cat([x_text, x_media_img,
                                            x_user_img, x_user_des], dim=1))
        # import pdb; pdb.set_trace()
        x = torch.cat([x_num, x_cat], axis=1)
        for i,l in enumerate(self.dnn_modulelist):
            x = self.dnn_modulelist[i](x)
        # x = self.dnn_hidden2(x)
        x = self.joint_dnn(torch.cat([x, x_ready_feats], axis=1))
        x = self.head(x)
        return x

class BranchedDNNwithEmbeddings(nn.Module):
    def __init__(self, input_dim: int, output_dim, layers_dims: list,
                 dropouts: list, feat_idxs: dict = {}, ready_feats_divisor=10,
                 emb_dims=None, emb_dropout=0.1, num_dropout=0.1, media_dropout=0.6,
                 joint_dropout=0.1, head_dropout=0.1,
                 embed_lin_output_size=64,
                 numeric_linear_output_size=64
                 ):
        super(BranchedDNNwithEmbeddings, self).__init__()
        # dropout_head = dropouts[-1]
        # dropout_ready_feats = dropouts[-1]
        self.text_idxs = feat_idxs['text_idxs']
        self.media_img_idxs = feat_idxs['media_img_feats']
        self.user_img_idxs = feat_idxs['user_img_feats']
        self.user_des_idxs = feat_idxs['user_des_feats']
        self.cat_idxs = feat_idxs['category_feats']
        self.cat_topic_idxs = feat_idxs['category_topic_feats']
        self.num_idxs = feat_idxs['numerical_feats']
        # self.numeric_linear_output_size = len(self.num_idxs)*4//5  #  // 2
        # self.embed_linear_output_size = len(self.num_idxs) // 2

        # embeddings for categorical vars
        # emb_dims = get_emb_dims(df, feat_idxs)
        topics_emb_dims = (60, 12)  # 12)  # 60 topics
        self.dim_flattened_topic_embeds = topics_emb_dims[1]*len(self.cat_topic_idxs)
        self.num_embeds = sum([dim1 for (dim0,dim1) in emb_dims]) + self.dim_flattened_topic_embeds
        # self.embed_linear_output_size = int(self.num_embeds * embed_linear_output_size_multiplier)
        self.emb_layers = nn.ModuleList([nn.Embedding(dim1, dim2) for dim1, dim2
                                         in emb_dims])
        self.emb_layer_topic = nn.Embedding(topics_emb_dims[0], topics_emb_dims[1])
        self.cat_embedding_dropout = nn.Dropout(emb_dropout)
        self.cat_embedding_topics_dropout = nn.Dropout2d(emb_dropout)
        self.embed_activation = nn.ELU()

        self.embedding_linear = nn.Linear(self.num_embeds, embed_lin_output_size)
        # self.embedding_linear_topics = nn.Linear(self.num_embeds_topics, self.embed_linear_output_size_topics)
        self.numeric_linear = nn.Linear(len(self.num_idxs), numeric_linear_output_size)
        self.num_dropout = nn.Dropout(num_dropout)
        self.num_activation = nn.ReLU()

        n_ready_feats = np.sum([len(feat_idxs[k]) for k in ('text_idxs', 'media_img_feats',
                                                           'user_img_feats','user_des_feats')])
        input_dim_num_cat = embed_lin_output_size + numeric_linear_output_size
        if len(layers_dims)>1:
            # in_dims = [input_dim] + layers_dims[:-1]
            in_numcat_dims = ([input_dim_num_cat] +
                       layers_dims[:-1])
            out_numcat_dims = layers_dims
        else:
            in_numcat_dims = [input_dim_num_cat]
            out_numcat_dims = layers_dims

        if isinstance(dropouts, float):
            dropouts_adj = [dropouts]*len(in_numcat_dims)
        elif isinstance(dropouts, list):
            assert len(dropouts) == len(in_numcat_dims), "len(dropouts) wrong"
            dropouts_adj = dropouts

        # self.dnn_hidden1 = dnn_block_layer(in_dims[0], out_dims[0], dropouts_adj[0])
        # if len(layers_dims)>1: self.dnn_hidden2 = dnn_block_layer(in_dims[1], out_dims[1], dropouts_adj[1])
        act_func_ls = [nn.ReLU(), nn.ELU(), nn.SELU()]
        self.numcat_dnn_modulelist = nn.ModuleList([dnn_block_layer(dim_in, dim_out, drop, a_func)
                                             for (dim_in, dim_out, drop, a_func) in
                                             zip(in_numcat_dims, out_numcat_dims, dropouts_adj, act_func_ls)])
        out_dim_ready_feats1 = int(n_ready_feats / ready_feats_divisor)
        out_dim_joint_feats = int(out_dim_ready_feats1 / ready_feats_divisor)
        # self.linear1 = nn.Linear(n_ready_feats, out_dim_ready_feats)
        self.dnn_ready_feats = dnn_block_layer(n_ready_feats, out_dim_ready_feats1,
                                               media_dropout)
        self.joint_dnn = dnn_block_layer(out_numcat_dims[-1] + out_dim_ready_feats1,
                                         out_dim_joint_feats, joint_dropout)
        self.head_input_dim = out_dim_joint_feats
        # self.head = nn.Linear(in_features=self.head_input_dim, out_features=output_dim)
        self.head = dnn_block_layer(self.head_input_dim, output_dim, head_dropout,
                                    act_func=nn.ELU())

    def forward(self, x):
        # import pdb; pdb.set_trace()
        bs = x.size()[0]
        x_text = x[:, self.text_idxs]
        x_media_img = x[:, self.media_img_idxs]
        x_user_img = x[:, self.user_img_idxs]
        x_user_des = x[:, self.user_des_idxs]
        x_num = x[:, self.num_idxs]
        x_cat = x[:, self.cat_idxs].type(torch.long)
        x_cat_topics = x[:, self.cat_topic_idxs].type(torch.long)

        embedding_out = []
        for i in range(len(self.cat_idxs)):
            x_cat_i_emb = self.emb_layers[i](x_cat[:, i])
            # import pdb;
            # pdb.set_trace()
            x_cat_i_emb = torch.squeeze(self.cat_embedding_dropout(torch.unsqueeze(x_cat_i_emb, 0)), 0)
            embedding_out.append(x_cat_i_emb)

        x_cat_topics_emb = self.emb_layer_topic(x_cat_topics).view(bs, self.dim_flattened_topic_embeds)
        embedding_out_topics = torch.squeeze(self.cat_embedding_dropout(torch.unsqueeze(x_cat_topics_emb,1)),1)
        embedding_out = torch.cat([torch.cat(embedding_out, 1), embedding_out_topics], 1)
        # import pdb;
        # pdb.set_trace()
        embedding_out = self.embed_activation(
            self.cat_embedding_dropout(
                self.embedding_linear(embedding_out)
            )
        )
        # import pdb;
        # pdb.set_trace()
        x_num = self.numeric_linear(x_num)
        x_num = self.num_dropout(x_num)
        x_num = self.num_activation(x_num)
        x_ready_feats = self.dnn_ready_feats(torch.cat([x_text, x_media_img,
                                            x_user_img, x_user_des], dim=1))
        # import pdb; pdb.set_trace()
        x = torch.cat([x_num, embedding_out], axis=1)
        for i, numcat_layer in enumerate(self.numcat_dnn_modulelist):
            x = numcat_layer(x)
        # x = self.dnn_hidden2(x)
        x = self.joint_dnn(torch.cat([x, x_ready_feats], axis=1))
        x = self.head(x)
        return x


class SimpleDNN(nn.Module):
    def __init__(self, input_dim: int, output_dim, layers_dims: list,
                 dropouts: list, feat_idxs = None):
        super(SimpleDNN, self).__init__()
        self.len_layers_dims = len(layers_dims)
        if self.len_layers_dims>1:
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
        if self.len_layers_dims>1: self.dnn_hidden2 = dnn_block_layer(in_dims[1], out_dims[1], dropouts_adj[1])
        self.head = nn.Linear(in_features=out_dims[-1], out_features=output_dim)

    def forward(self, x):
        # import pdb; pdb.set_trace()
        x = self.dnn_hidden1(x)
        if self.len_layers_dims>1:
            x = self.dnn_hidden2(x)
        x = self.head(x)
        return x

class OldBranchedDNNwithEmbeddings(nn.Module):
    def __init__(self, input_dim: int, output_dim, layers_dims: list,
                 dropouts: list, feat_idxs: dict = {}, ready_feats_divisor=10,
                 emb_dims=None, emb_dropout=0.3, num_dropout=0.3,
                 embed_lin_output_size=64,
                 numeric_linear_output_size=64
                 ):
        super(BranchedDNNwithEmbeddings, self).__init__()
        dropout_head = dropouts[-1]
        dropout_ready_feats = dropouts[-1]
        self.text_idxs = feat_idxs['text_idxs']
        self.media_img_idxs = feat_idxs['media_img_feats']
        self.user_img_idxs = feat_idxs['user_img_feats']
        self.user_des_idxs = feat_idxs['user_des_feats']
        self.cat_idxs = feat_idxs['category_feats']
        self.cat_topic_idxs = feat_idxs['category_topic_feats']
        self.num_idxs = feat_idxs['numerical_feats']
        # self.numeric_linear_output_size = len(self.num_idxs)*4//5  #  // 2
        # self.embed_linear_output_size = len(self.num_idxs) // 2

        # embeddings for categorical vars
        # emb_dims = get_emb_dims(df, feat_idxs)
        topics_emb_dims = (60, 12)  # 12)  # 60 topics
        self.dim_flattened_topic_embeds = topics_emb_dims[1]*len(self.cat_topic_idxs)
        self.num_embeds = sum([dim1 for (dim0,dim1) in emb_dims]) + self.dim_flattened_topic_embeds
        # self.embed_linear_output_size = int(self.num_embeds * embed_linear_output_size_multiplier)
        self.emb_layers = nn.ModuleList([nn.Embedding(dim1, dim2) for dim1, dim2
                                         in emb_dims])
        self.emb_layer_topic = nn.Embedding(topics_emb_dims[0], topics_emb_dims[1])
        self.cat_embedding_dropout = nn.Dropout(emb_dropout)
        self.cat_embedding_topics_dropout = nn.Dropout2d(emb_dropout)
        self.embed_activation = nn.ELU()

        self.embedding_linear = nn.Linear(self.num_embeds, embed_lin_output_size)
        # self.embedding_linear_topics = nn.Linear(self.num_embeds_topics, self.embed_linear_output_size_topics)
        self.numeric_linear = nn.Linear(len(self.num_idxs), numeric_linear_output_size)
        self.num_dropout = nn.Dropout(num_dropout)
        self.num_activation = nn.ReLU()

        n_ready_feats = np.sum([len(feat_idxs[k]) for k in ('text_idxs', 'media_img_feats',
                                                           'user_img_feats','user_des_feats')])
        input_dim_num_cat = embed_lin_output_size + numeric_linear_output_size
        if len(layers_dims)>1:
            # in_dims = [input_dim] + layers_dims[:-1]
            in_numcat_dims = ([input_dim_num_cat] +
                       layers_dims[:-1])
            out_numcat_dims = layers_dims
        else:
            in_numcat_dims = [input_dim_num_cat]
            out_numcat_dims = layers_dims

        if isinstance(dropouts, float):
            dropouts_adj = [dropouts]*len(in_numcat_dims)
        elif isinstance(dropouts, list):
            assert len(dropouts) == len(in_numcat_dims), "len(dropouts) wrong"
            dropouts_adj = dropouts

        # self.dnn_hidden1 = dnn_block_layer(in_dims[0], out_dims[0], dropouts_adj[0])
        # if len(layers_dims)>1: self.dnn_hidden2 = dnn_block_layer(in_dims[1], out_dims[1], dropouts_adj[1])
        act_func_ls = [nn.ReLU(), nn.ELU(), nn.SELU()]
        self.numcat_dnn_modulelist = nn.ModuleList([dnn_block_layer(dim_in, dim_out, drop, a_func)
                                             for (dim_in, dim_out, drop, a_func) in
                                             zip(in_numcat_dims, out_numcat_dims, dropouts_adj, act_func_ls)])
        out_dim_ready_feats1 = int(n_ready_feats / ready_feats_divisor)
        out_dim_joint_feats = int(out_dim_ready_feats1 / ready_feats_divisor)
        # self.linear1 = nn.Linear(n_ready_feats, out_dim_ready_feats)
        self.dnn_ready_feats = dnn_block_layer(n_ready_feats, out_dim_ready_feats1,
                                               dropout_ready_feats)
        self.joint_dnn = dnn_block_layer(out_numcat_dims[-1] + out_dim_ready_feats1,
                                         out_dim_joint_feats, dropout_ready_feats)
        self.head_input_dim = out_dim_joint_feats
        # self.head = nn.Linear(in_features=self.head_input_dim, out_features=output_dim)
        self.head = dnn_block_layer(self.head_input_dim, output_dim, dropout_head,
                                    act_func=nn.ELU())

    def forward(self, x):
        # import pdb; pdb.set_trace()
        bs = x.size()[0]
        x_text = x[:, self.text_idxs]
        x_media_img = x[:, self.media_img_idxs]
        x_user_img = x[:, self.user_img_idxs]
        x_user_des = x[:, self.user_des_idxs]
        x_num = x[:, self.num_idxs]
        x_cat = x[:, self.cat_idxs].type(torch.long)
        x_cat_topics = x[:, self.cat_topic_idxs].type(torch.long)

        embedding_out = []
        for i in range(len(self.cat_idxs)):
            x_cat_i_emb = self.emb_layers[i](x_cat[:, i])
            # import pdb;
            # pdb.set_trace()
            x_cat_i_emb = torch.squeeze(self.cat_embedding_dropout(torch.unsqueeze(x_cat_i_emb, 0)), 0)
            embedding_out.append(x_cat_i_emb)

        x_cat_topics_emb = self.emb_layer_topic(x_cat_topics).view(bs, self.dim_flattened_topic_embeds)
        embedding_out_topics = torch.squeeze(self.cat_embedding_dropout(torch.unsqueeze(x_cat_topics_emb,1)),1)
        embedding_out = torch.cat([torch.cat(embedding_out, 1), embedding_out_topics], 1)
        # import pdb;
        # pdb.set_trace()
        embedding_out = self.embed_activation(
            self.cat_embedding_dropout(
                self.embedding_linear(embedding_out)
            )
        )
        # import pdb;
        # pdb.set_trace()
        x_num = self.numeric_linear(x_num)
        x_num = self.num_dropout(x_num)
        x_num = self.num_activation(x_num)
        x_ready_feats = self.dnn_ready_feats(torch.cat([x_text, x_media_img,
                                            x_user_img, x_user_des], dim=1))
        # import pdb; pdb.set_trace()
        x = torch.cat([x_num, embedding_out], axis=1)
        for i, numcat_layer in enumerate(self.numcat_dnn_modulelist):
            x = numcat_layer(x)
        # x = self.dnn_hidden2(x)
        x = self.joint_dnn(torch.cat([x, x_ready_feats], axis=1))
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



def extract_features(model, layer_name, loader, feat_name_prefix='extracted_'):
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
