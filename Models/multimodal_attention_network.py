import torch
import torch.nn.functional as F
import torch.nn as nn
from Models.base_man import BaseMan
import torch_utils
from setting_keywords import KeyWordSettings
from thirdparty.head_cnns import PACRRPlaneMaxPooling
import time
import layers
from Models.base_man import AttentionType


class MultiModalAttentionNetwork(BaseMan):
    """
    Multimodal attention network
    Examples:
        >>> model = MultiModalAttentionNetwork()
        >>> model.params['fixed_text_left'] = 50
        >>> model.params['fixed_text_right'] = 1000
        >>> model.params['dropout'] = 0.1
        >>> model.params['context_window'] = 9  # size of context windows
        >>> model.params['filters'] = 5  # number of filters when applying a conv
        >>> model.params['max_ngram'] = 1
        >>> model.params["norm_type"] = "l2"  # to save mem
        >>> model.params["beta"] = 1
        >>> model.params["decomposition_type"] = 0  # type of decompsitional
    """
    def __init__(self, params):
        super(BaseMan, self).__init__()
        self._params = params
        self.src_word_emb = self._make_default_embedding_layer(params)
        # n_position = max(self._params["fixed_text_left"], self._params["fixed_text_right"])  # checked
        self.fixed_length_left = self._params["fixed_length_left"]
        self.fixed_length_right = self._params["fixed_length_right"]
        d_word_vec = self._params['embedding_output_dim']
        dropout = self._params["dropout"]
        self.input_channels = 4
        self.attention_type = self._params["attention_type"]
        self.use_average_dcompositional_att = self._params["use_average_dcompositional_att"]

        ################################################################################################
        self.q_convs = nn.ModuleList()
        self.d_convs = nn.ModuleList()
        self.q_context_convs, self.d_context_convs = nn.ModuleList(), nn.ModuleList()
        self.max_ngram = self._params["max_ngram"]
        for i in range(self.max_ngram):
            conv = nn.Sequential(
                layers.Permute([0, 2, 1]),
                nn.ConstantPad1d((0, i), 0),
                nn.Conv1d(
                    in_channels=self._params["embedding_output_dim"],
                    out_channels=self._params['filters'],
                    kernel_size=i + 1
                ),
                nn.Tanh()
            )
            self.q_convs.append(conv)
            self.d_convs.append(conv)

            conv2 = nn.Sequential(
                layers.Permute([0, 2, 1]),
                nn.ConstantPad1d((0, i), 0),
                nn.Conv1d(
                    in_channels = self._params["elmo_vec_size"],
                    out_channels = self._params['filters'],
                    kernel_size = i + 1
                ),
                nn.Tanh()
            )
            self.q_context_convs.append(conv2)
            self.d_context_convs.append(conv2)

        ################################################################################################
        # visual component
        self.use_visual = self._params["use_visual"]
        if self.use_visual:
            num_ftrs = self._params["visual_feature_size"]
            self.last_visual_size = 300  # same as word embeddings dim
            self.image_fc1 = nn.Linear(num_ftrs, self.last_visual_size)
            self.full_left_images_tensor = self._params["full_left_images_tensor"]
            self.full_right_images_tensor = self._params["full_right_images_tensor"]
        ################################################################################################
        # contextextualized component
        self.left_elmo_tensor = self._params["left_elmo_tensor"]
        self.right_elmo_tensor = self._params["right_elmo_tensor"]

        ################################################################################################
        if self.attention_type in [AttentionType.UsingBilinearOnly, AttentionType.UsingBilinearDissim]:
            self.linearQ = nn.Linear(self._params['filters'], 1, bias=False)
            self.linearD = nn.Linear(self._params['filters'], 1, bias=False)
            self.linearQD = nn.Linear(self._params['filters'], self._params['filters'], bias=False)
        ################################################################################################
        self.head_conv_layers = nn.ModuleList()
        if self._params["head_cnn_type"] == "pacrr_plane":
            self.head_conv_layers.append(PACRRPlaneMaxPooling(num_conv_layers = self._params["conv_layers"],
                 input_channels = self.input_channels, filters_count = self._params["filters_count_pacrr"],
                                                              ns = self._params["n_s"]))
        # assert len(self.head_conv_layers) == self.max_ngram
        ################################################################################################
        factors = self.max_ngram * self.head_conv_layers[0].last_in_channels * self.head_conv_layers[0].L_new * self.head_conv_layers[0].R_new
        if self.use_visual: factors += 1
        self.linear = nn.Sequential(
            # layers.Flatten(dim = 1),
            nn.Linear(factors, 128),
            nn.ReLU(), nn.Linear(128, 64),
            nn.ReLU(), nn.Linear(64, 1))

    def forward(self, query: torch.Tensor, document: torch.Tensor, verbose = False, **kargs):
        """Forward. of integer query tensor and document tensor """
        max_left_len, max_right_len = query.size(1), document.size(1)
        # Process left & right input.
        # https://github.com/AdeDZY/K-NRM/blob/master/knrm/model/model_base.py#L96
        tensor_mask = torch_utils.create_mask_tensor(query, document, threshold = 1)
        doc_mask = (document > 0).float()
        query_mask = (query > 0).float()  # B, L
        embed_query = self.src_word_emb(query.long())  # (B, L, D)
        embed_doc = self.src_word_emb(document.long())  # (B, R, D)
        # normalizing vectors
        embed_query = F.normalize(embed_query, p = 2, dim = -1)
        embed_doc = F.normalize(embed_doc, p = 2, dim = -1)
        ################################# For Contextualized Representation using ELMO #############################
        query_ids = kargs[KeyWordSettings.QueryIDs]  # (B, )
        doc_ids = kargs[KeyWordSettings.DocIDs]  # (B, )
        assert query_ids.shape == doc_ids.shape
        use_cuda = kargs[KeyWordSettings.UseCuda]
        query_char_repr = self.left_elmo_tensor[query_ids]
        doc_char_repr = self.right_elmo_tensor[doc_ids]
        # I have to load to gpu at this step because left_tensor is too large to load to GPU
        query_char_repr = torch_utils.gpu(query_char_repr, use_cuda)  # (B, L, D1)
        doc_char_repr = torch_utils.gpu(doc_char_repr, use_cuda)  # (B, R, D1)
        assert query_char_repr.size(1) == embed_query.size(1)
        assert doc_char_repr.size(1) == embed_doc.size(1)
        ###############################################################################################
        q_convs, d_convs = [], []
        q_ctx_convs, d_ctx_convs = [], []
        for q_conv, d_conv, \
            q_context_conv, d_context_conv in zip(self.q_convs, self.d_convs, self.q_context_convs, self.d_context_convs):
            q_out = q_conv(embed_query).transpose(1, 2)  # to shape (B, D, L) => (B, F, L) => (B, L, F)
            d_out = d_conv(embed_doc).transpose(1, 2)  # to shape (B, D, R) => (B, F, R) => (B, R, F)
            q_out = F.normalize(q_out, p = 2, dim = -1)  # good stuff for relevance matching
            d_out = F.normalize(d_out, p = 2, dim = -1)
            q_convs.append(q_out)
            d_convs.append(d_out)

            q_ctx_out = q_context_conv(query_char_repr).transpose(1, 2)  # B, L, F
            d_ctx_out = d_context_conv(doc_char_repr).transpose(1, 2)  # B, R, F
            q_ctx_out = F.normalize(q_ctx_out, p=2, dim=-1)
            d_ctx_out = F.normalize(d_ctx_out, p=2, dim=-1)
            q_ctx_convs.append(q_ctx_out)
            d_ctx_convs.append(d_ctx_out)

        output_phis = []
        for idx in range(self.max_ngram):
            query_local_context = q_ctx_convs[idx]  # (B, L, D)
            doc_local_context = d_ctx_convs[idx]  # (B, R, D)
            sim_mat = self._get_sim_matrix(q_convs[idx], d_convs[idx])
            sim_mat = sim_mat * tensor_mask

            if self.attention_type == AttentionType.UsingDotProductOnly:
                # using sim_mat, context_mat, sim_mat - context_mat, sim_mat * context_mat
                # [S, L, S - L, S * L]
                context_aware_mat = self._get_sim_matrix(query_local_context, doc_local_context) * tensor_mask
                tensors = torch.stack([sim_mat,
                                       context_aware_mat,
                                       sim_mat - context_aware_mat,
                                       sim_mat * context_aware_mat], dim=-1)  # B, L, R, C
            elif self.attention_type == AttentionType.UsingDotProductDisim:
                # using sim_mat, context_mat, sim_mat - context_mat, dissimilarity * sim_mat
                # [S, L, S - L, S * D]
                context_aware_mat = self._get_sim_matrix(query_local_context, doc_local_context) * tensor_mask
                dissimilarity = self._get_disimilarity_mat(query_local_context, doc_local_context, tensor_mask, self.use_average_dcompositional_att) * tensor_mask
                tensors = torch.stack([sim_mat,
                                       context_aware_mat,
                                       sim_mat - context_aware_mat,
                                       sim_mat * dissimilarity], dim=-1)   # B, L, R, C
            elif self.attention_type == AttentionType.UsingBilinearOnly:
                # [S, B, S - B, S * B]
                bilinear = self._get_bilinear_attention(query_local_context, doc_local_context) * tensor_mask
                tensors = torch.stack([sim_mat,
                                       bilinear,
                                       sim_mat - bilinear,
                                       bilinear * sim_mat], dim=-1)  # B, L, R, C
            elif self.attention_type == AttentionType.UsingBilinearDissim:
                # [S, B, S - B, S * D]
                bilinear = self._get_bilinear_attention(query_local_context, doc_local_context) * tensor_mask
                dissimilarity = self._get_disimilarity_mat(query_local_context, doc_local_context, tensor_mask, self.use_average_dcompositional_att) * tensor_mask
                tensors = torch.stack([sim_mat,
                                       bilinear,
                                       sim_mat - bilinear,
                                       dissimilarity * sim_mat], dim=-1)  # B, L, R, C

            tensors = tensors.permute(0, 3, 1, 2)  # (B, C, L, R)
            phi = torch.flatten(self.head_conv_layers[0](tensors), start_dim = 1)
            output_phis.append(phi)

        phi = torch.cat(output_phis, dim = -1)  # (B, x)
        if self.use_visual:
            # a list of size B, where each element is a list of image tensors
            t1 = time.time()
            query_images_indices = kargs[KeyWordSettings.QueryImagesIndices]
            B1, n1, M1 = query_images_indices.shape  # expected shape
            assert n1 == 1
            query_images = self.full_left_images_tensor[query_images_indices.flatten().long()]  # B1 * n1 * M1, VD
            doc_imgs_indices = kargs[KeyWordSettings.DocImagesIndices]  # (B, n, M2, VD) or (B, M2, VD)
            B, n, M2 = doc_imgs_indices.shape  # expected shape
            images_mask = torch_utils.create_mask_tensor_image(query_images_indices, doc_imgs_indices)  # (B, n, M1, M2)
            doc_images = self.full_right_images_tensor[doc_imgs_indices.flatten().long()]  # B * n * M2, VD

            left_feats = self.image_fc1(query_images)  # (B * n1 * M1, H) we don't want visual_cnn on 30 duplicated queries images (not wise)
            right_feats = self.image_fc1(doc_images)  # (B * n * M2, H)
            left_feats = left_feats.view(B1, M1, self.last_visual_size)
            if B1 == 1: left_feats = left_feats.expand(B, M1, self.last_visual_size)  # during testing
            right_feats = right_feats.view(B, n * M2, self.last_visual_size)
            right_feats = F.normalize(right_feats, p=2, dim=-1)
            left_feats = F.normalize(left_feats, p=2, dim=-1)
            scores = torch.bmm(left_feats, right_feats.permute(0, 2, 1))  # (B, M1, n * M2)
            scores = scores.view(B, M1, n, M2).permute(0, 2, 1, 3)  # (B, n, M1, M2)
            # masking
            assert scores.size() == images_mask.size(), (scores.size(), images_mask.size())
            scores = scores * images_mask
            scores = scores.view(B * n, M1, M2)
            visual_scores, _ = torch.flatten(scores, start_dim = 1).max(-1)
            visual_scores = visual_scores.unsqueeze(-1)  # (B * n, 1)
            phi = torch.cat([phi, visual_scores], dim = -1)
            t2 = time.time()
            # print("Running time of CNN in forward: ", (t2 - t1), "seconds")
        out = self.linear(phi)
        if verbose:
            print("out: ", out.squeeze())
            # print("After dense and tanh: ", out)
        if KeyWordSettings.OutputRankingKey in kargs and kargs[KeyWordSettings.OutputRankingKey] and self.use_visual:
            return torch.cat([out, torch.flatten(scores, start_dim = 1)], dim = -1)  # for error analysis (B, 2)
        return out.squeeze()
