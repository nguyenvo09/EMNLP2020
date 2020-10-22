import torch
import torch.nn.functional as F
import torch.nn as nn
from Models.base_model import BaseModel
import torch_utils
import numpy as np
from setting_keywords import KeyWordSettings
import math
from thirdparty.head_cnns import PACRRPlaneMaxPooling
import time
import layers
from enum import IntEnum


class AttentionType(IntEnum):
    UsingDotProductOnly = 1
    UsingDotProductDisim = 2
    UsingBilinearOnly = 3
    UsingBilinearDissim = 4


class BaseMan(BaseModel):

    def __init__(self, params):
        super(BaseModel, self).__init__()
        self._params = params
        self.src_word_emb = self._make_default_embedding_layer(params)
        self.fixed_length_left = self._params["fixed_length_left"]
        self.fixed_length_right = self._params["fixed_length_right"]
        d_word_vec = self._params['embedding_output_dim']
        dropout = self._params["dropout"]
        self.input_channels = 4
        self.attention_type = self._params["attention_type"]
        self.use_average_dcompositional_att = self._params["use_average_dcompositional_att"]
        self.q_convs = nn.ModuleList()
        self.d_convs = nn.ModuleList()
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
        if self.attention_type in [AttentionType.UsingBilinearOnly, AttentionType.UsingBilinearDissim]:
            self.linearQ = nn.Linear(self._params['filters'], 1, bias=False)
            self.linearD = nn.Linear(self._params['filters'], 1, bias=False)
            self.linearQD = nn.Linear(self._params['filters'], self._params['filters'], bias=False)

        ################################################################################################
        self.context_window = self._params["context_window"]
        # self.nb_conv_matching_layers = self._params["nb_conv_matching_layers"]
        ################################################################################################
        self.head_conv_layers = nn.ModuleList()
        # for idx in range(self.max_ngram):
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
        # query, document = query.float(), document.float()
        # https://github.com/AdeDZY/K-NRM/blob/master/knrm/model/model_base.py#L96
        tensor_mask = torch_utils.create_mask_tensor(query, document, threshold = 1)
        doc_mask = (document > 0).float()
        query_mask = (query > 0).float()  # B, L
        # query_pos = kargs[KeyWordSettings.QueryPositions]
        # doc_pos = kargs[KeyWordSettings.DocPositions]

        embed_query = self.src_word_emb(query.long())  # (B, L, D)
        embed_doc = self.src_word_emb(document.long())  # (B, R, D)
        # normalizing vectors
        embed_query = F.normalize(embed_query, p = 2, dim = -1)
        embed_doc = F.normalize(embed_doc, p = 2, dim = -1)
        q_convs, d_convs = [], []
        for q_conv, d_conv in zip(self.q_convs, self.d_convs):
            q_out = q_conv(embed_query).transpose(1, 2)  # to shape (B, D, L) => (B, F, L) => (B, L, F)
            d_out = d_conv(embed_doc).transpose(1, 2)  # to shape (B, D, R) => (B, F, R) => (B, R, F)
            q_out = F.normalize(q_out, p = 2, dim = -1)  # good stuff for relevance matching
            d_out = F.normalize(d_out, p = 2, dim = -1)
            q_convs.append(q_out)
            d_convs.append(d_out)

        output_phis = []
        for idx in range(self.max_ngram):
            query_local_context = self._get_tensor_context(q_convs[idx], query_mask, self.context_window)  # (B, L, D)
            doc_local_context = self._get_tensor_context(d_convs[idx], doc_mask, self.context_window)  # (B, R, D)
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
            # if self.use_double_attention:
            #     tensors_new = self.double_att_net(tensors, tensor_mask)  # B, C, L, R
            #     tensors_new = tensors_new * tensor_mask.unsqueeze(1)  # reset pad tokens again!!!
            #     tensors = tensors + tensors_new  # add residual connection
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

    def _get_bilinear_attention(self, left_tsr: torch.Tensor, right_tsr: torch.Tensor):
        """
        Parameters
        ----------
        left_tsr: (B, L, D)
        right_tsr: (B, R, D)
        """
        B, L, D = left_tsr.size()
        w1 = self.linearQ(left_tsr.view(B * L, -1)).view(B, L, 1)  # (B, L, 1)
        B, R, D = right_tsr.size()
        w2 = self.linearD(right_tsr.view(B * R, -1)).view(B, R, 1)  # (B, R, 1)
        dot = self.linearQD(left_tsr.view(B * L, -1)).view(B, L, D)
        dot = torch.bmm(dot, right_tsr.permute(0, 2, 1))  # (B, L, R)
        return w1 + dot + w2.permute(0, 2, 1)

    def _get_decompositional_mat(self, left_tsr: torch.Tensor, right_tsr: torch.Tensor, mask: torch.Tensor):
        # we usually normalize left_tsr and right_tsr so basically it is a tanh version already.
        # Therefore, we don't need to use tanh second time
        E = self._get_sim_matrix(left_tsr, right_tsr)
        # E = E * mask
        # meanE = torch.sum(E) / float(torch.sum(mask))
        # E = E - meanE  # centering E first, then tanh

        N = self._get_disimilarity_mat(left_tsr, right_tsr, mask)
        return E * N

    def _get_disimilarity_mat(self, left_tsr: torch.Tensor, right_tsr: torch.Tensor, mask: torch.Tensor, average: bool):
        """
        disimilarity matrix (using either l1 or l2 norm) as gating
        https://vanzytay.github.io/files/NeurIPS_2019_CODA.pdf  (NIPS 2019)
        the values shoule be in range [0, 2 * sqrt(2)]
        Parameters
        ----------
        left_tsr: `torch.Tensor` (B, L, D)
        right_tsr: `torch.Tensor` (B, R, D)
        mask: (B, L, R)
        Returns
        -------
        output a tensor of shape (B, L, R) where each mat is a disimilarity matrix
        """
        self.beta = 1
        A = -self.beta * torch_utils.cosine_distance(left_tsr, right_tsr)
        assert torch.max(A) <= 0, torch.max(A)

        if average:
            # centering
            A = A * mask
            mean = torch.sum(A) / float(torch.sum(mask))
            A = A - mean
            return torch.sigmoid(A)
        else:
            # using 2 * sigmoid similar to the paper, since all memembers in A is <= 0, sigmoid(A) in [0, 0.5],
            # then we need to double
            # print("Using sigmoid")
            # print(A)
            return 2 * torch.sigmoid(A)

    def _get_sim_matrix(self, left_tensor: torch.Tensor, right_tensor: torch.Tensor):
        """
        Right now, I use cosine sime, but we can override this function in the future
        Parameters
        ----------
        left_tensor : (B, L, D)
        right_tensor : (B, R, D)

        Returns
        -------

        """
        right_tensor = right_tensor.permute(0, 2, 1)
        matching_matrix = torch.bmm(left_tensor, right_tensor)  # shape = [B, L, R]
        return matching_matrix

    def _get_tensor_context(self, doc: torch.Tensor, doc_mask: torch.Tensor, context_window_size: int) -> torch.Tensor:
        """
        for document context or query context
        Parameters
        ----------
        doc: (B, R, D)
        doc_mask (B, R)
        context_window_size: int

        Returns
        -------
        shape: (B, R, D)
        """
        document_context = torch_utils._get_doc_context_copacrr(doc, doc_mask, context_window_size=context_window_size)
        return F.normalize(document_context, p=2, dim=-1)
