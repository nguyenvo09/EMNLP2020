
class KeyWordSettings(object):

    Doc_cID = "doc_cid"
    Doc_URL = "doc_ciurl"
    Doc_cLabel = "doc_clabel"
    Doc_wImages = "doc_wimages"
    Doc_wContent = "doc_wcontent"
    Relevant_Score = "relevant_score"

    Query_id = "qid"
    Query_TweetID = "qtweetid"
    Query_Images = "query_images"
    Ranked_Docs = "ranked_docs"
    Query_Content = "query_content"

    Query_lens = "query_lens"
    Doc_lens = "docs_lens"

    Query_Idf = "query_idf"

    QueryImagesIndices = "query_images_indices"
    DocImagesIndices = "doc_images_indices"
    NegDocImagesIndices = "neg_doc_images_indices"

    QueryIDs = "query_ids"
    DocIDs = "doc_ids"
    UseVisual = "use_visual"

    FullLeftImgTensor = "full_left_images_tensor"
    FullRightImgTensor = "full_right_images_tensor"
    ImageLoaderKey = "image_loader"

    OutputRankingKey = "output_ranking"
    Index2Query = "index2queries"
    Index2Doc = "index2docs"

    Test2Hard = "test2_hard"
    Test3Hard = "test3_hard"

    QueryCountVal = [1116, 1000, 187, 1159]
    QueryCountTest = [1001, 1164, 1118, 187, 156, 1160, 1500]

    UseCuda = "use_cuda"

    LOSS_FUNCTIONS = ('pointwise', 'bpr', 'hinge', 'adaptive_hinge', "single_pointwise_square_loss", "pce", "bce",
                      "cosine_max_margin_loss_dvsh", "cross_entropy", "vanilla_cross_entropy", "regression_loss",
                      "masked_cross_entropy")
