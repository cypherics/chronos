from torch import nn


class SpatialAttentionFusionModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, low_level_features, high_level_features):
        """

        :param low_level_features: features extracted from backbone
        :param high_level_features: up sampled features
        :return:
        """
        high_level_features_sigmoid = high_level_features.sigmoid()
        weighted_low_level_features = high_level_features_sigmoid * low_level_features

        feature_fusion = weighted_low_level_features + high_level_features
        return feature_fusion
