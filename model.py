import torch.nn as nn


class emo_classifier_768(nn.Module):
    def __init__(self, ):
        super(emo_classifier_768, self).__init__()
        self.fc = nn.Linear(768, 8)

    def forward(self, x):
        x = self.fc(x)
        return x


class emo_classifier_1024(nn.Module):
    def __init__(self, ):
        super(emo_classifier_1024, self).__init__()
        self.fc = nn.Linear(1024, 8)

    def forward(self, x):
        x = self.fc(x)
        return x