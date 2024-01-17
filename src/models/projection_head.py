import torch
from torch import nn
from torch.nn import functional as F

class Projection(nn.Module):

    def __init__(self, input_dim=640, hidden_dim=2048, output_dim=2048):
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.model = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim), nn.BatchNorm1d(self.hidden_dim), nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim, bias=False)
        )

    def forward(self, x):
        x = self.model(x)
        return F.normalize(x, dim=1)


class VLHead(nn.Module):

    def __init__(self,
                 image_input_dim, image_hidden_dim, image_head_out_dim,
                 text_input_dim, text_hidden_dim, text_head_out_dim,
                 temperature
                 ):

        super().__init__()

        self.image_input_dim = image_input_dim
        self.image_hidden_dim = image_hidden_dim
        self.text_input_dim = text_input_dim
        self.text_hidden_dim = text_hidden_dim
        self.image_head_output_dim = image_head_out_dim
        self.text_head_output_dim = text_head_out_dim
        self.temperature = temperature

        # initialize text and image projection heads
        self.image_projection_head = Projection(
            input_dim=self.image_input_dim,
            hidden_dim=self.image_hidden_dim,
            output_dim=self.image_head_output_dim
            )
        self.text_projection_head = Projection(
            input_dim=self.text_input_dim,
            hidden_dim=self.text_hidden_dim,
            output_dim=self.text_head_output_dim
            )

    def cross_entropy(self, preds, targets, reduction='none'):
        log_softmax = nn.LogSoftmax(dim=-1)
        loss = (-targets * log_softmax(preds)).sum(1)
        if reduction == "none":
            return loss
        elif reduction == "mean":
            return loss.mean()

    def symmetric_contrastive_loss(self, image_embeddings, text_embeddings):
        logits = (text_embeddings @ image_embeddings.T) / self.temperature
        texts_similarity = text_embeddings @ text_embeddings.T
        images_similarity = image_embeddings @ image_embeddings.T
        targets = F.softmax(
            (texts_similarity + images_similarity) / 2 * self.temperature
            , dim=-1
        )
        texts_loss = cross_entropy(logits, targets, reduction='none')
        images_loss = cross_entropy(logits.T, targets.T, reduction='none')
        # print(f'Text loss: {texts_loss}, image loss: {images_loss}')
        loss = (images_loss + texts_loss) / 2.0  # shape: (batch_size)
        return loss.mean()

    def forward(self, batch):
        # extract feature representations of each modality
        raw_capts, raw_imgs, caption_ids, image_ids, image_filenames, idx, img_embs, capt_embs = batch
        image_embeddings = self.image_projection_head(img_embs)
        text_embeddings = self.text_projection_head(capt_embs)

        # calculate the loss
        # loss = self.symmetric_contrastive_loss(image_embeddings, text_embeddings)
        # return loss.mean()
        return image_embeddings, text_embeddings