import torch
import torch.nn as nn
from torch.nn import functional as F

try:
    import torch.distributed.nn
    from torch import distributed as dist
    has_distributed = True
except ImportError:
    has_distributed = False

def gather_features(
        image_features,
        text_features,
        local_loss=False,
        gather_with_grad=False,
        rank=0,
        world_size=1,
        args=None,
        positive_text_features = None,
):
    assert has_distributed, 'torch.distributed did not import correctly, please use a PyTorch version with support.'
    # We gather tensors from all gpus
    if args.vl_pos:
        gathered_positive_text_features = [
            torch.zeros_like(positive_text_features) for _ in range(world_size)
        ]
        dist.all_gather(gathered_positive_text_features, positive_text_features)
        all_positive_text_features = torch.cat(
            [positive_text_features]
            + gathered_positive_text_features[:rank]
            + gathered_positive_text_features[rank + 1:]
        )
    if args.vl_negs:
        text_features = text_features[:len(image_features)]
        negative_text_features = text_features[len(image_features)+1:]
        gathered_negative_text_features = [
            torch.zeros_like(negative_text_features) for _ in range(world_size)
        ]
        dist.all_gather(gathered_negative_text_features, negative_text_features)
        all_negative_text_features = torch.cat(
            [negative_text_features]
            + gathered_negative_text_features[:rank]
            + gathered_negative_text_features[rank + 1:]
        )

    if gather_with_grad:
        all_image_features = torch.cat(torch.distributed.nn.all_gather(image_features), dim=0)
        all_text_features = torch.cat(torch.distributed.nn.all_gather(text_features), dim=0)
    else:
        gathered_image_features = [torch.zeros_like(image_features) for _ in range(world_size)]
        gathered_text_features = [torch.zeros_like(text_features) for _ in range(world_size)]
        dist.all_gather(gathered_image_features, image_features)
        dist.all_gather(gathered_text_features, text_features)
        if not local_loss:
            # ensure grads for local rank when all_* features don't have a gradient
            gathered_image_features[rank] = image_features
            gathered_text_features[rank] = text_features
        all_image_features = torch.cat(gathered_image_features, dim=0)
        all_text_features = torch.cat(gathered_text_features, dim=0)
    if args.vl_negs:
        all_text_features = torch.cat([all_text_features] + [all_negative_text_features])

    return all_image_features, all_text_features


class ClipLoss(nn.Module):

    def __init__(
            self,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            args=None
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        # cache state
        self.prev_num_logits = 0
        self.labels = {}
        self.args = args

    def forward(self, image_features, text_features, logit_scale, pos_that_have_negs):
        device = image_features.device
        positive_text_features = None
        loss_pos = loss_neg = 0.
        if self.args.vl_pos:
            positive_text_features = text_features[:len(image_features)]
            text_features = text_features[len(image_features):]

            loss_pos = self.get_loss_pos_only(image_features, text_features[:len(image_features)], positive_text_features,
                                         logit_scale)
        if self.args.vl_negs and pos_that_have_negs:
            loss_neg = self.get_loss_neg_only(image_features, text_features, logit_scale, pos_that_have_negs)
            if self.args.no_neg_in_contrastive:
                text_features = text_features[:len(image_features)]

        if self.world_size > 1:
            all_image_features, all_text_features = gather_features(
                image_features, text_features,
                self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.args, positive_text_features)

            if self.local_loss:
                logits_per_image = logit_scale * image_features @ all_text_features.T
                logits_per_text = logit_scale * text_features @ all_image_features.T
            else:
                logits_per_image = logit_scale * all_image_features @ all_text_features.T
                logits_per_text = logits_per_image.T
        else:
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logit_scale * text_features @ image_features.T

        if self.args.vl_negs and pos_that_have_negs:
            logits_per_text = logits_per_text[:len(logits_per_image)]

        # calculated ground-truth and cache if enabled
        num_logits = logits_per_image.shape[0]
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]

        total_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
            ) / 2

        if self.args.vl_pos:
            total_loss += loss_pos
        if self.args.vl_negs and pos_that_have_negs:
            total_loss += self.args.neg_w*loss_neg
            return total_loss, loss_neg
        return total_loss, total_loss*0

    def get_loss_neg_only(self, image_features, text_features, logit_scale,pos_that_have_negs):
        num_imgs = image_features.shape[0]
        image_features = image_features[pos_that_have_negs]
        pos = text_features[:num_imgs][pos_that_have_negs].unsqueeze(1)
        neg = text_features[num_imgs:].view((len(pos_that_have_negs), -1, text_features.shape[-1]))
        pos_neg = torch.cat([pos,neg], dim=1)
        image_features = image_features.unsqueeze(2)
        logits = logit_scale * torch.matmul(pos_neg,image_features)[:,:,0]
        ground_truth = torch.zeros(len(pos_that_have_negs)).long()
        ground_truth = ground_truth.to(self.args.device, non_blocking=True)
        total_loss = F.cross_entropy(logits, ground_truth)#zero is the right "class". the positive are always on the 0 place
        return total_loss



    def get_loss_pos_only(self, image_features, text_features,poss_features, logit_scale):
        logits_per_image_text_pos = logit_scale * image_features @ poss_features.t()
        ground_truth = (torch.arange(len(logits_per_image_text_pos)).long()).to(self.args.device, non_blocking=True)
        logits_text_pos_to_text = logit_scale * text_features @ poss_features.t()
        if self.args.symmetric:
            logits_per_image_text_pos_op = logit_scale * poss_features @ image_features.t()
            logits_text_pos_to_text_op = logit_scale * poss_features @ text_features.t()
            total_loss = (F.cross_entropy(logits_per_image_text_pos, ground_truth)
                          + F.cross_entropy(logits_per_image_text_pos_op, ground_truth)
                         ) / 2
            total_loss += ( F.cross_entropy(logits_text_pos_to_text, ground_truth)
                    + F.cross_entropy(logits_text_pos_to_text_op, ground_truth))/2
            total_loss = total_loss/2
        else:
            total_loss = F.cross_entropy(logits_per_image_text_pos, ground_truth)
            total_loss+= F.cross_entropy(logits_text_pos_to_text, ground_truth)
            total_loss = total_loss / 2


        if self.args.kl_pos:
            kl_loss = nn.KLDivLoss(reduction="batchmean")
            logits_per_image_text = logit_scale * image_features @ text_features.t()
            two_pos_feat = torch.stack([torch.diagonal(logits_per_image_text,0),torch.diagonal(logits_per_image_text_pos,0)],dim=1)
            ground_truth = F.softmax(0.5 + torch.zeros_like(two_pos_feat),dim=1).to(self.args.device, non_blocking=True)
            log_probs = F.log_softmax(two_pos_feat, dim=1)
            loss_kl = 0.1*kl_loss(log_probs, ground_truth)
            total_loss += loss_kl
        if self.args.common_batch_pos:
            kl_loss = nn.KLDivLoss(reduction="batchmean")
            text_and_pos_feat = torch.cat([text_features,poss_features])
            logits_per_image_text_and_pos_feat = logit_scale * image_features @ text_and_pos_feat.t()
            log_probs = F.log_softmax(logits_per_image_text_and_pos_feat, dim=1)
            ground_truth = F.softmax((torch.cat([torch.eye(self.args.batch_size), torch.eye(self.args.batch_size)], dim=1) / 2),dim=1).to(self.args.device, non_blocking=True)
            loss_kl_common_batch_pos = 0.01*kl_loss(log_probs, ground_truth)
            total_loss += loss_kl_common_batch_pos


        return total_loss
