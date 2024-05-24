import mindspore as ms
from mindspore import nn, ops
 
class SigmoidFoaclLoss(nn.Cell):
    def __init__(self, weight=None, gamma=2.0, alpha=0.25, reduction='mean'):
        super(SigmoidFoaclLoss, self).__init__()
        self.sigmoid = ops.Sigmoid()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = ms.Tensor(weight) if weight is not None else weight
        self.reduction = reduction
        self.binary_cross_entropy_with_logits = nn.BCEWithLogitsLoss(reduction="none")
        self.is_weight = (weight is not None)
 
    def reduce_loss(self, loss):
        """Reduce loss as specified.
        Args:
            loss (Tensor): Elementwise loss tensor.
        Return:
            Tensor: Reduced loss tensor.
        """
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss
 
    def weight_reduce_loss(self, loss):
        # if avg_factor is not specified, just reduce the loss
        if self.avg_factor is None:
            loss = self.reduce_loss(loss)
        else:
            # if reduction is mean, then average the loss by avg_factor
            if self.reduction == 'mean':
                loss = loss.sum() / self.avg_factor
            # if reduction is 'none', then do nothing, otherwise raise an error
            elif self.reduction != 'none':
                raise ValueError('avg_factor can not be used with reduction="sum"')
        return loss
 
    def construct(self, pred, target, avg_factor):
        self.avg_factor = avg_factor
        pred_sigmoid = self.sigmoid(pred)
        target = ops.cast(target, pred.dtype)
        pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
        focal_weight = (self.alpha * target + (1 - self.alpha) * (1 - target)) * ops.pow(pt, self.gamma)
        loss = self.binary_cross_entropy_with_logits(pred, target) * focal_weight
        if self.is_weight:
            weight = self.weight
            if self.weight.shape != loss.shape:
                if self.weight.shape[0] == loss.shape[0]:
                    # For most cases, weight is of shape (num_priors, ),
                    #  which means it does not have the second axis num_class
                    weight = self.weight.view(-1, 1)
                elif self.weight.size == loss.size:
                    # Sometimes, weight per anchor per class is also needed. e.g.
                    #  in FSAF. But it may be flattened of shape
                    #  (num_priors x num_class, ), while loss is still of shape
                    #  (num_priors, num_class).
                    weight = self.weight.view(loss.shape[0], -1)
                elif self.weight.ndim != loss.ndim:
                    raise ValueError(f"weight shape {self.weight.shape} is not match to loss shape {loss.shape}")
            loss = loss * weight
        #print(loss)
        loss = self.weight_reduce_loss(loss)
        return loss
 
if __name__ == "__main__":
    import mindspore as ms
    import pickle
    import torch
    ms_s_focal_loss = SigmoidFoaclLoss(weight=None, gamma=2.0, alpha=0.25,
                                       reduction="mean", avg_factor=None)
    with open("/home/ma-user/work/SOLO/data/cate_preds.pkl", "rb") as f:
        pred = pickle.load(f)
    with open("/home/ma-user/work/SOLO/data/flatten_cate_labels.pkl", "rb") as f:
        target = pickle.load(f)
    print(type(pred))
    pred = ms.Tensor(pred.cpu().detach().numpy())
    target = ms.Tensor(target.cpu().detach().numpy())
    one_hot = ops.OneHot()
    num_classes = pred.shape[1]
    on_value = ms.Tensor(1.0, ms.float32)
    off_value = ms.Tensor(0.0, ms.float32)
    target = one_hot(target, num_classes + 1, on_value, off_value)
    target = target[:, 1:]
    # pred = ms.Tensor([[0.1, 0.3, 0.8, 0.1, 0.1]])
    # target = ms.Tensor([[0, 1, 0, 0, 0]])
    loss = ms_s_focal_loss(pred, target)
    print(loss)