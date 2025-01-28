import torch
import torch.nn.functional as F
from torchvision.transforms import GaussianBlur
from lava.lib.dl.slayer.utils.time.replicate import replicate
from lava.lib.dl.slayer.classifier import Rate, MovingWindow
from lava.lib.dl.slayer.loss import SpikeRate
import logging

class SpikeCenterLoss(torch.nn.Module):
    """SpikeCenterLoss.

    Note: input is always collapsed in spatial dimension.

    Parameters
    ----------
    moving_window : int
        size of moving window. If not None, assumes label to be specified
        at every time step. Defaults to None.

    Returns
    -------

    """
    def __init__(
        self, reduction='mean', n_classes = 13, use_target_weight=False, target_weight=[1], sequence_length=1, regression_weight=1, offset_weight=1):
        super(SpikeCenterLoss, self).__init__()
        self.n_classes = n_classes
        self.reduction = reduction
        self.blur = GaussianBlur((25, 25), 4.0)
        self.use_target_weight = use_target_weight
        self.target_weight = target_weight
        self.regression_weight = regression_weight
        self.offset_weight = offset_weight
        self.sequence_length = sequence_length

    def centernetfocalLoss(self, pred, gt):
        ''' Modified focal loss. Exactly the same as CornerNet.
          Runs faster and costs a little bit more memory
        Arguments:
          pred (batch x c x h x w)
          gt_regr (batch x c x h x w)
        '''
        pred = pred.clamp(1e-4,1-1e-4)
        pos_inds = gt.eq(1).float()
        neg_inds = gt.lt(1).float()

        neg_weights = torch.pow(1 - gt, 4)

        loss = 0

        pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
        neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

        num_pos = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()
        if num_pos == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos
        return loss

    def maxPointPth(self, heatmap, center=False):
        # from the heatmap find the center point
        # pytorch version
        # n,1,h,w
        # 计算center heatmap的最大值得到中心点
        if center:
            heatmap = heatmap * self.center_weight[:heatmap.shape[0], ...]
            # 加权取最靠近中间的

        n, c, h, w = heatmap.shape
        heatmap = heatmap.reshape((n, -1))  # 64, 48x48
        # print(heatmap[0])
        # max_id = torch.argmax(heatmap, 1)#64, 1
        # print(max_id)
        max_v, max_id = torch.max(heatmap, 1)  # 64, 1
        # print(max_v)
        # print("max_i: ",max_i)

        # mask0 = torch.zeros(max_v.shape).to(heatmap.device)
        # mask1 = torch.ones(max_v.shape).to(heatmap.device)
        # mask = torch.where(torch.gt(max_v,th), mask1, mask0)
        # print(mask)
        # b
        y = torch.div(max_id, w, rounding_mode='floor')
        x = max_id % w

        return x, y

    def myMSEwithWeight(self, pre, target):
        # target 0-1
        # pre = torch.sigmoid(pre)
        # print(torch.max(pre), torch.min(pre))
        # b
        loss = torch.pow((pre - target), 2)
        # loss = torch.abs(pre-target)

        # weight_mask = (target+0.1)/1.1
        weight_mask = target * 8 + 1
        # weight_mask = torch.pow(target,2)*8+1

        # gamma from focal loss
        # gamma = torch.pow(torch.abs(target-pre), 2)

        loss = loss * weight_mask  # *gamma

        loss = torch.sum(loss) / target.shape[0] / target.shape[1]

        # bg_loss = self.bgLoss(pre, target)
        return loss

    
    def heatmapLoss(self, pred, target, batch_size):
        # [64, 7, 48, 48]
        #print(pred.shape, target.shape)
        #logging.info(pred.shape)
        #logging.info(target.shape)
        heatmaps_pred = pred.reshape((batch_size, pred.shape[1], -1)).split(1, 1)
        # #对tensor在某一dim维度下，根据指定的大小split_size=int，或者list(int)来分割数据，返回tuple元组
        #print(len(heatmaps_pred), heatmaps_pred[0].shape)#7 torch.Size([64, 1, 48*48]
        heatmaps_gt = target.reshape((batch_size, pred.shape[1], -1)).split(1, 1)

        loss = 0
        for idx in range(pred.shape[1]):
            heatmap_pred = heatmaps_pred[idx].squeeze()#[64, 40*40]
            
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                loss += self.centernetfocalLoss(
                                heatmap_pred.mul(self.target_weight[idx//2]),
                                heatmap_gt.mul(self.target_weight[idx//2])
                            )
            else:
                loss += self.centernetfocalLoss(heatmap_pred, heatmap_gt)
        loss /= pred.shape[1]
        return loss #self.myMSEwithWeight(pred, target)

    def centerLoss(self, pred, target, batch_size):
        # heatmaps_pred = pred.reshape((batch_size, -1))
        # heatmaps_gt = target.reshape((batch_size, -1))
        return self.myMSEwithWeight(pred, target)

    
    def regsLoss(self, pred, target, cx0, cy0, kps_mask, batch_size, num_joints):
        # [64, 14, 48, 48]
        # print(target.shape, cx0.shape, cy0.shape)#torch.Size([64, 14, 48, 48]) torch.Size([64]) torch.Size([64])

        _dim0 = torch.arange(0, batch_size).long()
        _dim1 = torch.zeros(batch_size).long()

        # print("regsLoss: " , cx0,cy0)
        # print(target.shape)#torch.Size([1, 14, 48, 48])
        # print(torch.max(target[0][2]), torch.min(target[0][2]))
        # print(torch.max(target[0][3]), torch.min(target[0][3]))

        # cv2.imwrite("t.jpg", target[0][2].cpu().numpy()*255)
        loss = 0
        for idx in range(num_joints):
            gt_x = target[_dim0, _dim1 + idx * 2, cy0, cx0]
            gt_y = target[_dim0, _dim1 + idx * 2 + 1, cy0, cx0]

            pre_x = pred[_dim0, _dim1 + idx * 2, cy0, cx0]
            pre_y = pred[_dim0, _dim1 + idx * 2 + 1, cy0, cx0]

            # print(torch.max(target[_dim0,_dim1+idx*2,:,:]),torch.min(target[_dim0,_dim1+idx*2,:,:]))
            # print(gt_x,pre_x)                                       
            # print(gt_y,pre_y)

            # print(kps_mask[:,idx])
            # print(gt_x,pre_x)
            # print(self.l1(gt_x,pre_x,kps_mask[:,idx]))
            # print('---')
            # 
            if kps_mask :
                loss += self.l1(gt_x, pre_x, kps_mask[:, idx])
                loss += self.l1(gt_y, pre_y, kps_mask[:, idx])
            else :
                loss += torch.sum(torch.abs(pre_x - gt_x))
                loss += torch.sum(torch.abs(pre_y - gt_y))
        
        # b
        # offset_x_pre = torch.clip(pre_x,0,_feature_map_size-1).long()
        # offset_y_pre = torch.clip(pre_y,0,_feature_map_size-1).long()
        # offset_x_gt = torch.clip(gt_x+cx0,0,_feature_map_size-1).long()
        # offset_y_gt = torch.clip(gt_y+cy0,0,_feature_map_size-1).long()

        return loss / num_joints

    def offsetLoss(self, pred, target, cx0, cy0, regs, kps_mask, batch_size, num_joints):
        _dim0 = torch.arange(0, batch_size).long()
        _dim1 = torch.zeros(batch_size).long()
        loss = 0
        # print(gt_y,gt_x)
        # print(num_joints)
        h,w = target.shape[2], target.shape[3]

        for idx in range(num_joints):
            gt_x = regs[_dim0, _dim1 + idx * 2, cy0, cx0].long() + cx0
            gt_y = regs[_dim0, _dim1 + idx * 2 + 1, cy0, cx0].long() + cy0

            gt_x[gt_x > w-1] = w-1
            gt_x[gt_x < 0] = 0
            gt_y[gt_y > h-1] = h-1
            gt_y[gt_y < 0] = 0

            gt_offset_x = target[_dim0, _dim1 + idx * 2, gt_y, gt_x]
            gt_offset_y = target[_dim0, _dim1 + idx * 2 + 1, gt_y, gt_x]

            pre_offset_x = pred[_dim0, _dim1 + idx * 2, gt_y, gt_x]
            pre_offset_y = pred[_dim0, _dim1 + idx * 2 + 1, gt_y, gt_x]

            # print(gt_offset_x, torch.max(target[_dim0,_dim1+idx*2,...]),torch.min(target[_dim0,_dim1+idx*2,...]))
            # print(gt_offset_y, torch.max(target[_dim0,_dim1+idx*2+1,...]),torch.min(target[_dim0,_dim1+idx*2+1,...]))
            if kps_mask :
                loss += self.l1(gt_offset_x, pre_offset_x, kps_mask[:, idx])
                loss += self.l1(gt_offset_y, pre_offset_y, kps_mask[:, idx])
            else :
                loss += torch.sum(torch.abs(pre_offset_x - gt_offset_x))
                loss += torch.sum(torch.abs(pre_offset_y - gt_offset_y))
        
        #     print(gt_y,gt_x)    
        # b
        return loss / num_joints

        """
        0.0 0.5
        0.0 0.75
        0.75 0.25
        0.0 0.75
        0.0 0.5
        """

    def forward(self, input, target):
        """Forward computation of loss.
        """
        joints_loss, center_loss, reg_loss, off_loss = 0, 0, 0, 0
        for s in range(self.sequence_length):
            seq_input, seq_target = input[s], target[:,s,...]
            pred_joints, pred_center, pred_reg, pred_off = seq_input[0][:], seq_input[1][:], seq_input[2][:], seq_input[3][:]
            heatmaps = seq_target[:, :self.n_classes, :, :]
            centers = seq_target[:, self.n_classes:self.n_classes+1, :, :] #torch.Size([batch_size, 1, 32, 32])
            regs = seq_target[:, self.n_classes+1:(self.n_classes*3)+1, :, :]
            offsets = seq_target[:, (self.n_classes*3)+1:, :, :]
            cx, cy = self.maxPointPth(centers)
            batch_size = pred_joints.shape[0]
            
            cx, cy = torch.clip(cx, 0, pred_joints.shape[2]-1).long(), \
                        torch.clip(cy, 0,pred_joints.shape[3]-1).long()

            #logging.info("Joints output sum: " + str(torch.sum(torch.mean(pred_joints, 4))))
            #logging.info("Joints output mean: " + str(torch.mean(torch.mean(pred_joints, 4))))

            #logging.info("GT Joints output max: ", torch.max(heatmaps))
            joints_loss += self.heatmapLoss(pred_joints, heatmaps, batch_size)
            center_loss += self.centerLoss(pred_center, centers, batch_size)
            reg_loss += self.regsLoss(pred_reg, regs, cx, cy, None, batch_size, self.n_classes)
            off_loss += self.offsetLoss(pred_off, offsets, cx, cy, regs, None, batch_size, self.n_classes)
            
        loss = center_loss + joints_loss + self.regression_weight*reg_loss + self.offset_weight*off_loss
        logging.info(f"Total loss {loss/self.sequence_length} : joint loss {joints_loss/self.sequence_length} |" \
            f"center loss {center_loss/self.sequence_length} | regression loss {reg_loss/self.sequence_length} | offset loss {off_loss/self.sequence_length}")
                
        return loss


class ClassificationLoss(torch.nn.Module):
    """SpikeCenterLoss.

    Note: input is always collapsed in spatial dimension.

    Parameters
    ----------
    moving_window : int
        size of moving window. If not None, assumes label to be specified
        at every time step. Defaults to None.

    Returns
    -------

    """
    def __init__(self, reduction='sum', mode = {'actions':8,'legs':8,'arms':8}, spiking=True):
        super(ClassificationLoss, self).__init__()
        self.mode = mode
        self.spiking = spiking
        if spiking:
            self.class_loss = SpikeRate(true_rate=0.2, false_rate=0.03, reduction=reduction)
            #self.class_loss = SpikingBodyRate(true_rate=0.2, false_rate=0.03, reduction=reduction)
        else:
            self.class_loss = torch.nn.CrossEntropyLoss()
        self.sum = sum(mode.values())

    def forward(self, input, target):
        """Forward computation of loss.
        """
        loss = []
        for i, key in enumerate(self.mode.keys()):
            if self.spiking:
                #logging.info(input.shape) (#output_heads,#batch,#classes_per_head,#timesteps)
                #output_spikes = input.swapaxes(0,1).swapaxes(1,2).flatten(2)
                #output_spikes = input[0] #for tennis detection with one head
                output_spikes = input[i] #for multiclass detection with 2 heads
                if len(self.mode.keys()) > 1 :
                    target_head = target[:,i] #for multiclass detection with 2 heads
                else :
                    target_head = target #for 1 head classification
                #SpikeRateLoss expect (#batch,output,#time) as an input
                #Target has the shape (#batch,#heads)
                #loss.append(self.mode[key]*self.class_loss(output_spikes, target.long())/self.sum)
                loss.append(self.class_loss(output_spikes, target_head.long()))
            else:
                target_non_indexed = torch.tensor.zero((target.shape[0],input[i].shape[-2]))
                target_non_indexed[target[:,i]] = 1
                loss.append(self.mode[key]*self.class_loss(input[i], target_non_indexed.long())/self.sum)

        #logging.info(f"Total loss: {sum(loss)}")
                
        return sum(loss)

class SpikingBodyRate(torch.nn.Module):
    """Spike rate loss.

    .. math::
        \\hat {\\boldsymbol r} &=
            r_\\text{true}\\,{\\bf 1}[\\text{label}] +
            r_\\text{false}\\,(1 - {\\bf 1}[\\text{label}])\\

        L &= \\begin{cases}
        \\frac{1}{2}\\int_T(
            {\\boldsymbol r}(t) - \\hat{\\boldsymbol r}(t)
        )^\\top {\\bf 1}\\,\\text dt &\\text{ if moving window}\\\\
        \\frac{1}{2}(
            \\boldsymbol r - \\hat{\\boldsymbol r}
        )^\\top 1 &\\text{ otherwise}
        \\end{cases}


    Note: input is always collapsed in spatial dimension.

    Parameters
    ----------
    true_rate : float
        true spiking rate.
    false_rate : float
        false spiking rate.
    moving_window : int
        size of moving window. If not None, assumes label to be specified
        at every time step. Defaults to None.
    reduction : str
        loss reduction method. One of 'sum'|'mean'. Defaults to 'sum'.

    Returns
    -------

    """
    def __init__(
        self, true_rate, false_rate,
        moving_window=None, reduction='sum'
    ):
        super(SpikingBodyRate, self).__init__()
        if not (true_rate >= 0 and true_rate <= 1):
            raise AssertionError(
                f'Expected true rate to be between 0 and 1. Found {true_rate=}'
            )
        if not (false_rate >= 0 and false_rate <= 1):
            raise AssertionError(
                f'Expected false rate to be between 0 and 1. '
                f'Found {false_rate=}'
            )
        self.true_rate = true_rate
        self.false_rate = false_rate
        self.reduction = reduction
        if moving_window is not None:
            self.window = MovingWindow(moving_window)
        else:
            self.window = None

    def forward(self, input, label):
        """Forward computation of loss.
        """
        input = input.reshape(input.shape[0], -1, input.shape[-1])
        noise_mask = label == 0
        if self.window is None:  # one label for each sample in a batch
            label = label.clone() - 1
            label[noise_mask] = 0
            one_hot = F.one_hot(label, input.shape[1])
            one_hot[noise_mask, :] = 0
            spike_rate = Rate.rate(input)
            target_rate = self.true_rate * one_hot \
                + self.false_rate * (1 - one_hot)
            return F.mse_loss(
                spike_rate.flatten(),
                target_rate.flatten(),
                reduction=self.reduction
            )

        if len(label.shape) == 1:  # assume label is in (batch, time) form
            label = replicate(label, input.shape[-1])
        # transpose the time dimension to the end
        # (batch, time, num_class) -> (batch, num_class, time)
        one_hot = F.one_hot(
            label,
            num_classes=input.shape[1]
        ).transpose(2, 1)  # one hot encoding in time
        spike_rate = self.window.rate(input)
        target_rate = self.true_rate * one_hot \
            + self.false_rate * (1 - one_hot)
        return F.mse_loss(
            spike_rate.flatten(),
            target_rate.flatten(),
            reduction=self.reduction
        )
    

class SpikingBodyClassifier(torch.nn.Module):
    """Global rate based classifier. It considers the event rate of the spike
    train over the entire duration as the confidence score.

    .. math::
        \\text{rate: } {\\bf r} &= \\frac{1}{T}\\int_T{\\bf s}(t)\\,\\text dt\\

        \\text{confidence: } {\\bf c} &= \\begin{cases}
            \\frac{\\bf r}{\\bf r^\\top 1}
                &\\text{ if mode=probability} \\\\
            \\frac{\\exp({\\bf r})}{\\exp({\\bf r})^\\top \\bf1}
                &\\text{ if mode=softmax} \\\\
            \\log\\left(
                \\frac{\\exp({\\bf r})}{\\exp({\\bf r})^\\top \\bf1}
            \\right)
                &\\text{ if mode=softmax}
        \\end{cases} \\\\

        \\text{prediction: } p &= \\arg\\max(\\bf r)

    Examples
    --------

    >>> classifier = Rate
    >>> prediction = classifier(spike)
    """

    def __init__(self):
        super(SpikingBodyClassifier, self).__init__()

    def forward(self, spike):
        """
        """
        return SpikingBodyClassifier.predict(spike)

    @staticmethod
    def rate(spike):
        """Given spike train, returns the output spike rate.

        Parameters
        ----------
        spike : torch tensor
            spike tensor. First dimension is assumed to be batch, and last
            dimension is assumed to be time. Spatial dimensions are collapsed
            by default.

        Returns
        -------
        torch tensor
            spike rate.

        Examples
        --------

        >>> rate = classifier.rate(spike)
        >>> rate = Rate.rate(spike)
        """
        return torch.mean(spike, dim=-1)

    @staticmethod
    def confidence(spike, mode='probability', eps=1e-6):
        """Given spike train, returns the confidence of the output class based
        on spike rate.

        Parameters
        ----------
        spike : torch tensor
            spike tensor. First dimension is assumed to be batch, and last
            dimension is assumed to be time. Spatial dimensions are collapsed
            by default.
        mode : str
            confidence mode. One of 'probability'|'softmax'|'logsoftmax'.
            Defaults to 'probability'.
        eps : float
            infinitesimal value. Defaults to 1e-6.

        Returns
        -------
        torch tensor
            confidence.

        Examples
        --------

        >>> confidence = classifier.confidence(spike)
        >>> confidence = Rate.confidence(spike)
        """
        rate = SpikingBodyClassifier.rate(spike).reshape(spike.shape[0], -1)
        if mode == 'probability':
            return rate / (torch.sum(rate, dim=1, keepdim=True) + eps)
        elif mode == 'softmax':
            return F.softmax(rate, dim=1)
        elif mode == 'logsoftmax':
            return F.log_softmax(rate, dim=1)

    @staticmethod
    def predict(spike):
        """Given spike train, predicts the output class based on spike rate.

        Parameters
        ----------
        spike : torch tensor
            spike tensor. First dimension is assumed to be batch, and last
            dimension is assumed to be time. Spatial dimensions are collapsed
            by default.

        Returns
        -------
        torch tensor
            indices of max spike activity.

        Examples
        --------

        >>> prediction = classifier.predict(spike)
        >>> prediction = Rate.predict(spike)
        """
        pos_threshold = 0.05
        rate = SpikingBodyClassifier.rate(spike)
        pred = torch.zeros(rate.shape[0]).to(spike.device)
        for i, r in enumerate(rate):
            if r.any() > pos_threshold:
                pred[i] = torch.argmax(r) + 1
        return pred