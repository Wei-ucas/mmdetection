import torch

from ..builder import HEADS, build_head
from .standard_roi_head import StandardRoIHead
from mmdet.core import bbox2roi


@HEADS.register_module()
class MaskTextspotterHead(StandardRoIHead):

    def __init__(self, mask_textspotter_head, **kwargs):
        assert mask_textspotter_head is not None
        super(MaskTextspotterHead, self).__init__(**kwargs)
        self.mask_textspotter_head = build_head(mask_textspotter_head)

    def init_weights(self, pretrained):
        """Initialize the weights in head.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        super(MaskTextspotterHead, self).init_weights(pretrained)
        self.mask_textspotter_head.init_weights()

    def _mask_forward_train(self, x, sampling_results, bbox_feats, gt_masks,
                            img_metas, gt_texts=None, gt_text_masks=None):
        """Run forward function and calculate loss for Mask head in
               training."""
        pos_labels = torch.cat([res.pos_gt_labels for res in sampling_results])
        mask_results = super(MaskTextspotterHead,
                             self)._mask_forward_train(x, sampling_results,
                                                       bbox_feats, gt_masks,
                                                       img_metas)
        if mask_results['loss_mask'] is None:
            return mask_results

        assert gt_texts is not None

        pos_assigned_gt_inds = [
            res.pos_assigned_gt_inds for res in sampling_results
        ]
        text_labels = []
        text_masks = []
        for i in range(len(pos_assigned_gt_inds)):
            text_labels.append(gt_texts[i][pos_assigned_gt_inds[i],:])
            text_masks.append(gt_text_masks[i][pos_assigned_gt_inds[i],:])
        text_labels = torch.cat(text_labels, dim=0)
        text_masks = torch.cat(text_masks, dim=0)
        logits = self.mask_textspotter_head(mask_results['mask_feats'], text_labels)
        loss_seq = self.mask_textspotter_head.loss(logits, text_masks)

        mask_results['loss_mask'].update(loss_seq)

        return mask_results

    def simple_test_mask(self,
                         x,
                         img_metas,
                         det_bboxes,
                         det_labels,
                         rescale=False):
        """Obtain mask prediction without augmentation."""
        # image shape of the first image in the batch (only one)
        ori_shape = img_metas[0]['ori_shape']
        scale_factor = img_metas[0]['scale_factor']

        if det_bboxes.shape[0] == 0:
            segm_result = [[] for _ in range(self.mask_head.num_classes)]
            pred_logits = [[] for _ in range(self.mask_head.num_classes)]
        else:
            # if det_bboxes is rescaled to the original image size, we need to
            # rescale it back to the testing scale to obtain RoIs.
            _bboxes = (
                det_bboxes[:, :4] *
                det_bboxes.new_tensor(scale_factor) if rescale else det_bboxes)
            mask_rois = bbox2roi([_bboxes])
            mask_results = self._mask_forward(x, mask_rois)
            segm_result = self.mask_head.get_seg_masks(
                mask_results['mask_pred'], _bboxes, det_labels, self.test_cfg,
                ori_shape, scale_factor, rescale)
            pred_logits = self.mask_textspotter_head(mask_results['mask_feats'])

        return segm_result, pred_logits
