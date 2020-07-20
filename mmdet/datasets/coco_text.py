from .builder import DATASETS
from .coco import CocoDataset
import numpy as np
from mmdet.utils import get_vocabulary



@DATASETS.register_module()
class CocoTextDataset(CocoDataset):
    CLASSES = ('text', )

    def __init__(self, ann_file,pipeline,max_seq_len=25, **kwargs):
        super(CocoTextDataset, self).__init__(ann_file, pipeline, **kwargs)
        self.voc, self.char2id, _ = get_vocabulary("ALLCASES_SYMBOLS")
        self.max_seq_len = max_seq_len

    def _parse_ann_info(self, img_info, ann_info):
        """Parse bbox and mask annotation with transcription.

        Args:
            ann_info (list[dict]): Annotation info of an image .
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,
                labels, masks, seg_map, text_labels. "masks" are raw annotations and not
                decoded into binary masks.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []
        gt_texts = []
        gt_text_masks = []
        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
                gt_masks_ann.append(ann['segmentation'])

                word = ann['transcription']
                word_label = []
                for char in word:
                    if char in self.char2id:
                        word_label.append(self.char2id[char])
                    else:
                        word_label.append(self.char2id['UNK'])
                seq_label = np.full(self.max_seq_len, self.char2id['PAD'], dtype=np.int)
                seq_mask = np.full(self.max_seq_len, 0, dtype=np.int)
                if len(word_label) > (self.max_seq_len - 1):
                    word_label = word_label[:(self.max_seq_len - 1)]
                word_label = word_label + [self.char2id['EOS']]
                seq_label[:len(word_label)] = np.array(word_label)
                word_len = len(word_label)
                seq_mask[:word_len] = 1
                gt_texts.append(seq_label)
                gt_text_masks.append(seq_mask)

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        seg_map = img_info['filename'].replace('jpg', 'png')

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_masks_ann,
            seg_map=seg_map,
            text_labels=gt_texts,
            text_masks=gt_text_masks)

        return ann