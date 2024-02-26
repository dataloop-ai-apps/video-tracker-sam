from .utils import Bbox


class TrackedBox:
    def __init__(self, bbox, *, max_age=None):
        self.max_age = max_age
        self.bbox = bbox
        self.visible = True
        self.dead_frames = 0

    @property
    def gone(self):
        return self.max_age and self.dead_frames > self.max_age

    def track(self, sam, frame=None, min_area=None, thresh=None):
        """
        Updates the box to the new location
        :param sam: SAM to use for tracking
        :param frame: frame to track on. if None, sam is assumed to be set with the frame (set_image)
        :param min_area:
        :param thresh:
        :return: the new bbox
        """
        if self.gone:
            return
        if frame:
            sam.set_image(frame)

        ann, score = sam.box(self.bbox)
        ann_bbox = Bbox.from_mask(ann)
        df_ratio = 1 if not self.max_age else (self.max_age - self.dead_frames) / self.max_age

        if ann_bbox:  # if there is an annotation
            if not min_area or ann_bbox.area > min_area:  # and if it's bigger than min_area
                # check if the result bbox is likely correct by averaging:
                #   1. IOU with previous bbox
                #   2. confidence of the annotation
                #   3. a ratio that represents the number of dead frames (frames without detection)
                if not thresh or sum((self.bbox.iou(ann_bbox), score, df_ratio)) / 3 > thresh:
                    # if it's likely the correct bbox, update it and return.
                    self.bbox = ann_bbox
                    self.dead_frames = 0
                    self.visible = True
                    return self.bbox

        self.dead_frames += 1
        self.visible = False
