import numpy as np
from .SAM import SAM
from mobile_sam import (
    SamPredictor as mb_SamPredictor,
    sam_model_registry as mb_sam_model_registry
)


class MobileSAM(SAM):
    def __init__(self, device=None):
        super().__init__(f'mobile-sam', device)
        sam = mb_sam_model_registry['vit_t'](checkpoint=f"models/mobile_sam.pt")
        sam.to(self.device)
        self.predictor = mb_SamPredictor(sam)

    def set_image(self, image):
        self.predictor.set_image(image)

    def box(self, bbox):
        masks, scores, logits = self.predictor.predict(
            point_coords=None,
            point_labels=None,
            box=np.array(bbox.get_xyxy())[None, :],
            multimask_output=True,
        )
        max_score_idx = np.argmax(scores)
        return masks[max_score_idx], scores[max_score_idx]
