from segment_anything import SamPredictor, sam_model_registry
from .SAM import SAM
import numpy as np


class FacebookSAM(SAM):
    def __init__(self, model="default", device=None):
        models = {
            "vit_h": "sam_vit_h_4b8939.pth",
            "vit_l": "sam_vit_l_0b3195.pth",
            "vit_b": "sam_vit_b_01ec64.pth"
        }
        models['default'] = models['vit_h']
        super().__init__(f'sam-{model}', device)
        sam = sam_model_registry[model](checkpoint=f"models/{models[model]}")
        sam.to(self.device)
        self.predictor = SamPredictor(sam)

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
