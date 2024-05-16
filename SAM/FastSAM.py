import cv2
from sam_tracker.utils import Bbox
from fastsam import (
    FastSAM as _FastSAM,
    FastSAMPrompt as _FastSAMPrompt,
    FastSAMPredictor

)
from .SAM import SAM


class FastSAM(SAM):
    def __init__(self, small=True, conf=0.35, device=None):
        model_name = f"FastSAM-{'s' if small else 'x'}"
        super().__init__(model_name, device)
        self.model = _FastSAM(f'models/{model_name}.pt')
        self.prompt_process = None
        self.conf = conf
        self.predictor = None

    def set_image(self, image):

        if self.predictor is None:
            overrides = self.model.overrides.copy()
            overrides['conf'] = 0.25
            overrides.update(dict(device=self.device,
                                  retina_masks=True,
                                  imgsz=1024,
                                  conf=self.conf,
                                  iou=0.80, ))  # prefer kwargs
            overrides['mode'] = 'predict'
            assert overrides['mode'] in ['track', 'predict']
            overrides['save'] = False  # do not save by default if called in Python
            self.predictor = FastSAMPredictor(overrides=overrides)
            self.predictor.setup_model(model=self.model.model, verbose=False)
        everything_results = self.predictor(image, stream=False)
        self.prompt_process = _FastSAMPrompt(
            cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
            everything_results,
            device=self.device
        )

    def box(self, bbox: Bbox):
        return self.prompt_process.box_prompt(bbox=bbox.get_xyxy())[0].astype(bool), self.conf
