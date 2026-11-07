#!/usr/bin/env python3
import os
import runpy

from efficient_ocr.model.effocr import EffOCR

conf = float(os.environ["EFFOCR_LOCALIZER_CONF"])
iou = float(os.environ["EFFOCR_LOCALIZER_IOU"])

original_load_config = EffOCR._load_config

def load_config_with_thresholds(self, config, **kwargs):
    merged = original_load_config(self, config, **kwargs)

    merged["Localizer"]["training"]["conf_thresh"] = conf
    merged["Localizer"]["training"]["iou_thresh"] = iou

    print(
        f"Threshold override: conf={conf}, iou={iou}",
        flush=True,
    )
    return merged

EffOCR._load_config = load_config_with_thresholds

runpy.run_path(
    "ocr_models/effocr/run_effocr_ocr.py",
    run_name="__main__",
)
