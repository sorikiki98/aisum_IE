from PIL import Image
from ultralytics import YOLO as backbone
import torch.nn as nn
import numpy as np
from object_detection_model import ObjectDetectionModel


class YOLO(ObjectDetectionModel):
    def __init__(self, model_name, config, indexing=False):
        super().__init__(model_name, config)
        self._model = backbone(self.model_cfg["weights"])
        self._num_objects = self.model_cfg["num_objects"]
        self._indexing = indexing

    def forward(self, pil_images, img_ids):
        if not isinstance(pil_images, list):
            pil_images = [pil_images]
        if not isinstance(img_ids, np.ndarray):
            img_ids = np.array([img_ids])
        detection_results = []
        for img in pil_images:
            pred = self.model.predict(img, verbose=False)[0]
            result = self._get_candidates(pred)
            detection_results.append(result)

        flattened_classes, flattened_coordinates, flattened_images, flattened_ids = \
            self._flatten_batch_detection_results(detection_results, img_ids)
        return {
            "detection_classes": flattened_classes,
            "detection_coordinates": flattened_coordinates,
            "detection_images": flattened_images,
            "image_segment_ids": flattened_ids
        }

    @property
    def model(self):
        return self._model

    def _get_candidates(self, pred):
        orig_arr = pred.orig_img[:, :, ::-1]
        original_image = Image.fromarray(orig_arr)
        orig_h, orig_w = pred.boxes.orig_shape

        candidates = []
        for cls_id, conf, bbox in zip(pred.boxes.cls,
                                      pred.boxes.conf,
                                      pred.boxes.xyxy):
            xmin, ymin, xmax, ymax = map(int, bbox)
            area = (xmax - xmin) * (ymax - ymin)
            candidates.append({
                "class": pred.names[int(cls_id)],
                "conf": float(conf),
                "bbox": (xmin, ymin, xmax, ymax),
                "area": area,
                "image": original_image.crop((xmin, ymin, xmax, ymax))
            })

        candidates = self._filter_approx_duplicate_bboxes(candidates, dif=15)
        candidates = self._filter_edge_boxes(candidates, orig_w, orig_h)
        candidates = self._filter_low_confidence(candidates, min_conf=0.4)

        # 위 조건 적용하고 num_objects개 이상일 경우 -> box 사이즈대로 num_objects개만 출력
        if len(candidates) > self._num_objects:
            candidates = self._select_topk_by_area(candidates, k=self._num_objects)

        return [["", (0, 0, orig_w, orig_h), original_image]] + [
            [c["class"], c["bbox"], c["image"]] for c in candidates
        ]

    @staticmethod
    def _flatten_batch_detection_results(detection_results, batch_ids):
        batch_flattened_images = []
        batch_flattened_classes = []
        batch_flattened_ids = []
        batch_flattened_coordinates = []
        for result, batch_id in zip(detection_results, batch_ids):
            detected_classes = [obj[0] for obj in result]
            detected_coordinates = [obj[1] for obj in result]
            detected_images = [obj[2] for obj in result]
            detected_ids = [np.char.add(batch_id, f"_{str(i)}") for i, _ in enumerate(range(len(detected_images)))]

            batch_flattened_classes.extend(detected_classes)
            batch_flattened_coordinates.extend(detected_coordinates)
            batch_flattened_images.extend(detected_images)
            batch_flattened_ids.extend(detected_ids)
        return batch_flattened_classes, batch_flattened_coordinates, batch_flattened_images, batch_flattened_ids

    # 중복 bounding box 제거 로직
    def _filter_approx_duplicate_bboxes(self, candidates, dif):
        sorted_c = sorted(candidates, key=lambda x: x["conf"], reverse=True)
        unique = []
        for c in sorted_c:
            xmin, ymin, xmax, ymax = c["bbox"]
            is_dup = False
            for u in unique:
                uxmin, uymin, uxmax, uymax = u["bbox"]
                if (abs(xmin - uxmin) <= dif and
                        abs(ymin - uymin) <= dif and
                        abs(xmax - uxmax) <= dif and
                        abs(ymax - uymax) <= dif):
                    is_dup = True
                    break
            if not is_dup:
                unique.append(c)
        return unique

    # bounding box가 가장자리 깊은 곳에 있을 경우 삭제하는 로직
    @staticmethod
    def _filter_edge_boxes(candidates, orig_w, orig_h):
        def is_at_border(c):
            xmin, ymin, xmax, ymax = c["bbox"]
            if (ymin < 0.02 * orig_h) and (ymax < 0.20 * orig_h):
                return True
            if (ymin > 0.80 * orig_h) and (ymax > 0.98 * orig_h):
                return True
            if (xmin < 0.02 * orig_w) and (xmax < 0.20 * orig_w):
                return True
            if (xmin > 0.80 * orig_w) and (xmax > 0.98 * orig_w):
                return True
            return False

        return [c for c in candidates if not is_at_border(c)]

    # confidence 낮은 값 삭제 로직
    @staticmethod
    def _filter_low_confidence(candidates, min_conf=0.4):
        return [c for c in candidates if c["conf"] >= min_conf]

    # top-k 박스 선택 로직
    def _select_topk_by_area(self, candidates, k=3):
        if self._indexing:
            return sorted(candidates, key=lambda x: x["area"], reverse=True)[:k]
        else:
            return sorted(candidates, key=lambda x: x["area"], reverse=True)
