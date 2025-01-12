import os

# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["TORCH_CUDA_ARCH_LIST"] = "8.9"
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

import supervision as sv
from supervision.draw.color import ColorPalette

# select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

if device.type == "cuda":
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.float16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif device.type == "mps":
    print(
        "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
        "give numerically different outputs and sometimes degraded performance on MPS. "
        "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
    )

from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

os.environ["TORCH_CUDA_ARCH_LIST"] = "8.9"


def load_models(
    dino_id="IDEA-Research/grounding-dino-base", sam2_id="facebook/sam2-hiera-large"
):
    mask_predictor = SAM2ImagePredictor.from_pretrained(sam2_id, device=device)
    grounding_processor = AutoProcessor.from_pretrained(dino_id)
    grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(dino_id).to(
        device
    )

    return mask_predictor, grounding_processor, grounding_model


mask_predictor, grounding_processor, grounding_model = load_models()

import yaml

with open("cic_objects.yaml", "r") as file:
    data = yaml.safe_load(file)
text_prompt = []
for value in data.values():
    text_prompt += value

text_prompt = " . ".join(text_prompt) + " ."

label_to_id = {
    "wall": -1,
    "floor": -2,
    "ceiling": -3,
}

from tqdm import tqdm
import pathlib
import cv2

from bytetrack.byte_tracker import BYTETracker

from types import SimpleNamespace

# args for BYTETracker
args = SimpleNamespace(
    **{
        "track_thresh": 0.1,
        "track_buffer": 90,
        "match_thresh": 0.85,
        "mot20": False,
        "min_box_area": 100,
    }
)

img_dir = "/home/chadwick/Downloads/cic_1217_image/image"
img_dir = pathlib.Path(img_dir)
img_paths = list(img_dir.glob("*.*"))
img_paths = sorted(img_paths, key=lambda x: int(x.stem))
img_paths = img_paths[380:]

tracker = BYTETracker(args)

paused = False

for img_path in tqdm(img_paths, desc="Processing images"):
    while paused:  # If paused, wait here until resumed
        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):  # Press Space to resume
            paused = False
            print("Resuming processing...")
        elif key == ord("q"):  # Press 'q' to exit
            print("Exiting program...")
            cv2.destroyAllWindows()
            exit()

    image = Image.open(img_path)
    image = np.array(image.convert("RGB"))

    inputs = grounding_processor(
        images=image,
        text=text_prompt,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        outputs = grounding_model(**inputs)

    results = grounding_processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=0.30,
        text_threshold=0.30,
        target_sizes=[image.shape[:2]],
    )

    class_names = np.array(results[0]["labels"])

    for i, class_name in enumerate(class_names):
        for label, des in data.items():
            if class_name in des:
                class_names[i] = label

    input_boxes = results[0]["boxes"].cpu().numpy()  # (n_boxes, 4)
    confidences = results[0]["scores"].cpu().numpy()  # (n_boxes,)
    detection_data = np.hstack((input_boxes, confidences.reshape(-1, 1)))

    class_ids = np.array(list(range(len(class_names))))

    labels = [
        f"{class_name} {confidence:.2f}"
        for class_name, confidence in zip(class_names, confidences)
    ]

    online_targets = tracker.update(detection_data, class_names)
    online_tlwhs = []
    online_tlbrs = []
    onlin_orig_bboxes = []
    online_ids = []
    online_scores = []
    online_class_names = []

    for t in online_targets:
        tlwh = t.tlwh
        tlbr = t.tlbr
        tid = t.track_id
        if tlwh[2] * tlwh[3] > args.min_box_area:
            online_tlwhs.append(tlwh)
            onlin_orig_bboxes.append(t.curr_bbox)
            online_tlbrs.append(tlbr)
            online_scores.append(t.score)
            online_class_names.append(t.class_name)
            if t.class_name in label_to_id:
                online_ids.append(label_to_id[t.class_name])
            else:
                online_ids.append(tid)

    # sam2
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        mask_predictor.set_image(image)

        if len(onlin_orig_bboxes) > 0:
            online_masks, _, _ = mask_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=np.array(onlin_orig_bboxes),
                multimask_output=False,
            )

            if online_masks.ndim == 4:
                online_masks = online_masks.squeeze(1)

            class_ids = np.array(list(range(len(online_class_names))))

    if len(online_ids) > 0:
        detections = sv.Detections(
            xyxy=np.array(onlin_orig_bboxes),
            mask=online_masks.astype(bool),
            class_id=class_ids,
        )
        box_annotator = sv.BoxAnnotator(color=ColorPalette.DEFAULT)
        annotated_frame = box_annotator.annotate(
            scene=image.copy(), detections=detections
        )

        labels = [
            f"{class_name} {id} {confidence:.2f}"
            for class_name, id, confidence in zip(
                online_class_names, online_ids, online_scores
            )
        ]
        label_annotator = sv.LabelAnnotator(
            color=ColorPalette.DEFAULT,
            text_padding=4,
            text_scale=0.3,
            text_position=sv.Position.TOP_LEFT,
            color_lookup=sv.ColorLookup.INDEX,
            smart_position=True,
        )
        annotated_frame = label_annotator.annotate(
            scene=annotated_frame, detections=detections, labels=labels
        )

        mask_annotator = sv.MaskAnnotator(color=ColorPalette.DEFAULT)
        annotated_frame = mask_annotator.annotate(
            scene=annotated_frame, detections=detections
        )
    else:
        annotated_frame = image.copy()

    final_frame = annotated_frame
    cv2.putText(
        final_frame,
        f"{img_path.stem}.png",
        (10, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        2,
        (255, 255, 255),
        1,
    )
    final_frame = cv2.cvtColor(final_frame, cv2.COLOR_RGB2BGR)
    cv2.imshow("Annotated Frame", final_frame)

    # Handle key inputs
    key = cv2.waitKey(0 if paused else 1) & 0xFF  # If paused, wait for input
    if key == ord("q"):  # Press 'q' to quit
        print("Exiting program...")
        break
    elif key == ord(" "):  # Press Space to pause
        paused = True
    # print("Paused. Press Space to resume...")

    # save annotated frame, numpy array h x w x 3
    # try:
    #     img_dir = pathlib.Path("/home/chadwick/Downloads/extracted/image_s")
    #     img_dir.mkdir(exist_ok=True)
    #     Image.fromarray(final_frame).save(img_dir / f"{img_path.stem}.png")
    # except Exception as e:
    #     print(f"Error saving image: {e}")

    # # save class_names, input_boxes, masks, and confidences into npz file
    # npz_dir = pathlib.Path("/home/chadwick/Downloads/cic_1217_npz")
    # npz_dir.mkdir(exist_ok=True)
    # np.savez_compressed(
    #     npz_dir / f"{img_path.stem}.npz",
    #     labels=np.array(online_class_names),
    #     bboxes=np.array(onlin_orig_bboxes),
    #     masks=online_masks,
    #     confidences=np.array(online_scores),
    #     ids=np.array(online_ids),
    # )
