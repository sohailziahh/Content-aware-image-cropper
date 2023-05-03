import detectron2
import streamlit as st
import torch
from detectron2.utils.logger import setup_logger

# import some common libraries
import numpy as np
import cv2
from enum import Enum
# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

setup_logger()

# Desired sizes; the minimum size sustainable for the resultant images
THUMBNAIL = 720
LANDSCAPE_HEIGHT, LANDSCAPE_WIDTH = 1080, 1920
PORTRAIT_HEIGHT, PORTRAIT_WIDTH = 1350, 1080

# Desired Aspect Ratios
AR_LN = 1.78
AR_PT = 0.8
AR_TN = 1.0

# Error Messages
F_ASPECT_RATIO = "Failure: Failed to match the desired aspect ratio!"
F_INFERIOR_IMG_SIZE_LN = "The resultant image size will be lower than acceptable size for landscape"
F_INFERIOR_IMG_SIZE_PT = "The resultant image size will be lower than acceptable size for portrait"
F_INFERIOR_IMG_SIZE_TN = "The resultant image size will be lower than acceptable size for thumbnail"
FAILURE = "Failure"
SUCCESS = "Success"

TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
CUDA_VERSION = torch.__version__.split("+")[-1]

print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION)
print("detectron2:", detectron2.__version__)


def make_thumbnail(outputs, biggest_box_id, img):
    bb = {'starting_x': 0, 'starting_y': 0, 'width': 0, 'height': 0}
    H, W, C, N = body_stats(outputs, biggest_box_id)

    st.write(C[1] - N[1])
    st.write((C[1] - N[1]) / H)

    # radius = Distance between nose & chest * 2 , as a % of the height
    radius_height_perc = 2 * (C[1] - N[1]) / H

    # Take either 23% of the height, or the above, whichever is bigger
    r = int(np.max([radius_height_perc, 0.33]) * H)

    if W > H:
        r = int(0.35 * H)

    bb['starting_x'] = abs(int(C[0] - r))
    bb['width'] = int(r * 2)

    bb['starting_y'] = abs(int(C[1] - r))
    bb['height'] = int(r * 2)

    thmbnl = img[bb['starting_y']:bb['starting_y'] + bb['height'],
             bb['starting_x']:bb['starting_x'] + bb['width']]

    h, w, c = thmbnl.shape
    image_height, image_width = outputs['instances'].image_size

    print('Thumbnail: ', h / image_height)

    return bb, thmbnl


def make_thumbnail2(bb, p_img, img):
    h, w, c = p_img.shape

    diff = abs(h - w)

    if h > w:
        thmbnl = img[bb['starting_y']:bb['starting_y'] + (bb['height'] - diff),
                 bb['starting_x']:bb['starting_x'] + bb['width']]
    else:
        thmbnl = img[bb['starting_y']:bb['starting_y'] + bb['height'],
                 bb['starting_x']:bb['starting_x'] + (bb['width'] - diff)]

    bb['height'], bb['width'], _ = thmbnl.shape

    return bb, thmbnl


def thumbnails(outputs, biggest_box_id, image, portrait_img, bb):
    bb_t, thmbnl = make_thumbnail2(bb, portrait_img, image)
    h, w, c = thmbnl.shape
    img_res_ok = h >= THUMBNAIL and w >= THUMBNAIL
    title = "Thumbnail 2nd Version"
    if not img_res_ok:
        bb_t, thmbnl = make_thumbnail(outputs, biggest_box_id, image)
        title = "Thumbnail 1st Version"

    st.title(title)
    h, w, c = thmbnl.shape
    img_res_ok = h >= THUMBNAIL and w >= THUMBNAIL

    if not img_res_ok:
        response['status_code'] = 400
        response['error_msgs'] = ["Bad quality!"]
        response['message'] = "Failure"
        print(response)
        result = get_result(response, False, 400, ["Bad quality!"])
        print(result)
        return False, None, None  # Failure

    st.image(thmbnl)
    st.write(bb_t)
    result = get_result(response, True, 200, [], {"thumbnail": bb_t})
    print(result)
    print(response)
    return True, thmbnl, bb_t  # Success


def make_portrait(outputs, biggest_box_id, img, portrait_width=1080, portrait_height=1350):
    bb = {'starting_x': 0, 'starting_y': 0, 'width': 0, 'height': 0}
    H, W, C, N = body_stats(outputs, biggest_box_id)

    expected_AR = portrait_width / portrait_height

    st.write(C[1] - N[1])
    st.write((C[1] - N[1]) / H)

    # radius = Distance between nose & chest * 2 , as a % of the height
    radius_height_perc = 2 * (C[1] - N[1]) / H

    # Take either 33% of the height, or the above, whichever is bigger
    scale_factor = H if W > H else W
    r = int(np.max([radius_height_perc, 0.33]) * scale_factor)

    if W > H:
        r = int(0.35 * H)

    bb['starting_x'] = max(0, abs(int(C[0] - r)))  # TODO cud this ever be a negative?
    width = min(int(r * 2), W)
    bb['width'] = width
    bb['starting_y'] = abs(int(C[1] - r))
    bb['height'] = int(bb['width'] / expected_AR)

    prtrt = img[bb['starting_y']:bb['starting_y'] + bb['height'],
            bb['starting_x']:bb['starting_x'] + bb['width']]

    return bb, prtrt, (prtrt.shape[1] / prtrt.shape[0])


def make_portrait2(biggest_box, biggest_box_id, img, outputs):
    bb = {'starting_x': 0, 'starting_y': 0, 'width': 0, 'height': 0}

    image_height, image_width = outputs['instances'].image_size

    box_area = (biggest_box[2] - biggest_box[0]) * (biggest_box[3] - biggest_box[1])
    image_area = image_height * image_width
    rel_ratio = box_area / image_area
    print(f'BBox % of Image : {rel_ratio * 100.0}')

    # get chest tuple
    left_shoulder = np.asarray(outputs['instances'].get_fields()['pred_keypoints'][biggest_box_id][5][:-1])
    right_shoulder = np.asarray(outputs['instances'].get_fields()['pred_keypoints'][biggest_box_id][6][:-1])

    chest = np.mean([right_shoulder, left_shoulder], axis=0)

    # Calculate distance from chest to the extreme ends i.e. left and right ends of the bbox.
    # ands then take the maximum one. this ensures person is always in the center when its cropped
    # Its an alternate to the radius logic that Mr.Alki devised.
    distances = (abs(biggest_box[2] - chest[0]), abs(chest[0] - biggest_box[0]))
    distances = sorted(distances)

    # distance_weight kindof means; how much of the distance shud be considered when cropping.
    distance_weight = 1.0 if (rel_ratio < 0.46 or image_width < image_height) else 0.5  # TODO use of magic numbersss

    # Update width of the bbox. (left and right)
    biggest_box[0] = max(0, (chest[0] - (distance_weight * distances[-1])))
    biggest_box[2] = min(image_width, (chest[0] + (distance_weight * distances[-1])))

    biggest_box[3] = ((int(biggest_box[2] - biggest_box[0])) / AR_PT)  # add top when its non-zero ??

    height = int(biggest_box[3])
    width = int(biggest_box[2] - biggest_box[0])
    left = int(biggest_box[0])
    top = 0  # TODO int(biggest_box[1] / 2)

    example = img[top:top + height, left:left + width]
    print('Aspect Ratio for Cropped Image ', (example.shape[1] / example.shape[0]))

    bb['starting_x'] = left
    bb['width'] = width

    bb['starting_y'] = top
    bb['height'] = height

    bb['height'], bb['width'], _ = example.shape

    return bb, example, (example.shape[1] / example.shape[0])


def portrait(outputs, biggest_box, biggest_box_id, image):
    st.title("Portrait 2nd Version")
    bb_p, p_img, ar = make_portrait2(biggest_box, biggest_box_id, image, outputs)

    # Check if resolution is good enough.
    h, w, c = p_img.shape
    img_res_ok = h >= PORTRAIT_HEIGHT and w >= PORTRAIT_WIDTH

    if float("{:.1f}".format(ar)) != AR_PT or not img_res_ok:

        # Let's try Alki's logic then, shall we?
        bb_p, p_img, ar = make_portrait(outputs, biggest_box_id, image)
        h, w, c = p_img.shape
        img_res_ok = h >= PORTRAIT_HEIGHT and w >= PORTRAIT_WIDTH

        if float("{:.1f}".format(ar)) != AR_PT or not img_res_ok:
            st.write(F_ASPECT_RATIO)
            reasons = deduce_reasons_for_failure(outputs, biggest_box_id)
            for value in reasons:
                st.write(value)

            reasons.insert(0, F_ASPECT_RATIO)
            if not img_res_ok:
                reasons.insert(0, F_INFERIOR_IMG_SIZE_PT)
            response['status_code'] = 400
            response['error_msgs'] = reasons
            response['message'] = "Failure"

            result = get_result(response, False, 400, reasons)
            print(result)
            print(response)
            return False, None, None

        else:
            st.write("Portrait 2nd Failed")
            st.write(F_INFERIOR_IMG_SIZE_PT)
            st.title("Portrait 1st Worked! ")


    st.image(p_img)
    st.write(bb_p)
    st.write('Aspect Ratio for Cropped Image: ', ar)
    result = get_result(response, True, 200, [], {"portrait": bb_p})
    print(result)
    print(response)
    return True, p_img, bb_p



def make_landscape(outputs, biggest_box_id, img):
    bb = {'starting_x': 0, 'starting_y': 0, 'width': 0, 'height': 0}
    H, W, C, N = body_stats(outputs, biggest_box_id)

    needed_AR = AR_LN

    bb['starting_x'] = 0
    bb['width'] = W

    bb['height'] = int(bb['width'] / needed_AR)
    if (int(N[1]) - int((bb['width'] / needed_AR) / 2)) < 0:  # the face is way too up
        bb['starting_y'] = 0
    else:
        bb['starting_y'] = int(N[1]) - int((bb['width'] / needed_AR) / 2)

    lndscp = img[bb['starting_y']:bb['starting_y'] + bb['height'],
             bb['starting_x']:bb['starting_x'] + bb['width']]

    return bb, lndscp, (lndscp.shape[1] / lndscp.shape[0])



def landscape(outputs, biggest_box_id, image):
    st.title("Landscape")
    bb_l, land, ar = make_landscape(outputs, biggest_box_id, image)

    # Check if resolution is good enough.
    h, w, c = land.shape
    img_res_ok = h >= LANDSCAPE_HEIGHT and w >= LANDSCAPE_WIDTH

    print('lands ', ar, float("{:.2f}".format(ar)))

    if float("{:.2f}".format(ar)) != AR_LN or not img_res_ok:

        if float("{:.2f}".format(ar)):
            st.write(F_ASPECT_RATIO)

        reasons = deduce_reasons_for_failure(outputs, biggest_box_id)
        for value in reasons:
            st.write(value)

        reasons.insert(0, F_ASPECT_RATIO)

        if not img_res_ok:
            reasons.insert(0, F_INFERIOR_IMG_SIZE_LN)
            st.write(F_INFERIOR_IMG_SIZE_LN)

        response['status_code'] = 400
        response['error_msgs'] = reasons
        response['message'] = "Failure"
        result = get_result(response, False, 400, reasons)
        print(result)
        print(response)
        return False, None, None

    else:
        # send success response here
        st.image(land)
        st.write(bb_l)
        st.write('Aspect Ratio for Cropped Image: ', ar)

        result = get_result(response, True, 200, [], {"landscape": bb_l})
        print(result)
        print(response)
        return True, land, bb_l



def body_stats(outputs, biggest_box_id=0):
    height, width = outputs['instances'].image_size

    # Orientation is mirrored.
    right_shoulder = [i.item() for i in (outputs['instances'].get_fields()['pred_keypoints'][biggest_box_id][5])][
                     :-1]  # '5'-th in the Tensor is the Right Shoulder
    left_shoulder = [i.item() for i in (outputs['instances'].get_fields()['pred_keypoints'][biggest_box_id][6])][
                    :-1]  # '6'-th in the Tensor is the Left Shoulder

    chest = np.mean([right_shoulder, left_shoulder], axis=0)

    nose = [i.item() for i in (outputs['instances'].get_fields()['pred_keypoints'][biggest_box_id][0])][
           :-1]  # '0'-th in the Tensor is the Nose

    return height, width, chest, nose


def get_result(result, res, status_code, error_msgs, data=None):
    result['response'] = res
    result['status_code'] = status_code
    result['message'] = 'Success' if result else 'Failure'
    result['error_msgs'] = error_msgs
    if data is not None:
        result['data'].update(data)

    return result


def get_biggest_bbox(model_outputs):
    # Get the area for all the boxes, and find the maximum one
    # this assumes that there's at least one person who has been detected.
    boxes = model_outputs["instances"].get_fields()['pred_boxes'].tensor.numpy()
    areas = np.prod([boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]], axis=0)  # WxH -> (R-L) x (B-T)
    box_id = np.argmax(areas)
    return box_id, boxes[box_id]  # forcefully return 0 for debug


def get_model():
    # ---model---
    cfg = get_cfg()
    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_X_101_32x8d_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8  # set threshold for this model
    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_X_101_32x8d_FPN_3x.yaml")
    cfg.MODEL.DEVICE = 'cpu'
    return DefaultPredictor(cfg), cfg


def mins(h, w, c):
    dist_l = c[0]
    dist_r = w - c[0]

    dist_t = c[1]
    dist_b = h - c[1]

    radius = min([dist_l, dist_r, dist_t, dist_b])

    st.write(h, w, radius)

    return


def draw_model_output(img, outputs, config):
    v = Visualizer(img[:, :, ::-1], MetadataCatalog.get(config.DATASETS.TRAIN[0]), scale=1)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    return out.get_image()[:, :, ::-1]


def filter_out_small_bboxes(model_outputs):
    image_height, image_width = model_outputs['instances'].image_size
    image_area = image_height * image_width

    boxes = model_outputs["instances"].get_fields()['pred_boxes'].tensor.numpy()
    box_areas = np.prod([boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]], axis=0)  # WxH -> (R-L) x (B-T)

    rel_ratios = np.divide(box_areas, image_area)
    print("Rel Bbox % : ", rel_ratios)
    return np.all(rel_ratios < 0.1, axis=0)


def any_person_in_the_image(model_outputs):
    return model_outputs['instances'].get_fields()['pred_boxes'].tensor.shape[0] > 0


def deduce_reasons_for_failure(outputs, biggest_box_id):
    reasons = list()
    nose = outputs['instances'].get_fields()['pred_keypoints'][biggest_box_id][0][:-1]
    nose_normalized_position = nose[1] / outputs['instances'].image_size[0]
    if nose_normalized_position > 0.45:
        reasons.append("Person is relatively below in the image.")
    return reasons



response = {
    "response": True,
    "status_code": 200,
    "message": "Success",
    "error_msgs": [],
    "data": {}
}


class ImageFormatAR(Enum):
    LANDSCAPE = 0.56
    PORTRAIT = 1.25
    THUMBNAIL = 1.0



def fit_window(center, width, maxWidth, squeeze_vertically, is_landscape):
    if width > maxWidth:
        raise RuntimeError("Error: Exceeding Max Image Size")

    divisor = 2.5 if squeeze_vertically else 2.0
    if is_landscape:
        divisor = 1.4
    print(divisor, is_landscape)
    start = int(center - width // divisor)
    end = int(start + width)

    if start < 0:
        # window too far left
        start = 0
        end = width
    elif end > maxWidth:
        # window too far right
        end = maxWidth
        start = end - width
    return start, end



def generate_crop(img, x, y, imageFormatENUM):
    (
        imageHeight,
        imageWidth,
    ) = img.shape[:2]

    imageRatio: float = (imageHeight) / imageWidth

    targetRatio = ImageFormatAR[imageFormatENUM].value

    print(f"TargetRatio: {targetRatio}, ImageRatio: {imageRatio}")

    is_landscape = True if imageFormatENUM.__eq__(ImageFormatAR.LANDSCAPE.name) else False

    if targetRatio < imageRatio:
        # squeeze vertically
        window = fit_window(y, np.round(targetRatio * imageWidth), imageHeight, True, is_landscape)
        top = window[0]
        height = max(window[1] - window[0], 1)
        left = 0
        width = imageWidth
    else:
        # squeeze horizontally
        window = fit_window(x, np.round(imageHeight / targetRatio), imageWidth, False, is_landscape)
        top = 0
        height = imageHeight
        left = window[0]
        width = max(window[1] - window[0], 1)

    return left, top, width, height



def main():
    uploaded_file = st.file_uploader("Choose a file")

    if uploaded_file is not None:
        # To read file as bytes:
        bytes_data = uploaded_file.getvalue()

        st.title("Original")
        st.image(bytes_data)
        nparr = np.fromstring(bytes_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # cv2.IMREAD_COLOR in OpenCV 3.1

        # color was wrong, it was kinda bluish, thus the following line of code.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        predictor, config = get_model()

        outputs = predictor(image)

        output_image = draw_model_output(image, outputs, config)

        st.title("Outputs Visualized")
        st.image(output_image)

        # 1st Check
        any_person_detected = any_person_in_the_image(outputs)

        if not any_person_detected:
            # return something else
            st.write('Failure: No person was detected!')

            response['status_code'] = 400
            response['message'] = "Failure"
            response['error_msgs'] = ["No person was detected! Wrong image maybe."]

            print(response)
            print('Exiting because no person was detected!')
            return

        # 2nd Check
        are_all_boxes_small = filter_out_small_bboxes(outputs)

        if are_all_boxes_small:
            # return something else
            st.write('Failure: All the boxes are too small; implying that main subject is not in the image')

            response['status_code'] = 400
            response['message'] = "Failure"
            response['error_msgs'] = ["All the detected humans in the image are too small; implying that main subject "
                                      "is not in the image"]

            print(response)
            print('Exiting because all the boxes are too small; implying that main subject is not in the image')
            return

        biggest_box_id, biggest_box = get_biggest_bbox(outputs)

        # Landscape
        landscape(outputs, biggest_box_id, image)

        # Portrait
        res, p_img, bb_p = portrait(outputs, biggest_box, biggest_box_id, image)
        # Note: The outputs are used for the 2nd version of Thumbnail.

        # Thumbnail
        if res:
            thumbnails(outputs, biggest_box_id, image, p_img, bb_p)



        # NEW NEW NEW NEW NEW NEW NEW NEW NEW

        # get chest tuple
        left_shoulder = np.asarray(outputs['instances'].get_fields()['pred_keypoints'][biggest_box_id][5][:-1])
        right_shoulder = np.asarray(outputs['instances'].get_fields()['pred_keypoints'][biggest_box_id][6][:-1])

        chest = np.mean([right_shoulder, left_shoulder], axis=0)
        mid_x_bbox = int(abs(biggest_box[2] + biggest_box[0]) // 2)
        print(mid_x_bbox, chest[0])

        # LANDSCAPE
        x, y, w, h = generate_crop(image, mid_x_bbox, chest[1], ImageFormatAR.LANDSCAPE.name)
        salient_cropped_image = image[int(y):int(y + h), int(x):int(x + w)]

        #cv2.imwrite(f'salient_crop_ln.jpg', salient_cropped_image)
        print(salient_cropped_image.shape)

        st.title("New Landscape")
        st.image(salient_cropped_image)
        st.write({"X": x, "Y": y, "Width": w, "Height": h})

        # Portrait
        x, y, w, h = generate_crop(image, mid_x_bbox, chest[1], ImageFormatAR.PORTRAIT.name)
        salient_cropped_image = image[int(y):int(y + h), int(x):int(x + w)]

        #cv2.imwrite(f'salient_crop_pt.jpg', salient_cropped_image)
        print(salient_cropped_image.shape)

        st.title("New Portrait")
        st.image(salient_cropped_image)
        st.write({"X": x, "Y": y, "Width": w, "Height": h})

        # Thumbnail
        x, y, w, h = generate_crop(image, mid_x_bbox, chest[1], ImageFormatAR.THUMBNAIL.name)
        salient_cropped_image = image[int(y):int(y + h), int(x):int(x + w)]

        #cv2.imwrite(f'salient_crop_tn.jpg', salient_cropped_image)
        print(salient_cropped_image.shape)

        st.title("New Thumbnail")
        st.image(salient_cropped_image)
        st.write({"X": x, "Y": y, "Width": w, "Height": h})


if __name__ == '__main__':
    main()

    print(response['error_msgs'])



