import time 
import numpy as np
from onnxruntime import InferenceSession
import cv2
import paddlex as pdx
import six


# 'model/onnx_model/yolov3.onnx'
class OnnxPredictor():
    def __init__(self, model_dir):
        self.session = InferenceSession(model_dir)

    def get_io_name(self):
        print([input.name for input in self.session.get_inputs()])
        print([output.name for output in self.session.get_outputs()])

    def get_io_shape(self):
        print([input.shape for input in self.session.get_inputs()])
        print([output.shape for output in self.session.get_outputs()])

    def bbox2out(self, results, clsid2catid):
        """
        Args:
            results: request a dict, should include: `bbox`, `im_id`,
                    if is_bbox_normalized=True, also need `im_shape`.
            clsid2catid: class id to category id map of COCO2017 dataset.
            is_bbox_normalized: whether or not bbox is normalized.
        """
        xywh_res = []
        for t in results:
            bboxes = t['bbox'][0]
            lengths = t['bbox'][1][0]
            im_ids = np.array(t['im_id'][0]).flatten()
            if bboxes.shape == (1, 1) or bboxes is None:
                continue

            k = 0
            for i in range(len(lengths)):
                num = lengths[i]
                im_id = int(im_ids[i])
                for j in range(num):
                    dt = bboxes[k]
                    clsid, score, xmin, ymin, xmax, ymax = dt.tolist()
                    catid = (clsid2catid[int(clsid)])

                    
                    w = xmax - xmin + 1
                    h = ymax - ymin + 1

                    bbox = [xmin, ymin, w, h]
                    coco_res = {
                        'image_id': im_id,
                        'category_id': catid,
                        'bbox': bbox,
                        'score': score
                    }
                    xywh_res.append(coco_res)
                    k += 1
        return xywh_res

    def _postprocess(self, res, batch_size, num_classes, labels):
            clsid2catid = dict({i: i for i in range(num_classes)})
            xywh_results = self.bbox2out([res], clsid2catid)
            preds = [[] for i in range(batch_size)]
            for xywh_res in xywh_results:
                image_id = xywh_res['image_id']
                del xywh_res['image_id']
                xywh_res['category'] = labels[xywh_res['category_id']]
                preds[image_id].append(xywh_res)

            return preds

    def postprocess(self, results,
                    batch_size=1):
        """ 对预测结果做后处理

            Args:
                results (list): 预测结果
                topk (int): 分类预测时前k个最大值
                batch_size (int): 预测时图像批量大小
                im_shape (list): MaskRCNN的图像输入大小
                im_info (list)：RCNN系列和分割网络的原图大小
        """

        def offset_to_lengths(lod):
            offset = lod[0]
            lengths = [
                offset[i + 1] - offset[i] for i in range(len(offset) - 1)
            ]
            return [lengths]

        
        res = {'bbox': (results[0][0], offset_to_lengths(results[0][1])), }
        res['im_id'] = (np.array(
            [[i] for i in range(batch_size)]).astype('int32'), [[]])
        preds = self._postprocess(res, batch_size, 1,
                                    ['drill'])
            
        
        return preds[0]
    
    def predict(self, img):
        img = cv2.resize(img,(608,608))
        img = np.array(img).astype('float32')
        img2 = np.array([img/255]).transpose(0,3,1,2).astype('float32')
        output_names = [output.name for output in self.session.get_outputs()]
        ort_outs = self.session.run(output_names=output_names, input_feed={'im_shape':np.array([[608,608]], dtype=np.float32), 'image': img2, 'scale_factor': np.array([[1,1]], dtype=np.float32)})

        preds = self.postprocess([[ort_outs[0],[[0,ort_outs[0].shape[0]]]]])

        return preds

# 'model/onnx_model/yolov3.onnx'
class TinyOnnxPredictor():
    def __init__(self, model_dir):
        self.session = InferenceSession(model_dir)
        self.labels = ['drill']
        self.num_classes = 1

    def get_io_name(self):
        print([input.name for input in self.session.get_inputs()])
        print([output.name for output in self.session.get_outputs()])

    def get_io_shape(self):
        print([input.shape for input in self.session.get_inputs()])
        print([output.shape for output in self.session.get_outputs()])

    def _postprocess(self, batch_pred):
        infer_result = {}
        if 'bbox' in batch_pred:
            bboxes = batch_pred['bbox']
            bbox_nums = batch_pred['bbox_num']
            det_res = []
            k = 0
            for i in range(len(bbox_nums)):
                det_nums = bbox_nums[i]
                for j in range(det_nums):
                    dt = bboxes[k]
                    k = k + 1
                    num_id, score, xmin, ymin, xmax, ymax = dt.tolist()
                    if int(num_id) < 0:
                        continue
                    category = self.labels[int(num_id)]
                    w = xmax - xmin
                    h = ymax - ymin
                    bbox = [xmin, ymin, w, h]
                    dt_res = {
                        'category_id': int(num_id),
                        'category': category,
                        'bbox': bbox,
                        'score': score
                    }
                    det_res.append(dt_res)
            infer_result['bbox'] = det_res

        if 'mask' in batch_pred:
            masks = batch_pred['mask']
            bboxes = batch_pred['bbox']
            mask_nums = batch_pred['bbox_num']
            seg_res = []
            k = 0
            for i in range(len(mask_nums)):
                det_nums = mask_nums[i]
                for j in range(det_nums):
                    mask = masks[k].astype(np.uint8)
                    score = float(bboxes[k][1])
                    label = int(bboxes[k][0])
                    k = k + 1
                    if label == -1:
                        continue
                    category = self.labels[int(label)]
                    import pycocotools.mask as mask_util
                    rle = mask_util.encode(
                        np.array(
                            mask[:, :, None], order="F", dtype="uint8"))[0]
                    if six.PY3:
                        if 'counts' in rle:
                            rle['counts'] = rle['counts'].decode("utf8")
                    sg_res = {
                        'category_id': int(label),
                        'category': category,
                        'mask': rle,
                        'score': score
                    }
                    seg_res.append(sg_res)
            infer_result['mask'] = seg_res

        bbox_num = batch_pred['bbox_num']
        results = []
        start = 0
        for num in bbox_num:
            end = start + num
            curr_res = infer_result['bbox'][start:end]
            if 'mask' in infer_result:
                mask_res = infer_result['mask'][start:end]
                for box, mask in zip(curr_res, mask_res):
                    box.update(mask)
            results.append(curr_res)
            start = end

        return results

    def postprocess(self, results,
                    batch_size=1):
        """ 对预测结果做后处理

            Args:
                results (list): 预测结果
        """
        net_outputs = {
                'bbox': results[0][0],
                'bbox_num': np.array([results[0][0].shape[0]])
            }
        # print(net_outputs)
        preds = self._postprocess(net_outputs)
        if len(preds) == 1:
                preds = preds[0]

        return preds
    
    def predict(self, img):
        img = cv2.resize(img,(608,608))
        # img = np.array(img).astype('float32')
        img2 = np.array([img]).transpose(0,3,1,2).astype('float32')
        output_names = [output.name for output in self.session.get_outputs()]
        ort_outs = self.session.run(output_names=output_names, input_feed={'im_shape':np.array([[608,608]], dtype=np.float32), 'image': img2, 'scale_factor': np.array([[1,1]], dtype=np.float32)})
        # print(ort_outs)
        preds = self.postprocess([[ort_outs[0],[[0,ort_outs[0].shape[0]]]]])

        return preds

if __name__ == '__main__':
    preditor = TinyOnnxPredictor('model/onnx_model/yolotiny.onnx')

    preditor.get_io_name()

    img = cv2.imread('20211117103812.jpg')



    # predictor2 = pdx.deploy.Predictor('./model/infer_tiny_model')

    x1 = time.time()
    print(preditor.predict(img))
    # print(predictor2.predict(img))
    x2 = time.time()
    print(x2-x1)
    # print(input_names)
    # print(output_names)

    '''
    [{'category_id': 0, 'category': 'drill', 
    'bbox': [215.66433715820312, 410.9061584472656, 684.1917419433594, 256.0635681152344], 
    'score': 0.804372251033783}, 
    {'category_id': 0, 'category': 'drill', 
    'bbox': [438.6444091796875, 367.3341979980469, 624.6326904296875, 236.94180297851562], 
    'score': 0.04904673993587494}, 
    {'category_id': 0, 'category': 'drill', 
    'bbox': [580.9302978515625, 384.5379943847656, 307.218994140625, 120.27001953125], 
    'score': 0.036249738186597824}, 
    {'category_id': 0, 'category': 'drill', 
    'bbox': [58.9052734375, 351.2942810058594, 1196.2515869140625, 378.3089904785156], 
    'score': 0.026698026806116104}, 
    {'category_id': 0, 'category': 'drill', 
    'bbox': [33.044830322265625, 362.7183532714844, 617.3655700683594, 224.98110961914062], 
    'score': 0.02356852777302265}, 
    {'category_id': 0, 'category': 'drill', 
    'bbox': [850.13720703125, 312.94122314453125, 616.672119140625, 226.94677734375], 
    'score': 0.012034815736114979}, 
    {'category_id': 0, 'category': 'drill', 
    'bbox': [674.76171875, 370.36248779296875, 245.9744873046875, 96.3062744140625], 
    'score': 0.006974441464990377}]
    '''