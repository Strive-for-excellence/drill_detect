import time
import numpy as np
from onnxruntime import InferenceSession
import cv2


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
        img = cv2.resize(img, (608, 608))
        img = np.array(img).astype('float32')
        img2 = np.array([img / 255]).transpose(0, 3, 1, 2).astype('float32')
        output_names = [output.name for output in self.session.get_outputs()]
        ort_outs = self.session.run(output_names=output_names,
                                    input_feed={'im_shape': np.array([[608, 608]], dtype=np.float32), 'image': img2,
                                                'scale_factor': np.array([[1, 1]], dtype=np.float32)})

        preds = self.postprocess([[ort_outs[0], [[0, ort_outs[0].shape[0]]]]])

        return preds

if __name__ == '__main__':

    preditor = OnnxPredictor('model/onnx_model/yolotiny.onnx')

    preditor.get_io_name()

    img = cv2.imread('20211117103812.jpg')

    x1 = time.time()
    print(preditor.predict(img))
    x2 = time.time()
    print(x2 - x1)
    # print(input_names)
    # print(output_names)

    '''


'''