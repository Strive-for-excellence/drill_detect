import cv2
import numpy
import paddlex as pdx
import time
import tqdm
import os
# import pandas
from  matplotlib import pyplot,rcParams
import shutil
from onnx_run import OnnxPredictor
from scipy import signal
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
rcParams['font.family']='SimHei'
# 导入模型
print("* Loading model...")
predictor = pdx.deploy.Predictor('./model/inference_model')
# predictor = pdx.deploy.Predictor('./model/inference_model',use_gpu=False)
# predictor = OnnxPredictor('model/onnx_model/yolov3.onnx')
predictor = pdx.deploy.Predictor('./model/inference_model')
print("* Model loaded")
def signal_process(length_list, fast_foward_speed=1.0):
    # 峰值和谷值的阈值
    max_height = 1000  # 峰值阈值
    min_height = 300  # 峰谷阈值
    medfit_width = 5  # 中位数滤波器窗口宽度
    # cut_off_frequcy = 0.5 / fast_foward_speed  # 截止频率
    distance = int(240 / fast_foward_speed)  # 峰距

    length_list2 = length_list + [0 for i in range(100)]
    # 中值滤波
    length_list2 = signal.medfilt(length_list2, medfit_width)
    # 巴特沃斯滤波 去除
    length_list3 = length_list2
    x = numpy.array(length_list3)
    peaks_low, _ = signal.find_peaks(-x, height=-min_height, distance=distance)
    peaks_high, _ = signal.find_peaks(x, height=max_height, distance=distance)
    drill_num = 0
    if len(peaks_low) != 0 and len(peaks_high) != 0:
        drill_num = len(peaks_low)
        if peaks_high[0] > peaks_low[0]:
            drill_num -= 1
            peaks_low = peaks_low[1:]
    # drill_num_list = [i for i in range(len(x))]
    drill_num_list = numpy.zeros_like(length_list3)
    for i in range(len(length_list3)):
        for j in range(len(peaks_low)):
            if i >= peaks_low[j]:
                drill_num_list[i] = j + 1

    for i in range(len(length_list3)):
        length_list3[i] = abs(length_list3[i])
    return drill_num, list(length_list3[0:len(length_list)]), list(drill_num_list),list(peaks_low)

def classify_process():
    fast_foward_speed = 8.0
    # workspace = os.getcwd()
    workspace = './data'
    for vf in os.listdir(workspace):
        if vf.find('.mp4') == -1:
            continue
        print('正在处理 '+ vf)
        output_dir = os.path.join(workspace,vf.replace('.mp4',''))
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)
        res_t = []
        try:
            cap = cv2.VideoCapture(os.path.join(workspace,vf))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            print('fps = ',fps)
            total_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            for i in tqdm.tqdm(range(int(total_frame))):
                CUR_PROGRESS = i / total_frame
                ret = cap.grab()
                if not ret:
                    print("ERROR GRAB FRAME")
                    break
                if i % (fps * fast_foward_speed) == 0:
                    ret, frame = cap.retrieve()
                    if ret:
                        T1 = time.time()
                        new_frame = cv2.resize(frame,dsize=(1920,1080))
                        res = predictor.predict(new_frame)
                        T2 = time.time()
                        # print('程序运行时间:%s秒' % ((T2 - T1) ))
                        # print(res)
                        if len(res) > 0:
                            res_t.append(res[0])
                        else:
                            break
                    else:
                        print("ERROR RETRIEVE FRAME")
                        break

        except:
            print('不能处理 '+vf)
            continue
        results = res_t
        drill_length = []
        for line in results:
            drill_length.append(pow(pow(line['bbox'][2], 2) + pow(line['bbox'][3], 2), 0.5))
        print(drill_length)
        drill_num, drill_length, drill_num_list, peaks_low = signal_process(drill_length, fast_foward_speed=fast_foward_speed)


        # with open(vf.replace('.mp4','data.txt'),'w') as f:
        #     for i in range(len(drill_length)):
        #         f.write(str(drill_length[i])+'\n')
        x_vals = list(range(len(drill_length)))

        # 输出部分
        print('drill_num = ', drill_num)

        x_vals = [i * fast_foward_speed for i in range(len(drill_length))]
        end_location = []
        for location in peaks_low:
            if location < len(x_vals):
                end_location.append(location)
        with open(os.path.join(output_dir,'次数.txt'),'w') as f:
            f.write('更换钻头次数: '+ str(drill_num)+"\n")
            f.write('更换钻头时间: ')
            for item in end_location:
                f.write(time.strftime("%H时%M分",time.gmtime(item*fast_foward_speed))+" ")
            f.write("\n")
        # 画图
        pyplot.plot(x_vals,drill_length)
        pyplot.plot([x_vals[i] for i in end_location],[drill_length[i] for i in end_location],'x')
        # pyplot.show()
        pyplot.savefig(os.path.join(output_dir,'折线图.jpg'))
        pyplot.close()
if __name__ == '__main__':

    classify_process()
