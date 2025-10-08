//
// Created by wzq on 2024/10/21.
//

#include "yolo11_pose.h"
#include <cpu.h>
#include <iostream>
#include <vector>

value_result inference_result1;

const int MAX_STRIDE = 32;
const int COCO_POSE_POINT_NUM = 17;

const std::vector<std::vector<unsigned int>> KPS_COLORS =
        { {0,   255, 0}, {0,   255, 0},  {0,   255, 0}, {0,   255, 0},
          {0,   255, 0},  {255, 128, 0},  {255, 128, 0}, {255, 128, 0},
          {255, 128, 0},  {255, 128, 0},  {255, 128, 0}, {51,  153, 255},
          {51,  153, 255},{51,  153, 255},{51,  153, 255},{51,  153, 255},
          {51,  153, 255}};

const std::vector<std::vector<unsigned int>> SKELETON =
        { {16, 14},  {14, 12},  {17, 15},  {15, 13},   {12, 13}, {6,  12},
          {7,  13},  {6,  7},   {6,  8},   {7,  9},   {8,  10},  {9,  11},
          {2,  3}, {1,  2},  {1,  3},  {2,  4},  {3,  5},   {4,  6},  {5,  7} };

const std::vector<std::vector<unsigned int>> LIMB_COLORS =
        { {51,  153, 255}, {51,  153, 255},   {51,  153, 255},
          {51,  153, 255}, {255, 51,  255},   {255, 51,  255},
          {255, 51,  255}, {255, 128, 0},     {255, 128, 0},
          {255, 128, 0},   {255, 128, 0},     {255, 128, 0},
          {0,   255, 0},   {0,   255, 0},     {0,   255, 0},
          {0,   255, 0},   {0,   255, 0},     {0,   255, 0},
          {0,   255, 0} };

typedef struct {
    cv::Rect box;
    float confidence;
    int index;
}BBOX;

bool cmp_score(BBOX box1, BBOX box2) {
    return box1.confidence > box2.confidence;
}


static float get_iou_value(cv::Rect rect1, cv::Rect rect2)
{
    int xx1, yy1, xx2, yy2;

    xx1 = std::max(rect1.x, rect2.x);
    yy1 = std::max(rect1.y, rect2.y);
    xx2 = std::min(rect1.x + rect1.width - 1, rect2.x + rect2.width - 1);
    yy2 = std::min(rect1.y + rect1.height - 1, rect2.y + rect2.height - 1);

    int insection_width, insection_height;
    insection_width = std::max(0, xx2 - xx1 + 1);
    insection_height = std::max(0, yy2 - yy1 + 1);

    float insection_area, union_area, iou;
    insection_area = float(insection_width) * insection_height;
    union_area = float(rect1.width * rect1.height + rect2.width * rect2.height - insection_area);
    iou = insection_area / union_area;
    return iou;
}

void my_nms_boxes(std::vector<cv::Rect>& boxes, std::vector<float>& confidences, float confThreshold, float nmsThreshold, std::vector<int>& indices)
{
    BBOX bbox;
    std::vector<BBOX> bboxes;
    int i, j;
    for (i = 0; i < boxes.size(); i++)
    {
        bbox.box = boxes[i];
        bbox.confidence = confidences[i];
        bbox.index = i;
        bboxes.push_back(bbox);
    }
    sort(bboxes.begin(), bboxes.end(), cmp_score);

    int updated_size = bboxes.size();
    for (i = 0; i < updated_size; i++)
    {
        if (bboxes[i].confidence < confThreshold)
            continue;
        indices.push_back(bboxes[i].index);
        for (j = i + 1; j < updated_size; j++)
        {
            float iou = get_iou_value(bboxes[i].box, bboxes[j].box);
            if (iou > nmsThreshold)
            {
                bboxes.erase(bboxes.begin() + j);
                j=j-1;
                updated_size = bboxes.size();
            }
        }
    }
}

//计算两点角度
static int calculateangle(int x1, int y1, int x2, int y2) {
    // 计算两点之间的差值
    double dx = static_cast<double>(x2 - x1); // 使用 double 保持精度
    double dy = static_cast<double>(y2 - y1);

    // 使用atan2计算角度，结果为弧度
    double angle_radians = atan2(dy, dx);

    // 将弧度转换为度
    double angle_degrees = angle_radians * (180.0 / M_PI);

    // 确保角度为正值
    if (angle_degrees < 0) {
        angle_degrees += 360.0;
    }

    // 将结果转换为 int
    return static_cast<int>(angle_degrees + 0.5); // 四舍五入
}

//函数：检查深蹲时角度是否合适
static bool is_situp_bodyangle(float angle1,float angle2,float angle3,float angle_threshold,int direction)
{
    if (direction == 1)//右边视角
    {
        return ((angle1 >= 95) && (angle1 <= 125)) && ((angle2 >= 175) && (angle2 <= 205)) && ((angle3 >= 112) && (angle3 <= 130));
    }
    else//左边视角，这里还没有实现，需要改一改角度范围
    {
        return ((angle1 >= 95) && (angle1 <= 125)) && ((angle2 >= 175) && (angle2 <= 205)) && ((angle3 >= 112) && (angle3 <= 130));
    }
}

//函数：计算深蹲是否标准
static bool is_situp_standard(const float* preds,float height_threshold,int direction)
{
    static bool lastSitUpResult = false;
    static bool currentResult = false;
    if (direction == 1)//右边视角
    {
        float angle1 = calculateangle(preds[6],preds[6+17],preds[12],preds[12+17]);
        float angle2 = calculateangle(preds[14],preds[14+17],preds[12],preds[12+17]);
        float angle3 = calculateangle(preds[14],preds[14+17],preds[16],preds[16+17]);
        currentResult = is_situp_bodyangle(angle1,angle2,angle3,0,1);
        if(lastSitUpResult == false && currentResult)
        {
            lastSitUpResult = true;
            return true;
        }
        else if(lastSitUpResult == true && currentResult)
        {
            return false;
        }
        else if((!currentResult) && (angle1 < 95))  //背挺直了才算做完一个
        {
            lastSitUpResult = false;
            return false;
        }
        else
        {
            return false;
        }
    }
    if (direction == 0)//左边视角
    {
        float angle1 = calculateangle(preds[5],preds[5+17],preds[11],preds[11+17]);
        float angle2 = calculateangle(preds[13],preds[13+17],preds[11],preds[11+17]);
        float angle3 = calculateangle(preds[13],preds[13+17],preds[15],preds[15+17]);
        currentResult = is_situp_bodyangle(angle1,angle2,angle3,0,0);
        if(lastSitUpResult == false && currentResult)
        {
            lastSitUpResult = true;
            return true;
        }
        else if(lastSitUpResult == true && currentResult)
        {
            return false;
        }
        else if((!currentResult) && (angle1 < 95))  //背挺直了才算做完一个
        {
            lastSitUpResult = false;
            return false;
        }
        else
        {
            return false;
        }
    }

}

static bool ispullupStandard(const float* preds) {
    static enum {
        STATE_DOWN,      // 手臂伸直状态
        STATE_UP,        // 手臂弯曲状态
        STATE_COMPLETED  // 完成一次计数
    } state = STATE_DOWN;

    int angle = calculateangle(preds[6], preds[6 + 17], preds[8], preds[8 + 17]);

    switch (state) {
        case STATE_DOWN:
            // 从伸直状态开始弯曲（下巴过杠）
            if (angle < 160) {  // 调整为您需要的角度阈值
                state = STATE_UP;
            }
            break;

        case STATE_UP:
            // 从弯曲状态回到伸直（完成一次）
            if (angle > 220) {  // 调整为您需要的角度阈值
                state = STATE_COMPLETED;
                return true;    // 完成计数
            }
            break;

        case STATE_COMPLETED:
            // 重置状态，准备下一次计数
            if (angle > 223) {
                state = STATE_DOWN;
            }
            break;
    }

    return false;
}


//static bool ispullupStandard(const float* preds) {
//    static bool flag_yuandian = 0;
//    static bool flag_shangla = 0;
//    static bool flag_yicijieshu = 0;
//
//    static int angle = 0;
//    static int flag = 0;
//
//    static std::vector<int> angle_data;
//
//    static bool up = 0;
//    static bool down = 0;
//
//    static int i = 0;
//
//    angle = calculateangle(preds[6],preds[6+17],preds[8],preds[8+17]);
//    if(angle >= 210)
//    {
//        flag_yuandian = 1;
//    }
//    if(angle <= 190)
//    {
//        flag_shangla = 1;
//    }
//    if(angle >= 220 && flag_shangla == 1)
//    {
//        flag_yicijieshu = 1;
//        flag_shangla = 0;
//        flag = 1;
//    }
//    if(flag_yuandian)
//    {
//        inference_result1.result = 8;
//        if(flag_yicijieshu == 0)
//        {
//            angle_data.push_back(angle);
//        }
//        else {
//            if (flag) {
//                //下降到一定高度之后就停止上面的push，改为手动补帧，一共补5次，需要根据不同设备的帧率改这里
//                if (i < 5) {
//                    angle_data.push_back(angle);
//                    i++;
//                } else {
//                    i = 0;
//                    flag = 0;
//                }
//
//            } else {
//                //这里面找出最大最小值即是一次完整的引体向上，之后可以将一次完整运动的数据送入神经网络或者机器学习算法中进行判断等
//                int min_angle = *std::min_element(angle_data.begin(), angle_data.end());
//                int max_angle = *std::max_element(angle_data.begin(), angle_data.end());
//
//                if (min_angle <= 140) {
//                    //满足上拉条件
//                    up = 1;
//                } else {
//                    inference_result1.result = 1;//result = 1  表示上一个引体上拉时下巴未过杆，上一个不计数
//                    up = 0;
//                }
//                if (max_angle >= 223) {
//                    //满足下放条件
//                    down = 1;
//                } else {
//                    if(up == 0)
//                    {
//                        inference_result1.result = 2;//result = 2  表示下巴未过杆并且手臂未伸直
//                    }
//                    else
//                    {
//                        inference_result1.result = 0;//result = 0  表示上一个引体下落时手臂未伸直，上一个不计数
//                    }
//                    down = 0;
//                }
//                flag_yicijieshu = 0;
//                angle_data.clear();
//                if (up == 1 && down == 1) {
//                    up = 0;
//                    down = 0;
//                    inference_result1.result = 8;
//                    return true;
//                }
//            }
//        }
//    }
//    return false;
//}

Inference::Inference(){
    blob_pool_allocator.set_size_compare_ratio(0.f);
    workspace_pool_allocator.set_size_compare_ratio(0.f);
}

int Inference::loadNcnnNetwork(AAssetManager* mgr, const char* modeltype , const int& modelInputShape, const float* meanVals, const float* normVals, bool useGpu) {
    if (!mgr) {
        __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "Invalid AssetManager");
        return -1;
    }

    modelShape = modelInputShape;
    gpuEnabled = useGpu;

    net.clear();
    blob_pool_allocator.clear();
    workspace_pool_allocator.clear();

    ncnn::set_cpu_powersave(2);
    ncnn::set_omp_num_threads(ncnn::get_big_cpu_count());

    net.opt = ncnn::Option();
    net.opt.use_vulkan_compute = useGpu;
    net.opt.num_threads = ncnn::get_big_cpu_count();
    net.opt.blob_allocator = &blob_pool_allocator;
    net.opt.workspace_allocator = &workspace_pool_allocator;

    char parampath[256];
    char modelpath[256];
    sprintf(parampath, "yolov11n_pose.param");
    sprintf(modelpath, "yolov11n_pose.bin");

//    if (!net.load_param(mgr, parampath) || !net.load_model(mgr, modelpath)) {
//        //LOGE("Failed to load model");
//        __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "Failed to load model");
//        return -1;
//    }
    net.load_param(mgr, parampath);
    net.load_model(mgr, modelpath);

    inference_result1.count.assign({0, 0, 0, 0, 0});
    inference_result1.result = 8;// 默认给8，当引体下拉未伸直时置1，其它等待新设备调试

    memcpy(this->meanVals, meanVals, 3 * sizeof(float));
    memcpy(this->normVals, normVals, 3 * sizeof(float));
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "yolov11n-pose.bin load success");

    return 0;
}

std::vector<Pose> Inference::runInference(const cv::Mat &input)
{
    cv::Mat modelInput = input;
    int imgWidth = modelInput.cols;
    int imgHeight = modelInput.rows;

    int w = imgWidth;
    int h = imgHeight;
    float scale = 1.f;
    if (w > h) {
        scale = (float)modelShape / w;
        w = modelShape;
        h = (int)(h * scale);
    }
    else {
        scale = (float)modelShape / h;
        h = modelShape;
        w = (int)(w * scale);
    }

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(modelInput.data, ncnn::Mat::PIXEL_BGR2RGB, imgWidth, imgHeight, w, h);

    int wpad = (modelShape + MAX_STRIDE - 1) / MAX_STRIDE * MAX_STRIDE - w;
    int hpad = (modelShape + MAX_STRIDE - 1) / MAX_STRIDE * MAX_STRIDE - h;

    int top = hpad / 2;
    int bottom = hpad - hpad / 2;
    int left = wpad / 2;
    int right = wpad - wpad / 2;

    ncnn::Mat in_pad;
    ncnn::copy_make_border(in, in_pad, top, bottom, left, right, ncnn::BORDER_CONSTANT, 114.f);

    in_pad.substract_mean_normalize(meanVals, normVals);

    ncnn::Extractor ex = net.create_extractor();

    ex.input("in0", in_pad);

    ncnn::Mat out;
    ex.extract("out0", out);

    cv::Mat output(out.h, out.w, CV_32FC1, out.data);
    cv::transpose(output, output);
    std::cout<<output.rows << output.cols << output.channels()<<std::endl;
    float* data = (float*)output.data;

    std::vector<float>  confidences;
    std::vector<cv::Rect> boxes;
    std::vector<std::vector<float>> keyPoints;

    int rows = output.rows;
    int dimensions = output.cols;
    for (int row = 0; row < rows; row++) {
        float score = *(data + 4);
        if (score > modelScoreThreshold) {
            confidences.push_back(score);

            float x = data[0];
            float y = data[1];
            float w = data[2];
            float h = data[3];

            int left = int((x - 0.5 * w));
            int top = int((y - 0.5 * h));

            int width = int(w);
            int height = int(h);

            boxes.push_back(cv::Rect(left, top, width, height));

            std::vector<float> kps((data + 5), data + 5+COCO_POSE_POINT_NUM * 3);
            keyPoints.push_back(kps);
        }
        data += dimensions;
    }
    std::vector<int> nms_result;
    my_nms_boxes(boxes, confidences, modelScoreThreshold, modelNMSThreshold, nms_result);


    std::vector<Pose> poses;
    for (int i = 0; i < nms_result.size(); ++i) {
        int idx = nms_result[i];

        float confidence = confidences[idx];

        cv::Rect box = { int(((boxes[idx].x - int(wpad / 2)) / scale)),
                         int(((boxes[idx].y - int(hpad / 2))) / scale),
                         int(boxes[idx].width / scale),
                         int(boxes[idx].height / scale) };
        std::vector<float> kps;
        for (int j = 0; j < keyPoints[idx].size()/3; j++) {
            kps.push_back((keyPoints[idx][3 * j + 0] - int(wpad / 2)) / scale);
            kps.push_back((keyPoints[idx][3 * j + 1] - int(hpad / 2)) / scale);
            kps.push_back(keyPoints[idx][3 * j + 2]);
        };
        Pose pose;
        pose.box = box;
        pose.confidence = confidence;
        pose.kps = kps; //{ confidence, box, kps };
        poses.push_back(pose);
    }


    return poses;
}

struct value_result Inference::draw(cv::Mat& rgb, const std::vector<Pose>& objects,int selectmode) {
    static bool isStandardPullup = 0;
    static bool isStandardPushup = 0;
    static int id = 0;

    cv::Mat res = rgb;
    for (auto& obj : objects) {
        //cv::rectangle(res, obj.box, { 0, 0, 255 }, 2);

        int x = (int)obj.box.x;
        int y = (int)obj.box.y + 1;

        if (y > res.rows)
            y = res.rows;

        auto& kps = obj.kps;
        for (int k = 0; k < COCO_POSE_POINT_NUM + 2; k++) {
            if (k < COCO_POSE_POINT_NUM) {
                int kps_x = (int)std::round(kps[k * 3]);
                int kps_y = (int)std::round(kps[k * 3 + 1]);
                float kps_s = kps[k * 3 + 2];
                if (kps_s > 0.4f) {
                    cv::Scalar kps_color = cv::Scalar(KPS_COLORS[k][0], KPS_COLORS[k][1], KPS_COLORS[k][2]);
                    cv::circle(res, { kps_x, kps_y }, 5, kps_color, -1);
                }
            }
            auto& ske = SKELETON[k];
            int pos1_x = (int)std::round(kps[(ske[0] - 1) * 3]);
            int pos1_y = (int)std::round(kps[(ske[0] - 1) * 3 + 1]);

            int pos2_x = (int)std::round(kps[(ske[1] - 1) * 3]);
            int pos2_y = (int)std::round(kps[(ske[1] - 1) * 3 + 1]);

            float pos1_s = kps[(ske[0] - 1) * 3 + 2];
            float pos2_s = kps[(ske[1] - 1) * 3 + 2];

            if (pos1_s > 0.5f && pos2_s > 0.5f) {
                cv::Scalar limb_color = cv::Scalar(LIMB_COLORS[k][0], LIMB_COLORS[k][1], LIMB_COLORS[k][2]);
                cv::line(res, { pos1_x, pos1_y }, { pos2_x, pos2_y }, limb_color, 2);
            }
        }
    }

    if(objects.empty())
    {
        //如果没检测到目标啥也不干
    }
    else {
        float *preds = new float[objects.size() * 17 * 2];
        int x_index = 0; // 用于追踪preds中x坐标的索引位置,前17个填充为x
        int y_index = objects.size() * 17; // 用于追踪preds中y坐标的索引位置，后17个填充为y

        for (const auto &obj: objects) {
            for (size_t i = 0; i < obj.kps.size(); i += 3) {
                preds[x_index++] = obj.kps[i];   // 填充x坐标
                preds[y_index++] = obj.kps[i + 1]; // 填充y坐标
                // 跳过置信度
            }
        }

        //选择模式判断
        switch (selectmode) {
            case 0:
                isStandardPullup = ispullupStandard(preds);   //引体
                if (isStandardPullup) {
                    inference_result1.count[0]++;
                }
                break;
            case 1:
                if(preds[0] > preds[16])  //头朝右侧      //俯卧撑
                {
//                    if(is_body_straight(preds,1,1))
//                    {
//                        if(isStandardPushup = is_pushup_standard(preds,30,1))
//                        {
//                            inference_result1.count[1]++;
//                        }
//                    }
                }
                else if(preds[0] < preds[16])
                {
//                    if(is_body_straight(preds,1,0))
//                    {
//                        if(isStandardPushup = is_pushup_standard(preds,25,0))
//                        {
//                            inference_result1.count[1]++;
//                        }
//                    }
                }
                break;
            case 2:
                if(preds[0] > preds[16])//头朝右      //仰卧起坐
                {
//                    if(is_crunch_standard(preds,0,0))
//                    {
//                        inference_result1.count[2]++;
//                    }
                }
                else if(preds[0] < preds[16])
                {
//                    if(is_crunch_standard(preds,0,1))
//                    {
//                        inference_result1.count[2]++;
//                    }
                }
                break;
            case 3:

                if(is_situp_standard(preds,0,1))
                {
                    inference_result1.count[3]++;
                }
                break;
            case 4:
                //如果使用MLP来分类的话需要用到下面这些来制作数据集
                inference_result1.count[4] = calculateangle(preds[6],preds[6+17],preds[8],preds[8+17]);//左边手腕、手肘、肩膀的角度
                break;
            default:
                break;
        }
        delete[] preds;
    }

    for (int i = 0; i < 5; ++i) {
        if (i != selectmode)
            inference_result1.count[i] = 0;
    }
    return inference_result1;


}
