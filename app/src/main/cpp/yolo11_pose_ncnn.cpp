// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include <android/asset_manager_jni.h>
#include <android/native_window_jni.h>
#include <android/native_window.h>

#include <android/log.h>

#include <jni.h>

#include <string>
#include <vector>

#include <platform.h>
#include <benchmark.h>

#include "yolo11_pose.h"

#include "ndkcamera.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#if __ARM_NEON
#include <arm_neon.h>
#endif // __ARM_NEON

static int select_mode = 4;//选择模式，默认啥都不干
value_result inference_result;//推理层数据回调

void initialize_inference_result() {
    inference_result.count.push_back(0);  // 向 vector 中添加元素初始化
    inference_result.count.push_back(0);
    inference_result.count.push_back(0);
    inference_result.count.push_back(0);
    inference_result.count.push_back(0);

    inference_result.result = 8;
}

static int draw_unsupported(cv::Mat& rgb)
{
    const char text[] = "unsupported";

    int baseLine = 0;
    cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 1.0, 1, &baseLine);

    int y = (rgb.rows - label_size.height) / 2;
    int x = (rgb.cols - label_size.width) / 2;

    cv::rectangle(rgb, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                  cv::Scalar(255, 255, 255), -1);

    cv::putText(rgb, text, cv::Point(x, y + label_size.height),
                cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 0));

    return 0;
}

static int draw_fps(cv::Mat& rgb,const value_result &value)
{
    // resolve moving average
    float avg_fps = 0.f;
    {
        static double t0 = 0.f;
        static float fps_history[10] = {0.f};

        double t1 = ncnn::get_current_time();
        if (t0 == 0.f)
        {
            t0 = t1;
            return 0;
        }

        float fps = 1000.f / (t1 - t0);
        t0 = t1;

        for (int i = 9; i >= 1; i--)
        {
            fps_history[i] = fps_history[i - 1];
        }
        fps_history[0] = fps;

        if (fps_history[9] == 0.f)
        {
            return 0;
        }

        for (int i = 0; i < 10; i++)
        {
            avg_fps += fps_history[i];
        }
        avg_fps /= 10.f;
    }

    char text[32];
    char fps_text[32];
    sprintf(fps_text, "fps=%.2f", avg_fps);

    switch (select_mode) {
        case 0:
            sprintf(text, "pullupcount=%d", value.count[0]);
            break;
        case 1:
            sprintf(text, "pushupcount=%d", value.count[1]);
            break;
        case 2:
            sprintf(text, "crunchcount=%d" ,value.count[2]);
            break;
        case 3:
            sprintf(text, "situpcount=%d" ,value.count[3]);
            break;
        case 4:
            sprintf(text, "wait start=%d" , value.count[4]);
            //sprintf(text, "a=%d,6=%d,10=%d,n=%d,j=%d" ,value.count[4],value.count[5],value.count[6],value.count[7],value.count[8]);
            break;
        default:
            break;
    }

    int baseLine = 0;
    cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

    //右上角显示当前计数
    int y = 0;
    int x = rgb.cols - label_size.width;

    cv::rectangle(rgb, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                  cv::Scalar(255, 255, 255), -1);

    cv::putText(rgb, text, cv::Point(x, y + label_size.height),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));

    // 在左上角显示FPS
    x = 0; // 左上角的x坐标
    y = 0; // 左上角的y坐标
    cv::rectangle(rgb, cv::Rect(cv::Point(x, y), cv::Size(label_size.width+15, label_size.height + baseLine)),
                  cv::Scalar(255, 255, 255), -1); // 绘制白色背景
    cv::putText(rgb, fps_text, cv::Point(x, y + label_size.height),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0)); // 绘制FPS文本

    return 0;
}

static Inference* g_yolo = 0;
static ncnn::Mutex lock;

class MyNdkCamera : public NdkCameraWindow
{
public:
    virtual void on_image_render(cv::Mat& rgb) const;
};

void MyNdkCamera::on_image_render(cv::Mat& rgb) const
{
    {
        ncnn::MutexLockGuard g(lock);

        if (g_yolo)
        {
            std::vector<Pose> objects;
            objects = g_yolo->runInference(rgb);

            inference_result = g_yolo->draw(rgb, objects,select_mode);
        }
        else
        {
            draw_unsupported(rgb);
        }
    }

    draw_fps(rgb,inference_result);
}

static MyNdkCamera* g_camera = 0;

extern "C" {

JNIEXPORT jint JNI_OnLoad(JavaVM* vm, void* reserved)
{
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "JNI_OnLoad");

    initialize_inference_result();

    g_camera = new MyNdkCamera;

    return JNI_VERSION_1_4;
}

JNIEXPORT void JNI_OnUnload(JavaVM* vm, void* reserved)
{
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "JNI_OnUnload");

    {
        ncnn::MutexLockGuard g(lock);

        delete g_yolo;
        g_yolo = 0;
    }

    delete g_camera;
    g_camera = 0;
}

// public native boolean loadModel(AssetManager mgr, int modelid, int cpugpu);
JNIEXPORT jboolean JNICALL Java_com_example_facedetect_activity_Yolo11PoseNcnn_loadModel(JNIEnv* env, jobject thiz, jobject assetManager, jint modelid, jint cpugpu)
{
    if (modelid < 0 || modelid > 6 || cpugpu < 0 || cpugpu > 1)
    {
        return JNI_FALSE;
    }

    AAssetManager* mgr = AAssetManager_fromJava(env, assetManager);

    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "loadModel %p", mgr);

    const char* modeltypes[] =
            {
                    "yolov11n_pose",
                    "yolov11s_pose",
            };

    const int target_sizes[] =
            {
                    640,
                    640,
            };

    const float mean_vals[][3] =
            {
                    {0.0f, 0.0f, 0.0f},
                    {0.0f, 0.0f, 0.0f},
            };

    const float norm_vals[][3] =
            {
                    { 1 / 255.f, 1 / 255.f, 1 / 255.f },
                    { 1 / 255.f, 1 / 255.f, 1 / 255.f },
            };

    const char* modeltype = modeltypes[(int)modelid];
    int target_size = target_sizes[(int)modelid];
    bool use_gpu = (int)cpugpu == 1;

    // reload
    {
        ncnn::MutexLockGuard g(lock);

        if (use_gpu && ncnn::get_gpu_count() == 0)
        {
            // no gpu
            delete g_yolo;
            g_yolo = 0;
        }
        else
        {
            if (!g_yolo)
                g_yolo = new Inference;
                //最后一个参数是是否使用gpu，默认直接给1就是使用gpu，一般来说gpu的表现比cpu好
                g_yolo->loadNcnnNetwork(mgr, modeltype, target_size, mean_vals[(int)modelid], norm_vals[(int)modelid], 0);
        }
    }

    return JNI_TRUE;
}

// public native boolean openCamera(int facing);
JNIEXPORT jboolean JNICALL Java_com_example_facedetect_activity_Yolo11PoseNcnn_openCamera(JNIEnv* env, jobject thiz, jint facing)
{
    if (facing < 0 || facing > 1)
        return JNI_FALSE;

    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "openCamera %d", facing);

    g_camera->open((int)facing);

    return JNI_TRUE;
}

// public native boolean closeCamera();
JNIEXPORT jboolean JNICALL Java_com_example_facedetect_activity_Yolo11PoseNcnn_closeCamera(JNIEnv* env, jobject thiz)
{
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "closeCamera");

    g_camera->close();

    return JNI_TRUE;
}

// public native boolean setOutputWindow(Surface surface);
JNIEXPORT jboolean JNICALL Java_com_example_facedetect_activity_Yolo11PoseNcnn_setOutputWindow(JNIEnv* env, jobject thiz, jobject surface)
{
    ANativeWindow* win = ANativeWindow_fromSurface(env, surface);

    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "setOutputWindow %p", win);

    g_camera->set_window(win);



    return JNI_TRUE;
}

JNIEXPORT jboolean JNICALL Java_com_example_facedetect_activity_Yolo11PoseNcnn_setmode(JNIEnv* env, jobject thiz, jint selectmode)
{
    if (selectmode < 0 || selectmode > 5)
        return JNI_FALSE;

    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "selectmode %d", selectmode);

    select_mode = selectmode;

    return JNI_TRUE;
}

extern "C"
JNIEXPORT jint JNICALL
Java_com_example_facedetect_activity_Yolo11PoseNcnn_getresult(JNIEnv *env, jobject thiz) {
    return int(inference_result.result);// 获取推理层上一个动作的结果，也就是是否标准，由整型数字定义，暂时未完全完成
}

extern "C"
JNIEXPORT jint JNICALL
Java_com_example_facedetect_activity_Yolo11PoseNcnn_getcount(JNIEnv *env, jobject thiz) {
    // 将推理层的计数结果返回。
//    switch (select_mode) {
//        case 0:
//            return int(inference_result.count[0]);
//            break;
//        case 1:
//            return int(inference_result.count[1]);
//            break;
//        case 2:
//            return int(inference_result.count[2]);
//            break;
//        case 3:
//            return int(inference_result.count[3]);
//            break;
//        case 4:
//            return int(inference_result.count[4]);
//            break;
//        default:
//            break;
//    }
}

}
