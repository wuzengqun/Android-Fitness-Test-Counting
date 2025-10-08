ncnn-android-yolo11-pose 
---
这是一个示例 ncnn android 项目，可以实现深蹲和引体向上的计数，它依赖于 ncnn 库和 opencv：  
https://github.com/Tencent/ncnn  
https://github.com/nihui/opencv-mobile  
https://github.com/nihui/mesa-turnip-android-driver  

运行效果:  
---
<div align="center">
<video src="https://github.com/user-attachments/assets/76baf2dc-dbb9-453a-b230-7379bdd9776b" controls width="800">
  Your browser does not support the video tag.
</video>  
</div>



如何构建和运行
----
步骤一、  
&emsp;● 安装Android Studio（此项目使用2022.3.1版本）  

步骤二、  
&emsp;● https://mirrors.aliyun.com/github/releases/gradle/gradle-distributions/v6.7.1/  
&emsp;● 下载gradle6.7.1，并放在适当的位置    

步骤三、  
&emsp;● 克隆此项目，使用Android Studio打开   
&emsp;● 打开设置，按照下面这样选择gradle和JDK，可以减少出错概率  
  <img width="838" height="319" alt="c896c25182862b6acf34452e49f816d1" src="https://github.com/user-attachments/assets/253af3ae-0251-46f1-948b-e608366f5d59" />  


步骤四、  
&emsp;● 编译并烧录至手机享受它  








yolo11-pose模型转换指南   
---
步骤一、  
&emsp; https://github.com/ultralytics/ultralytics  
&emsp;● 克隆yolo11官方项目  

步骤二、  
&emsp;● 下载默认权重或使用自己训练的权重文件  
&emsp;● 在自己的环境里运行: yolo export model=yolo11n-pose.pt format=ncnn  
&emsp;● 在同级目录下的yolo11n-pose_ncnn_model文件里可以找到:model.ncnn.bin、model.ncnn.param  
