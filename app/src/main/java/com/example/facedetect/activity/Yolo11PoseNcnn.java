package com.example.facedetect.activity;

import android.content.res.AssetManager;
import android.view.Surface;

public class Yolo11PoseNcnn
{
    public native boolean loadModel(AssetManager mgr, int modelid, int cpugpu);
    public native boolean openCamera(int facing);
    public native boolean closeCamera();
    public native boolean setOutputWindow(Surface surface);

    public native boolean setmode(int select_mode);

    public native int getresult();

    public native int getcount();

    static {
        System.loadLibrary("native-lib");
    }
}