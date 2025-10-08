package com.example.facedetect.activity;

import android.Manifest;
import android.app.Activity;
import android.content.ContentValues;
import android.content.DialogInterface;
import android.content.pm.PackageManager;
import android.content.Intent;
import android.graphics.PixelFormat;
import android.os.Bundle;
import android.util.Log;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import android.view.View;
import android.view.WindowManager;
import android.widget.AdapterView;
import android.widget.Button;
import android.widget.Spinner;
import android.os.Handler;
import android.widget.LinearLayout;

import androidx.appcompat.app.AlertDialog;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;


import android.widget.TextView;
import android.widget.Toast;

import android.graphics.Bitmap;
import java.util.Calendar;
import android.database.sqlite.SQLiteDatabase;
import android.content.SharedPreferences;
import android.database.Cursor;
import android.view.MotionEvent;

import com.example.facedetect.R;

public class PhysicalTestActivity extends Activity implements SurfaceHolder.Callback {
    public static final int REQUEST_CAMERA = 100;

    private Yolo11PoseNcnn yolo11ncnn = new Yolo11PoseNcnn();
    private int facing = 1;
    private Spinner spinnerModel;
    private Spinner spinnerCPUGPU;
    private int current_model = 0;
    private int current_cpugpu = 0;

    private int countdown = 10; // 倒计时从60秒开始

    private boolean start_flag = false;

    private SurfaceView cameraView;
    int selectmode;  //传入的模式选择


    /** Called when the activity is first created. */
    /** Called when the activity is first created. */
    @Override
    public void onCreate(Bundle savedInstanceState)
    {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.physicaltest);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        cameraView = (SurfaceView) findViewById(R.id.cameraview);

        cameraView.getHolder().setFormat(PixelFormat.RGBA_8888);
        cameraView.getHolder().addCallback(this);

        //按键控制引体计数
        Button buttonSelectPullupcount = (Button) findViewById(R.id.buttonSelectPullup);
        buttonSelectPullupcount.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View arg0) {
                selectmode = 0;
                yolo11ncnn.setmode(selectmode);
            }
        });


        //按键控制深蹲计数
        Button buttonSelectSitupcount = (Button) findViewById(R.id.buttonSelectSitupcount);
        buttonSelectSitupcount.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View arg0) {
                selectmode = 3;
                yolo11ncnn.setmode(selectmode);
            }
        });


        //按键控制停止计数
        Button buttonSelectStopcount = (Button) findViewById(R.id.buttonSelectStopcount);
        buttonSelectStopcount.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View arg0) {
                selectmode = 4;
                yolo11ncnn.setmode(selectmode);
            }
        });

        reload();
    }



    @Override
    protected void onDestroy() {
        super.onDestroy();
    }

    private void reload()
    {
        boolean ret_init = yolo11ncnn.loadModel(getAssets(), current_model, current_cpugpu);
        if (!ret_init)
        {
            Log.e("MainActivity", "yolo11ncnn loadModel failed");
        }
    }

    //下面这三个是摄像头相关的回调函数
    @Override
    public void surfaceChanged(SurfaceHolder holder, int format, int width, int height)
    {
        yolo11ncnn.setOutputWindow(holder.getSurface());
    }


    @Override
    public void surfaceCreated(SurfaceHolder holder)
    {
    }

    @Override
    public void surfaceDestroyed(SurfaceHolder holder)
    {
    }

    @Override
    public void onResume()
    {
        super.onResume();

        if (ContextCompat.checkSelfPermission(getApplicationContext(), Manifest.permission.CAMERA) == PackageManager.PERMISSION_DENIED)
        {
            ActivityCompat.requestPermissions(this, new String[] {Manifest.permission.CAMERA}, REQUEST_CAMERA);
        }

        yolo11ncnn.openCamera(facing);
    }

    @Override
    public void onPause()
    {
        super.onPause();

        yolo11ncnn.closeCamera();
    }
}