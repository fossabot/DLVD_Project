package com.example.etienne.styletransferapptensorflow;

import android.Manifest;
import android.app.Activity;
import android.content.Intent;

import android.content.pm.PackageManager;
import android.graphics.*;
import android.net.Uri;
import android.os.*;
import android.provider.MediaStore;


import android.support.v4.app.ActivityCompat;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;

import java.io.File;

public class MainActivity extends AppCompatActivity {
    private Button button;
    private Bitmap result;
    private Model currentModel;
    private ImageView imageView;

    File myFilesDir;

    private static final int CAMERA_REQUEST = 1888;

    private static final int REQUEST_CAMERA_PERMISSION = 200;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        currentModel = new Model("gen_export.pb",getAssets());

        imageView = (ImageView) this.findViewById(R.id.imageView);
        myFilesDir = new File(Environment.getExternalStorageDirectory().getAbsolutePath() + "/Fotos");
        myFilesDir.mkdirs();


        button = (Button) this.findViewById(R.id.photoButton);
        button.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                 Intent cameraIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                 cameraIntent.putExtra(MediaStore.EXTRA_OUTPUT, Uri.fromFile(new File(myFilesDir.toString()+"/temp.jpg")));
                 startActivityForResult(cameraIntent,CAMERA_REQUEST);
            }
        });


        if (ActivityCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED && ActivityCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(MainActivity.this, new String[]{Manifest.permission.CAMERA, Manifest.permission.WRITE_EXTERNAL_STORAGE}, REQUEST_CAMERA_PERMISSION);
            return;
        }
    }

    @Override
    protected void onResume() {
        super.onResume();
    }

    @Override
    protected void onPause() {
        super.onPause();
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if(requestCode == CAMERA_REQUEST && resultCode == Activity.RESULT_OK){
            result = BitmapFactory.decodeFile(myFilesDir.toString() + "/temp.jpg");
            result = Utils.cropBitmapSquare(result);
            Log.d("size",String.valueOf(result.getWidth() + "   " + result.getHeight()));
            //result = BitmapFactory.decodeResource(getResources(),R.drawable.cat_604);
            result = Bitmap.createScaledBitmap(result,304,304,false);
            result = currentModel.applyModel(result);
            result = Bitmap.createScaledBitmap(result,400,400,false);
            imageView.setImageBitmap(result);
        }
    }

}
