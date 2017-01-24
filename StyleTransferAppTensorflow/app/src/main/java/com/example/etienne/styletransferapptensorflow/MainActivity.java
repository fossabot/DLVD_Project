package com.example.etienne.styletransferapptensorflow;

import android.app.Activity;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.provider.MediaStore;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

public class MainActivity extends AppCompatActivity {
    Button button;
    Bitmap photo;
    ImageView imageView;
    private static final int CAMERA_REQUEST = 1888;


    private static final String MODEL_FILE = "file:///android_asset/tensorflow_inception_graph.pb";
    private static final String INPUT_NODE = "input:0";
    private static final String OUTPUT_NODE = "output:0";

    TensorFlowInferenceInterface inferenceInterface;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        inferenceInterface = new TensorFlowInferenceInterface();
        inferenceInterface.initializeTensorFlow(getAssets(),MODEL_FILE);

        imageView = (ImageView) this.findViewById(R.id.imageView);

        button = (Button) this.findViewById(R.id.photoButton);
        button.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent cameraIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                startActivityForResult(cameraIntent,CAMERA_REQUEST);
            }
        });
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if(requestCode == CAMERA_REQUEST && resultCode == Activity.RESULT_OK){
            photo = (Bitmap) data.getExtras().get("data");
            if(imageView.getVisibility() == ImageView.INVISIBLE)
                imageView.setVisibility(ImageView.VISIBLE);
            imageView.setImageBitmap(photo);
            photo = BitmapFactory.decodeResource(getResources(),R.mipmap.ic_launcher_256);
            int[] intValues = new int[photo.getHeight()*photo.getWidth()];
            photo.getPixels(intValues, 0, photo.getWidth(), 0, 0, photo.getWidth(), photo.getHeight());
            float[] floatValues = new float[intValues.length * 3];
            for (int i = 0; i < intValues.length; ++i) {
                final int val = intValues[i];
                floatValues[i * 3] = ((val >> 16) & 0xFF) / 255.0f;
                floatValues[i * 3 + 1] = ((val >> 8) & 0xFF) / 255.0f;
                floatValues[i * 3 + 2] = (val & 0xFF) / 255.0f;
            }

            Log.d("floatvaluesisze",String.valueOf(floatValues.length));

            Log.d("Checkpoint","Get to network");

            inferenceInterface.fillNodeFloat(
                    INPUT_NODE, new int[] {1, photo.getWidth(), photo.getHeight(), 3}, floatValues);

            Log.d("Checkpoint","Created Input Node");


            inferenceInterface.runInference(new String[] {OUTPUT_NODE});

            Log.d("Checkpoint","Ran inference");
            inferenceInterface.readNodeFloat(OUTPUT_NODE, floatValues);

            Log.d("Checkpoint","Read Output of Network");

            for (int i = 0; i < intValues.length; ++i) {
                intValues[i] =
                        0xFF000000
                                | (((int) (floatValues[i * 3] * 255)) << 16)
                                | (((int) (floatValues[i * 3 + 1] * 255)) << 8)
                                | ((int) (floatValues[i * 3 + 2] * 255));
            }
            Bitmap toDraw = photo.copy(photo.getConfig(),true);
            toDraw.setPixels(intValues, 0, photo.getWidth(), 0, 0, photo.getWidth(), photo.getHeight());
            imageView.setImageBitmap(toDraw);
        }
    }
}
