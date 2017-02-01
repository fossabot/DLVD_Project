package com.example.etienne.styletransferapptensorflow;


import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.os.AsyncTask;
import android.util.Log;
import android.widget.TextView;
import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

/**
 * Created by etienne on 25.01.17.
 */
public class Model {
    private static String MODEL_FILE;
    private static final String INPUT_NODE = "ph_input_image";
    private static final String OUTPUT_NODE = "output";
    private static final int DESIRED_HEIGHT = 800;
    private static final int DESIRED_WIDTH = 800;
    private AssetManager am;

    TensorFlowInferenceInterface tensorFlowInferenceInterface;

    public Model(String name, AssetManager am){
        MODEL_FILE ="file:///android_asset/" + name;
        tensorFlowInferenceInterface = new TensorFlowInferenceInterface();
        this.am = am;
    }

    public void initializeStyle(){
        tensorFlowInferenceInterface.initializeTensorFlow(am,MODEL_FILE);
    }

    public void closeStyle(){
        tensorFlowInferenceInterface.close();
    }


    public Bitmap applyModel(Bitmap bm){
        int[] intValues = new int[bm.getHeight()*bm.getWidth()];
        bm.getPixels(intValues, 0, bm.getWidth(), 0, 0, bm.getWidth(), bm.getHeight());
        float[] floatValues = new float[intValues.length * 3];
        for (int i = 0; i < intValues.length; ++i) {
            final int val = intValues[i];
            floatValues[i * 3] = ((val >> 16) & 0xFF) / 255.0f;
            floatValues[i * 3 + 1] = ((val >> 8) & 0xFF) / 255.0f;
            floatValues[i * 3 + 2] = (val & 0xFF) / 255.0f;
        }


        Log.d("Checkpoint","Get to network");


        tensorFlowInferenceInterface.fillNodeFloat(
                INPUT_NODE, new int[] {1, bm.getWidth(), bm.getHeight(), 3}, floatValues);
        Log.d("Checkpoint","Created Input Node");


        tensorFlowInferenceInterface.runInference(new String[] {OUTPUT_NODE});
        Log.d("Checkpoint","Ran inference");

        float[] outputValues = new float[DESIRED_WIDTH * DESIRED_HEIGHT * 3];


        tensorFlowInferenceInterface.readNodeFloat(OUTPUT_NODE, outputValues);
        Log.d("Checkpoint","Read Output of Network");


        Bitmap toDraw = Bitmap.createBitmap(DESIRED_WIDTH,DESIRED_HEIGHT, Bitmap.Config.ARGB_8888);
        int[] colors = new int[DESIRED_WIDTH*DESIRED_HEIGHT];
        for (int i = 0; i < colors.length; i ++ ) {
            colors[i] =
                    0xFF000000
                            | (((int) (outputValues[i * 3] * 255)) << 16)
                            | (((int) (outputValues[i * 3 + 1] * 255)) << 8)
                            | ((int) (outputValues[i * 3 + 2] * 255));
        }
        toDraw.setPixels(colors, 0, toDraw.getWidth(), 0, 0, toDraw.getWidth(), toDraw.getHeight());
        return toDraw;

    }

}
