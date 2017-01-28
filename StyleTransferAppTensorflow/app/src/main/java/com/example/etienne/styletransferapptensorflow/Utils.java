package com.example.etienne.styletransferapptensorflow;

import android.graphics.Bitmap;
import android.util.Log;

/**
 * Created by etienne on 25.01.17.
 */
public class Utils {

    public static Bitmap cropBitmapSquare(Bitmap bm){
        int width = bm.getWidth();
        int height = bm.getHeight();

        Log.d("Size",String.valueOf(width + "|" + height));

        int shortEdge = width <  height ? width : height;

        Bitmap newMap = Bitmap.createBitmap(shortEdge,shortEdge, Bitmap.Config.ARGB_8888);

        if (shortEdge == width){
            int padding = (height - width)/2;
            for(int i = padding; i < padding + shortEdge; i++){
                for(int j = 0; j < shortEdge; j++){
                    newMap.setPixel(j,i - padding,bm.getPixel(j,i));
                }
            }
        }else if(shortEdge == height){
            int padding = (width - height)/2;
            for(int i = 0; i < shortEdge; i++){
                for(int j = padding; j < padding + shortEdge; j++){
                    newMap.setPixel(j - padding,i,bm.getPixel(j,i));
                }
            }
        }else{
            return bm;
        }
        Log.d("Cropped Size",String.valueOf(newMap.getWidth()+"|"+newMap.getHeight()));
        return newMap;

    }
}
