package com.example.etienne.styletransferapptensorflow;

import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.util.Log;

/**
 * Created by etienne on 25.01.17.
 */
public class Utils {

    public static Bitmap cropBitmapSquare(Bitmap bm){
        Bitmap dstBmp;
        if (bm.getWidth() >= bm.getHeight()){

            dstBmp = Bitmap.createBitmap(
                    bm,
                    bm.getWidth()/2 - bm.getHeight()/2,
                    0,
                    bm.getHeight(),
                    bm.getHeight()
            );

        }else{

            dstBmp = Bitmap.createBitmap(
                    bm,
                    0,
                    bm.getHeight()/2 - bm.getWidth()/2,
                    bm.getWidth(),
                    bm.getWidth()
            );
        }
        return dstBmp;
    }

    public static Bitmap resizeBitmapFitXY(int width, int height, Bitmap bitmap){
        Bitmap background = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
        float originalWidth = bitmap.getWidth(), originalHeight = bitmap.getHeight();
        Canvas canvas = new Canvas(background);
        float scale, xTranslation = 0.0f, yTranslation = 0.0f;
        if (originalWidth > originalHeight) {
            scale = height/originalHeight;
            xTranslation = (width - originalWidth * scale)/2.0f;
        }
        else {
            scale = width / originalWidth;
            yTranslation = (height - originalHeight * scale)/2.0f;
        }
        Matrix transformation = new Matrix();
        transformation.postTranslate(xTranslation, yTranslation);
        transformation.preScale(scale, scale);
        Paint paint = new Paint();
        paint.setFilterBitmap(true);
        canvas.drawBitmap(bitmap, transformation, paint);
        return background;
    }
}
