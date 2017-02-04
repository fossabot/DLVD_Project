package com.example.etienne.styletransferapptensorflow;

import android.graphics.Bitmap;

/**
 * Created by etienne on 03.02.17.
 */
public class PreviewListItem {
    private String styleName;
    private Bitmap styleImage;

    public PreviewListItem(String styleName, Bitmap styleImage) {
        this.styleName = styleName;
        this.styleImage = styleImage;
    }

    public String getStyleName() {
        return styleName;
    }

    public Bitmap getStyleImage() {
        return styleImage;
    }
}
