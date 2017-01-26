package com.example.etienne.styletransferapptensorflow;

import android.graphics.Bitmap;

/**
 * Created by etienne on 26.01.17.
 */
public class ListItem {
    private String styleName;
    private Model model;
    private Bitmap image;


    public ListItem(String name, Model model, Bitmap image){
        this.styleName = name;
        this.model = model;
        this.image = image;

    }

    public String getStyleName() {
        return styleName;
    }

    public Model getModel() {
        return model;
    }

    public void activateStyle(){
        this.model.initializeStyle();
    }

    public void disableStyle(){
        this.model.closeStyle();
    }

    public Bitmap getImage() {
        return image;
    }
}
