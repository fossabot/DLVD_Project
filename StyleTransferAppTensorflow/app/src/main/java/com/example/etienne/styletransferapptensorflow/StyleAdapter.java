package com.example.etienne.styletransferapptensorflow;

import android.app.Activity;
import android.content.Context;
import android.content.DialogInterface;
import android.content.res.Resources;
import android.graphics.*;
import android.media.Image;
import android.support.v7.widget.RecyclerView;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.MotionEvent;
import android.view.View;
import android.view.ViewGroup;
import android.widget.AdapterView;
import android.widget.BaseAdapter;
import android.widget.ImageView;
import android.widget.TextView;
import org.w3c.dom.Text;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by etienne on 26.01.17.
 */
public class StyleAdapter extends RecyclerView.Adapter<StyleAdapter.MyViewHolder> {
    private List<ListItem> items;

    public StyleAdapter(List<ListItem> items){
        this.items = items;
    }

    @Override
    public MyViewHolder onCreateViewHolder(ViewGroup parent, int viewType) {
        View itemView = LayoutInflater.from(parent.getContext()).inflate(R.layout.style_entry,parent,false);
        return new MyViewHolder(itemView);
    }

    @Override
    public void onBindViewHolder(MyViewHolder holder, int position) {
        ListItem item = items.get(position);
        holder.textView.setText(item.getStyleName());
        Bitmap toDraw = item.getImage();
        Bitmap imageRounded = Bitmap.createBitmap(toDraw.getWidth(), toDraw.getHeight(), toDraw.getConfig());
        Canvas canvas = new Canvas(imageRounded);
        Paint mpaint = new Paint();
        mpaint.setAntiAlias(true);
        mpaint.setShader(new BitmapShader(toDraw, Shader.TileMode.CLAMP, Shader.TileMode.CLAMP));
        canvas.drawRoundRect((new RectF(0, 0, toDraw.getWidth(), toDraw.getHeight())), 50, 50, mpaint);
        holder.imageView.setImageBitmap(imageRounded);
    }

    @Override
    public int getItemCount() {
        return items.size();
    }


    public class MyViewHolder extends RecyclerView.ViewHolder{
        ImageView imageView;
        TextView textView;
        public MyViewHolder(View view){
            super(view);
            imageView = (ImageView) view.findViewById(R.id.styleImage);
            textView = (TextView) view.findViewById(R.id.styleName);
        }
    }

}
