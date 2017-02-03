package com.example.etienne.styletransferapptensorflow;

import android.graphics.*;
import android.support.v7.widget.RecyclerView;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;

import java.util.List;

/**
 * Created by etienne on 03.02.17.
 */
public class StylePreviewAdapter extends RecyclerView.Adapter<StylePreviewAdapter.MyViewHolder> {
    private List<ModelListItem> items;

    public StylePreviewAdapter(List<ModelListItem> items){
        this.items = items;
    }

    @Override
    public StylePreviewAdapter.MyViewHolder onCreateViewHolder(ViewGroup parent, int viewType) {
        View itemView = LayoutInflater.from(parent.getContext()).inflate(R.layout.style_preview_entry,parent,false);
        return new StylePreviewAdapter.MyViewHolder(itemView);
    }

    @Override
    public void onBindViewHolder(MyViewHolder holder, int position) {
        ModelListItem item = items.get(position);
        Bitmap toDraw = item.getImage();
        holder.previewImage1.setImageBitmap(toDraw);
    }

    @Override
    public int getItemCount() {
        return items.size();
    }


    public class MyViewHolder extends RecyclerView.ViewHolder{
        ImageView previewImage1;

        public MyViewHolder(View view){
            super(view);
            previewImage1 = (ImageView) view.findViewById(R.id.previewImage1);
        }
    }
}
