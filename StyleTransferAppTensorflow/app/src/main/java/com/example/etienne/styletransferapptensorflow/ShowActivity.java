package com.example.etienne.styletransferapptensorflow;

import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import android.graphics.drawable.ColorDrawable;
import android.os.AsyncTask;
import android.support.v7.app.ActionBar;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.support.v7.widget.DefaultItemAnimator;
import android.support.v7.widget.LinearLayoutManager;
import android.support.v7.widget.RecyclerView;
import android.view.View;
import android.widget.*;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;

public class ShowActivity extends AppCompatActivity {
    private ImageView imageView;
    private Bitmap image;
    private RecyclerView recyclerView;
    private ProgressBar progressBar;


    private ArrayList<ModelListItem> items;
    private Model currentModel = null;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_show);

        progressBar = (ProgressBar) this.findViewById(R.id.showProgressBar);
        progressBar.setVisibility(View.GONE);

        getSupportActionBar().setDisplayOptions(ActionBar.DISPLAY_SHOW_CUSTOM);
        getSupportActionBar().setDisplayShowCustomEnabled(true);
        getSupportActionBar().setCustomView(R.layout.actionbar);
        getSupportActionBar().setBackgroundDrawable(new ColorDrawable(Color.rgb(0, 0, 0)));

        imageView = (ImageView) this.findViewById(R.id.showImageView);
        try {
            image = BitmapFactory.decodeStream(openFileInput(getIntent().getStringExtra("imageUri")));
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }

        imageView.setImageBitmap(image);

        items = new ArrayList<>();
        fetchData();

        StyleAdapter adapter = new StyleAdapter(items);

        LinearLayoutManager layoutManager= new LinearLayoutManager(this,LinearLayoutManager.HORIZONTAL, false);

        recyclerView = (RecyclerView) findViewById(R.id.listView);
        recyclerView.setLayoutManager(layoutManager);
        recyclerView.setItemAnimator(new DefaultItemAnimator());
        recyclerView.addItemDecoration(new DividerItemDecoration(this, LinearLayoutManager.HORIZONTAL));
        recyclerView.setAdapter(adapter);
        recyclerView.addOnItemTouchListener(new StyleTouchListener(getApplicationContext(), new StyleTouchListener.OnItemClickListener() {
            @Override
            public void onItemClick(View view, int position) {
                items.get(position).activateStyle();
                currentModel = items.get(position).getModel();
                new ApplyModelTask().execute(image);
            }
        }));

        imageView.setOnTouchListener(new OnSwipeTouchListener(ShowActivity.this){
            public void onSwipeTop() {
            }
            public void onSwipeRight() {
                Intent i = new Intent(ShowActivity.this,MainActivity.class);
                startActivity(i);
            }
            public void onSwipeLeft() {
            }
            public void onSwipeBottom() {

            }
        });
    }

    private void fetchData() {
        try {
            String[] names = getAssets().list("");
            for (String name : names) {
                if (name.contains(".pb")) {
                    Model model = new Model(name, getAssets());
                    String pureName = name.substring(0, name.indexOf("."));
                    int id = getResources().getIdentifier(pureName, "drawable", getPackageName());
                    Bitmap bmp = BitmapFactory.decodeResource(getResources(), id);
                    pureName = Character.toUpperCase(pureName.charAt(0))+ pureName.substring(1,pureName.length());
                    ModelListItem item = new ModelListItem(pureName, model, bmp);
                    items.add(item);
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private class ApplyModelTask extends AsyncTask<Bitmap,String,Bitmap> {

        @Override
        protected void onPreExecute() {
            super.onPreExecute();
            progressBar.setVisibility(View.VISIBLE);
        }

        @Override
        protected Bitmap doInBackground(Bitmap... params) {
            Bitmap bm = params[0];
            return currentModel.applyModel(bm);
        }

        @Override
        protected void onProgressUpdate(String... values) {
            super.onProgressUpdate(values);
        }

        @Override
        protected void onPostExecute(Bitmap bitmap) {
            super.onPostExecute(bitmap);
            progressBar.setVisibility(View.GONE);
            imageView.setImageBitmap(bitmap);
        }
    }
}
