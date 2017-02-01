package com.example.etienne.styletransferapptensorflow;

import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.AsyncTask;
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
    private ImageButton backButton;
    private ProgressBar progressBar;


    private ArrayList<ListItem> items;
    private int lastSelection = -1;
    private Model currentModel = null;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_show);

        progressBar = (ProgressBar) this.findViewById(R.id.showProgressBar);
        progressBar.setVisibility(View.GONE);

        imageView = (ImageView) this.findViewById(R.id.showImageView);
        try {
            image = BitmapFactory.decodeStream(getApplicationContext().openFileInput(getIntent().getStringExtra("imageUri")));
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }

        image = Bitmap.createScaledBitmap(image,882,882,false);
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
                lastSelection = position;
                items.get(position).activateStyle();
                currentModel = items.get(position).getModel();
                new ApplyModelTask().execute(image);
            }
        }));

        backButton = (ImageButton) this.findViewById(R.id.backButton);
        backButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent i = new Intent(ShowActivity.this,MainActivity.class);
                startActivity(i);
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
                    ListItem item = new ListItem(pureName, model, bmp);
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
            bm = Bitmap.createScaledBitmap(bm,304,304,false);
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
