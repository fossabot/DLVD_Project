package com.example.etienne.styletransferapptensorflow;

import android.Manifest;
import android.app.Activity;
import android.content.Intent;

import android.content.pm.PackageManager;
import android.graphics.*;
import android.net.Uri;
import android.os.*;
import android.provider.MediaStore;


import android.support.v4.app.ActivityCompat;
import android.support.v7.app.AppCompatActivity;
import android.support.v7.widget.DefaultItemAnimator;
import android.support.v7.widget.LinearLayoutManager;
import android.support.v7.widget.RecyclerView;
import android.util.Log;
import android.view.MotionEvent;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.Toast;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;

public class MainActivity extends AppCompatActivity {
    private Button button;
    private Bitmap result;
    private ImageView imageView;
    private RecyclerView recyclerView;
    private ArrayList<ListItem> items;
    private int lastSelection = -1;
    private Model currentModel = null;

    File myFilesDir;

    private static final int CAMERA_REQUEST = 1888;

    private static final int REQUEST_CAMERA_PERMISSION = 200;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        LinearLayoutManager layoutManager= new LinearLayoutManager(this,LinearLayoutManager.HORIZONTAL, false);
        recyclerView = (RecyclerView) findViewById(R.id.listView);
        recyclerView.setLayoutManager(layoutManager);
        imageView = (ImageView) this.findViewById(R.id.imageView);

        myFilesDir = new File(Environment.getExternalStorageDirectory().getAbsolutePath() + "/Fotos");
        myFilesDir.mkdirs();


        items = new ArrayList<>();
        fetchData();

        StyleAdapter adapter = new StyleAdapter(items);
        recyclerView.setItemAnimator(new DefaultItemAnimator());
        recyclerView.addItemDecoration(new DividerItemDecoration(this,LinearLayoutManager.HORIZONTAL));
        recyclerView.setAdapter(adapter);
        recyclerView.addOnItemTouchListener(new StyleTouchListener(getApplicationContext(), new StyleTouchListener.OnItemClickListener() {
            @Override
            public void onItemClick(View view, int position) {
                if(lastSelection == -1){
                    lastSelection = position;
                    items.get(position).activateStyle();
                    currentModel = items.get(position).getModel();
                }else{
                    items.get(lastSelection).disableStyle();
                    lastSelection = position;
                    items.get(position).activateStyle();
                    currentModel = items.get(position).getModel();
                }

                Toast.makeText(MainActivity.this, "Currently selected style "+ items.get(position).getStyleName() , Toast.LENGTH_SHORT).show();
            }
        }));

        button = (Button) this.findViewById(R.id.photoButton);
        button.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                    Intent cameraIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                    cameraIntent.putExtra(MediaStore.EXTRA_OUTPUT, Uri.fromFile(new File(myFilesDir.toString() + "/temp.jpg")));
                    startActivityForResult(cameraIntent, CAMERA_REQUEST);

            }
        });

        if (ActivityCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED && ActivityCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(MainActivity.this, new String[]{Manifest.permission.CAMERA, Manifest.permission.WRITE_EXTERNAL_STORAGE}, REQUEST_CAMERA_PERMISSION);
            return;
        }
    }

    private void fetchData() {
        try {
            String[] names = getAssets().list("");
            for(String name : names){
                if(name.contains(".pb")){
                    Model model = new Model(name,getAssets());
                    String pureName = name.substring(0,name.indexOf("."));
                    Log.d("purename",pureName);
                    int id = getResources().getIdentifier(pureName,"drawable",getPackageName());
                    Bitmap bmp = BitmapFactory.decodeResource(getResources(),id);
                    ListItem item = new ListItem(pureName,model,bmp);
                    items.add(item);
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    @Override
    protected void onResume() {
        super.onResume();
    }

    @Override
    protected void onPause() {
        super.onPause();
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if(requestCode == CAMERA_REQUEST && resultCode == Activity.RESULT_OK){
            result = BitmapFactory.decodeFile(myFilesDir.toString() + "/temp.jpg");
            result = Utils.cropBitmapSquare(result);
            Log.d("size",String.valueOf(result.getWidth() + "   " + result.getHeight()));
            //result = BitmapFactory.decodeResource(getResources(),R.drawable.gen_export);
            result = Bitmap.createScaledBitmap(result,304,304,false);
            result = currentModel.applyModel(result);
            result = Bitmap.createScaledBitmap(result,400,400,false);
            imageView.setImageBitmap(result);
            imageView.setVisibility(View.VISIBLE);
        }
    }
}
