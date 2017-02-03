package com.example.etienne.styletransferapptensorflow;

import android.Manifest;
import android.app.Activity;
import android.content.Context;

import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.*;
import android.graphics.drawable.ColorDrawable;
import android.net.Uri;
import android.os.*;
import android.provider.MediaStore;
import android.support.annotation.NonNull;
import android.support.v4.app.ActivityCompat;
import android.support.v7.app.ActionBar;
import android.support.v7.app.AppCompatActivity;
import android.support.v7.widget.DefaultItemAnimator;
import android.support.v7.widget.GridLayoutManager;
import android.support.v7.widget.RecyclerView;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.Toast;

import java.io.*;
import java.util.ArrayList;

public class MainActivity extends AppCompatActivity {
    private static final String TAG = "StyleTransferActivity";
    private static final int PICK_IMAGE_REQUEST = 2;
    private static final int REQUEST_TAKE_PHOTO = 222;
    private static final int REQUEST_CAMERA_PERMISSION = 100;

    private Uri imageUri;

    private Button takePhotoButton;
    private Button openGalleryButton;
    private RecyclerView recyclerView;

    private ArrayList<ModelListItem> items;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);


        if (ActivityCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED && ActivityCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(MainActivity.this, new String[]{Manifest.permission.CAMERA, Manifest.permission.WRITE_EXTERNAL_STORAGE}, REQUEST_CAMERA_PERMISSION);
            return;
        }


        getSupportActionBar().setDisplayOptions(ActionBar.DISPLAY_SHOW_CUSTOM);
        getSupportActionBar().setDisplayShowCustomEnabled(true);
        getSupportActionBar().setCustomView(R.layout.actionbar);
        getSupportActionBar().setBackgroundDrawable(new ColorDrawable(Color.rgb(0, 0, 0)));

        takePhotoButton = (Button) this.findViewById(R.id.takePictureButton);
        takePhotoButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                File photo = new File(Environment.getExternalStorageDirectory(),"pic.jpg");
                intent.putExtra(MediaStore.EXTRA_OUTPUT,
                        Uri.fromFile(photo));
                imageUri = Uri.fromFile(photo);
                startActivityForResult(intent, REQUEST_TAKE_PHOTO);
            }
        });

        openGalleryButton = (Button) this.findViewById(R.id.openGalleryButton);
        openGalleryButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent();
                intent.setType("image/*");
                intent.setAction(Intent.ACTION_GET_CONTENT);
                startActivityForResult(Intent.createChooser(intent, "Select Picture"), PICK_IMAGE_REQUEST);
            }
        });


        items = new ArrayList<>();
        fetchData();

        RecyclerView.LayoutManager layoutManager = new GridLayoutManager(getApplicationContext(),2);

        StylePreviewAdapter adapter = new StylePreviewAdapter(items);

        recyclerView = (RecyclerView) findViewById(R.id.showStylesRecyclerView);
        recyclerView.setLayoutManager(layoutManager);
        recyclerView.setItemAnimator(new DefaultItemAnimator());
        //recyclerView.addItemDecoration(new DividerItemDecoration(this, LinearLayoutManager.VERTICAL));
        recyclerView.setAdapter(adapter);
        recyclerView.addOnItemTouchListener(new StyleTouchListener(getApplicationContext(), new StyleTouchListener.OnItemClickListener() {
            @Override
            public void onItemClick(View view, int position) {
                Toast.makeText(MainActivity.this, "Something could happen here.", Toast.LENGTH_SHORT).show();
            }
        }));
    }

    @Override
    protected void onResume() {
        super.onResume();
        Log.e(TAG, "onResume");
    }

    @Override
    protected void onPause() {
        Log.e(TAG, "onPause");
        super.onPause();
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == PICK_IMAGE_REQUEST && resultCode == Activity.RESULT_OK && data != null && data.getData() != null) {
            Uri uri = data.getData();
            Bitmap pictureToSave = null;
            try {
                final Bitmap bitmap = MediaStore.Images.Media.getBitmap(getContentResolver(), uri);
                pictureToSave = bitmap;
            } catch (IOException e) {
                e.printStackTrace();
            }
            String imageName = saveImageInternal(pictureToSave);
            Intent showImage = new Intent(MainActivity.this, ShowActivity.class);
            showImage.putExtra("imageUri", imageName);
            startActivity(showImage);
        }

        if(requestCode == REQUEST_TAKE_PHOTO && resultCode == Activity.RESULT_OK){
            Bitmap pictureToSave = null;
            try {
                final Bitmap bitmap = MediaStore.Images.Media.getBitmap(getContentResolver(), imageUri);
                pictureToSave = bitmap;
            } catch (IOException e) {
                e.printStackTrace();
            }
            String imageName = saveImageInternal(pictureToSave);
            Intent showImage = new Intent(MainActivity.this, ShowActivity.class);
            showImage.putExtra("imageUri",imageName);
            startActivity(showImage);
        }
    }

    public String saveImageInternal(Bitmap bitmap) {
        String fileName = "myImage";
        try {
            ByteArrayOutputStream bytes = new ByteArrayOutputStream();
            bitmap.compress(Bitmap.CompressFormat.JPEG, 100, bytes);
            FileOutputStream fo = openFileOutput(fileName, Context.MODE_PRIVATE);
            fo.write(bytes.toByteArray());
            fo.close();
        } catch (Exception e) {
            e.printStackTrace();
            fileName = null;
        }
        return fileName;
    }



    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        if (requestCode == REQUEST_CAMERA_PERMISSION) {
            if (grantResults[0] == PackageManager.PERMISSION_DENIED) {
                // close the app
                Toast.makeText(MainActivity.this, "You denied the permissions needed to run this app", Toast.LENGTH_LONG).show();
                finish();
            }
        }
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
}
