package com.zanin.olhovivo

import android.content.pm.PackageManager
import org.opencv.core.MatOfRect
import org.opencv.core.Scalar
import org.opencv.core.Size
import android.graphics.Bitmap
import java.io.File
import java.io.FileOutputStream
import java.io.InputStream
import android.Manifest
import android.os.Bundle
import android.widget.Button
import android.widget.CheckBox
import android.widget.ImageView
import android.widget.TextView
import androidx.activity.enableEdgeToEdge
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import org.opencv.android.CameraBridgeViewBase
import org.opencv.android.OpenCVLoader
import org.opencv.core.Mat
import org.opencv.imgproc.Imgproc
import org.opencv.objdetect.CascadeClassifier
import org.opencv.android.Utils
import org.opencv.core.CvType.CV_8UC4
import android.media.AudioManager
import android.media.ToneGenerator
import android.os.Handler
import android.os.Looper

class MainActivity : AppCompatActivity(), CameraBridgeViewBase.CvCameraViewListener2 {

    private lateinit var buttonStartPreview: Button
    private lateinit var buttonStopPreview: Button
    private lateinit var checkboxGlassesMode: CheckBox
    private lateinit var imageView: ImageView
    private lateinit var openCvCameraView: CameraBridgeViewBase
    private lateinit var textViewStatus: TextView

    private var isPreviewActive = false
    private lateinit var inputMat: Mat
    private lateinit var eyeClassifier: CascadeClassifier
    private var isOpenCvInitialized = false
    private var lastEyeDetectionTime = System.currentTimeMillis()
    private val detectionTimeout = 3000

    private lateinit var toneGenerator: ToneGenerator
    private val handler = Handler(Looper.getMainLooper())
    private var isAlarmPlaying = false

    private val cameraPermissionRequestCode = 100

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContentView(R.layout.activity_main)

        textViewStatus = findViewById(R.id.textViewStatus)
        buttonStartPreview = findViewById(R.id.buttonStartPreview)
        buttonStopPreview = findViewById(R.id.buttonStopPreview)
        checkboxGlassesMode = findViewById(R.id.checkboxGlassesMode)
        imageView = findViewById(R.id.imageView)
        openCvCameraView = findViewById(R.id.cameraView)
        isOpenCvInitialized = OpenCVLoader.initLocal()
        eyeClassifier = loadCascadeClassifier(R.raw.haarcascade_eye_tree_eyeglasses)
        toneGenerator = ToneGenerator(AudioManager.STREAM_ALARM, 100)

        checkboxGlassesMode.setOnCheckedChangeListener { _, isChecked ->
            eyeClassifier = if (isChecked) {
                loadCascadeClassifier(R.raw.haarcascade_eye_tree_eyeglasses)
            } else {
                loadCascadeClassifier(R.raw.haarcascade_eye)
            }
        }

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.CAMERA), cameraPermissionRequestCode)
        }

        openCvCameraView.setCameraIndex(1)
        openCvCameraView.setCvCameraViewListener(this)

        buttonStartPreview.setOnClickListener {
            openCvCameraView.setCameraPermissionGranted()
            openCvCameraView.enableView()

            lastEyeDetectionTime = System.currentTimeMillis()
            isAlarmPlaying = false
            updateControls()
        }

        buttonStopPreview.setOnClickListener {
            openCvCameraView.disableView()
            isPreviewActive = false

            handler.removeCallbacks(alarmRunnable)
            toneGenerator.stopTone()
            isAlarmPlaying = false

            imageView.setImageBitmap(null)

            updateControls()
        }

        updateControls()
    }

    private fun loadCascadeClassifier(resourceId: Int): CascadeClassifier {
        val inputStream: InputStream = resources.openRawResource(resourceId)
        val tempFile = File(cacheDir, "haarcascade_eye.xml")
        FileOutputStream(tempFile).use { outputStream ->
            inputStream.copyTo(outputStream)
        }
        val classifier = CascadeClassifier(tempFile.absolutePath)
        if (classifier.empty()) {
            throw RuntimeException("Failed to load cascade classifier from " + tempFile.absolutePath)
        }
        return classifier
    }

    private fun updateControls() {
        if (!isOpenCvInitialized) {
            textViewStatus.text = "OpenCV initialization error"
            buttonStartPreview.isEnabled = false
            buttonStopPreview.isEnabled = false
        } else {
            textViewStatus.text = "OpenCV initialized"
            buttonStartPreview.isEnabled = !isPreviewActive
            buttonStopPreview.isEnabled = isPreviewActive
        }
    }

    override fun onCameraViewStarted(width: Int, height: Int) {
        isPreviewActive = true
        inputMat = Mat(height, width, CV_8UC4)
        updateControls()
    }

    override fun onCameraViewStopped() {
        isPreviewActive = false
        inputMat.release()
        updateControls()
    }

    override fun onCameraFrame(inputFrame: CameraBridgeViewBase.CvCameraViewFrame?): Mat {
        if (!isPreviewActive) {
            return inputFrame?.rgba() ?: Mat()
        }

        inputFrame!!.rgba().copyTo(inputMat)
        val width = inputMat.cols() / 4
        val height = inputMat.rows() / 4
        val resizedMat = Mat()
        Imgproc.resize(inputMat, resizedMat, Size(width.toDouble(), height.toDouble()))

        val grayMat = Mat()
        Imgproc.cvtColor(resizedMat, grayMat, Imgproc.COLOR_RGBA2GRAY)

        val eyes = MatOfRect()
        eyeClassifier.detectMultiScale(grayMat, eyes, 1.1, 3)

        if (eyes.toArray().isNotEmpty()) {
            lastEyeDetectionTime = System.currentTimeMillis()
            isAlarmPlaying = false
            handler.removeCallbacks(alarmRunnable)
            for (rect in eyes.toArray()) {
                Imgproc.rectangle(resizedMat, rect.tl(), rect.br(), Scalar(255.0, 0.0, 0.0), 2)
            }
        } else if (System.currentTimeMillis() - lastEyeDetectionTime > detectionTimeout && !isAlarmPlaying) {
            isAlarmPlaying = true
            handler.post(alarmRunnable)
        }

        Imgproc.resize(resizedMat, inputMat, Size(inputMat.cols().toDouble(), inputMat.rows().toDouble()))

        val bitmapToDisplay = Bitmap.createBitmap(inputMat.cols(), inputMat.rows(), Bitmap.Config.ARGB_8888)
        Utils.matToBitmap(inputMat, bitmapToDisplay)

        runOnUiThread {
            if (isPreviewActive) {
                imageView.setImageBitmap(bitmapToDisplay)
            }
        }

        return inputMat
    }

    private val alarmRunnable = object : Runnable {
        override fun run() {
            toneGenerator.startTone(ToneGenerator.TONE_CDMA_ALERT_CALL_GUARD, 500)
            handler.postDelayed(this, 500)
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        handler.removeCallbacks(alarmRunnable)
        toneGenerator.release()
    }
}