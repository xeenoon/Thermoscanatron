plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.euhack.hello"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.euhack.hello"
        minSdk = 23
        targetSdk = 35
        versionCode = 1
        versionName = "1.0"
        ndk { abiFilters += "arm64-v8a" }  // phones only; drops the x86_64 ExecuTorch libs
    }

    androidResources {
        noCompress += "pte"  // the model is copied out of the APK as-is
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
}

kotlin {
    jvmToolchain(17)
}

dependencies {
    val camerax = "1.4.1"
    implementation("androidx.camera:camera-camera2:$camerax")
    implementation("androidx.camera:camera-lifecycle:$camerax")
    implementation("androidx.camera:camera-view:$camerax")
    implementation("androidx.camera:camera-video:$camerax")
    implementation("androidx.activity:activity-ktx:1.9.3")
    // Must match the executorch version that exported the .pte (ML/uv.lock: 1.5.1).
    implementation("org.pytorch:executorch-android:1.5.1")
}
