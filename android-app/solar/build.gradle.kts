plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

// Solar panel cell tracker: a second app next to the hand demo. It reuses the hand app's thermal USB stream,
// calibration model and geometry as sources (../app), so both apps stay on one copy of that code.
android {
    namespace = "com.euhack.solar"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.euhack.solar"
        minSdk = 23
        targetSdk = 35
        versionCode = 1
        versionName = "1.0"
        ndk { abiFilters += "arm64-v8a" }
    }

    sourceSets["main"].java.srcDirs("src/main/java", "../app/src/main/java")
    sourceSets["main"].res.srcDirs("../app/src/main/res")

    androidResources {
        noCompress += "pte"
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
    testImplementation("junit:junit:4.13.2")
}
