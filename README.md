# Local LLM - build notes

The app is a host. Models are NOT in the APK. You connect a folder once and the app reads
the .gguf files there in place (no copying). Swap, add or delete files with any file manager,
or use Models > Download.

## Files in this package
    index.html                                   -> goes in www/ (with assets/icons/)
    android/app/src/main/java/com/reaper357/localllm/
        MainActivity.java                        -> REPLACES the generated one
        LlamaBridgePlugin.java, LlamaNative.java -> new
    android/app/src/main/cpp/
        CMakeLists.txt, llama_jni.cpp            -> new (llama.cpp goes next to them, step 3)

## Icons (copy from your lucide-main pack into www/assets/icons/, same names, .svg)
message-square, box, settings, send, square, plus, trash-2, folder-open,
refresh-cw, download, play, power, x

## Build steps
1. Project + Capacitor
       npm init -y
       npm i @capacitor/core @capacitor/cli @capacitor/android
       npx cap init "Local LLM" com.reaper357.localllm --web-dir www
       mkdir -p www/assets/icons        # then copy index.html and the 13 icons in
       npx cap add android
2. Copy the android/ folder from this package over your android/ folder (replaces MainActivity.java).
3. Get llama.cpp next to the CMake file:
       cd android/app/src/main/cpp
       git clone --depth 1 https://github.com/ggml-org/llama.cpp
4. Edit android/app/build.gradle:
   - inside `android { defaultConfig { ... } }` add:
         ndk { abiFilters "arm64-v8a" }
         externalNativeBuild { cmake { cppFlags "-std=c++17" } }
   - inside `android { ... }` add:
         externalNativeBuild { cmake { path "src/main/cpp/CMakeLists.txt" } }
   - inside `dependencies { ... }` add:
         implementation "androidx.documentfile:documentfile:1.0.1"
   - Install "NDK (Side by side)" and "CMake" from Android Studio > SDK Manager > SDK Tools.
5. npx cap sync, then open android/ in Android Studio and Build > Build APK.
   The first build compiles llama.cpp and takes several minutes. Use a Release build for the
   final APK (Debug is much slower at inference).

## Notes
- No internet permission is needed except for Download. Add this to AndroidManifest.xml
  if you want Download to work:  <uses-permission android:name="android.permission.INTERNET" />
- Folder access uses Android's folder picker (SAF). The app keeps the permission after restarts.
- Pick a folder that is NOT the root of storage or Downloads (Android refuses those). A folder
  you create, e.g. Documents/Models, works.
- A 3B Q4_K_M model (about 2 GB) fits easily in 11 GB RAM. Keep context at 4096 or lower.
- Every turn re-reads the whole chat (simple and consistent). Old turns are dropped
  automatically when the chat nears the context size.
- If the compile reports a renamed llama.cpp function, it is a one-line rename in llama_jni.cpp.
- Not tested on a device yet. Expect to fix small compile or API differences on the first build.
