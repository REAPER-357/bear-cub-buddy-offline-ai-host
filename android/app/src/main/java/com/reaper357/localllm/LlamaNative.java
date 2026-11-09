package com.reaper357.localllm;

/** Thin JNI wrapper around llama.cpp (see llama_jni.cpp). */
public class LlamaNative {
    static { System.loadLibrary("llamabridge"); }

    public interface TokenCallback { void onToken(byte[] utf8); }

    /** Returns a native handle, or 0 on failure. path may be /proc/self/fd/N. */
    public static native long loadModel(String path, int nCtx, int nThreads);

    /** Returns null on success, otherwise an error message. Blocks until finished or stopped. */
    public static native String generate(long handle, String[] roles, String[] contents,
                                         float temp, int topK, float topP, float minP,
                                         float repeat, int nPredict, int seed, TokenCallback cb);

    public static native void stop();
    public static native void unload(long handle);
}
