package com.reaper357.localllm;

import android.app.Activity;
import android.content.Context;
import android.content.Intent;
import android.content.SharedPreferences;
import android.net.Uri;
import android.os.ParcelFileDescriptor;

import androidx.activity.result.ActivityResult;
import androidx.documentfile.provider.DocumentFile;

import com.getcapacitor.JSArray;
import com.getcapacitor.JSObject;
import com.getcapacitor.Plugin;
import com.getcapacitor.PluginCall;
import com.getcapacitor.PluginMethod;
import com.getcapacitor.annotation.ActivityCallback;
import com.getcapacitor.annotation.CapacitorPlugin;

import org.json.JSONObject;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * LlamaBridge: user-connected models folder (Storage Access Framework) + llama.cpp inference.
 * Models are read IN PLACE through a file descriptor, never copied.
 */
@CapacitorPlugin(name = "LlamaBridge")
public class LlamaBridgePlugin extends Plugin {

    private static final String PREFS = "llamabridge";
    private static final String KEY_TREE = "tree";

    private final ExecutorService io = Executors.newCachedThreadPool();
    private final ExecutorService gen = Executors.newSingleThreadExecutor();
    private final Object lock = new Object();

    private long handle = 0;
    private ParcelFileDescriptor pfd = null;
    private volatile boolean dlCancel = false;

    // ---------------------------------------------------------------- folder

    private SharedPreferences prefs() {
        return getContext().getSharedPreferences(PREFS, Context.MODE_PRIVATE);
    }

    private DocumentFile tree() {
        String s = prefs().getString(KEY_TREE, null);
        if (s == null) return null;
        try {
            DocumentFile d = DocumentFile.fromTreeUri(getContext(), Uri.parse(s));
            return (d != null && d.exists() && d.canRead()) ? d : null;
        } catch (Exception e) {
            return null;
        }
    }

    @PluginMethod
    public void pickFolder(PluginCall call) {
        Intent i = new Intent(Intent.ACTION_OPEN_DOCUMENT_TREE);
        i.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION
                | Intent.FLAG_GRANT_WRITE_URI_PERMISSION
                | Intent.FLAG_GRANT_PERSISTABLE_URI_PERMISSION
                | Intent.FLAG_GRANT_PREFIX_URI_PERMISSION);
        startActivityForResult(call, i, "pickFolderResult");
    }

    @ActivityCallback
    private void pickFolderResult(PluginCall call, ActivityResult result) {
        if (call == null) return;
        Intent data = result.getData();
        if (result.getResultCode() != Activity.RESULT_OK || data == null || data.getData() == null) {
            call.reject("cancelled");
            return;
        }
        Uri uri = data.getData();
        try {
            getContext().getContentResolver().takePersistableUriPermission(uri,
                    Intent.FLAG_GRANT_READ_URI_PERMISSION | Intent.FLAG_GRANT_WRITE_URI_PERMISSION);
        } catch (Exception e) {
            call.reject("Could not keep access to that folder: " + e.getMessage());
            return;
        }
        prefs().edit().putString(KEY_TREE, uri.toString()).apply();
        call.resolve();
    }

    @PluginMethod
    public void getFolder(PluginCall call) {
        DocumentFile d = tree();
        JSObject o = new JSObject();
        o.put("connected", d != null);
        o.put("name", d != null ? d.getName() : "");
        call.resolve(o);
    }

    @PluginMethod
    public void listModels(PluginCall call) {
        io.execute(() -> {
            DocumentFile d = tree();
            if (d == null) { call.reject("No folder connected"); return; }
            try {
                JSArray arr = new JSArray();
                for (DocumentFile f : d.listFiles()) {
                    String n = f.getName();
                    if (f.isFile() && n != null && n.toLowerCase().endsWith(".gguf")) {
                        JSObject o = new JSObject();
                        o.put("name", n);
                        o.put("size", f.length());
                        o.put("uri", f.getUri().toString());
                        arr.put(o);
                    }
                }
                JSObject res = new JSObject();
                res.put("models", arr);
                call.resolve(res);
            } catch (Exception e) {
                call.reject("List failed: " + e.getMessage());
            }
        });
    }

    @PluginMethod
    public void deleteModel(PluginCall call) {
        String uri = call.getString("uri");
        io.execute(() -> {
            try {
                DocumentFile f = DocumentFile.fromSingleUri(getContext(), Uri.parse(uri));
                if (f != null && f.delete()) call.resolve();
                else call.reject("Could not delete the file");
            } catch (Exception e) {
                call.reject("Delete failed: " + e.getMessage());
            }
        });
    }

    // ---------------------------------------------------------------- model

    private void closePfd() {
        if (pfd != null) {
            try { pfd.close(); } catch (IOException ignored) { }
            pfd = null;
        }
    }

    private void unloadInternal() {
        LlamaNative.stop();                 // let a running generate() exit
        synchronized (lock) {
            if (handle != 0) { LlamaNative.unload(handle); handle = 0; }
            closePfd();
        }
    }

    @PluginMethod
    public void load(PluginCall call) {
        final String uriStr = call.getString("uri");
        final int nCtx = call.getInt("n_ctx", 4096);
        final int nThreads = call.getInt("n_threads", 4);
        io.execute(() -> {
            try {
                unloadInternal();
                synchronized (lock) {
                    pfd = getContext().getContentResolver().openFileDescriptor(Uri.parse(uriStr), "r");
                    if (pfd == null) { call.reject("Cannot open the model file"); return; }
                    String path = "/proc/self/fd/" + pfd.getFd();
                    long h = LlamaNative.loadModel(path, nCtx, nThreads);
                    if (h == 0) {
                        closePfd();
                        call.reject("Model failed to load (not enough free RAM, unsupported architecture, or unreadable file)");
                        return;
                    }
                    handle = h;
                }
                call.resolve();
            } catch (Throwable e) {
                synchronized (lock) { closePfd(); }
                call.reject("Load failed: " + e.getMessage());
            }
        });
    }

    @PluginMethod
    public void unload(PluginCall call) {
        io.execute(() -> { unloadInternal(); call.resolve(); });
    }

    @PluginMethod
    public void stop(PluginCall call) {
        LlamaNative.stop();
        call.resolve();
    }

    @PluginMethod
    public void generate(PluginCall call) {
        final JSArray msgs = call.getArray("messages");
        final float temp = call.getFloat("temperature", 0.7f);
        final int topK = call.getInt("top_k", 40);
        final float topP = call.getFloat("top_p", 0.95f);
        final float minP = call.getFloat("min_p", 0.05f);
        final float rep = call.getFloat("repeat_penalty", 1.1f);
        final int nPredict = call.getInt("n_predict", 1024);
        final int seed = call.getInt("seed", -1);
        gen.execute(() -> {
            try {
                int n = msgs.length();
                String[] roles = new String[n];
                String[] contents = new String[n];
                for (int i = 0; i < n; i++) {
                    JSONObject o = msgs.getJSONObject(i);
                    roles[i] = o.getString("role");
                    contents[i] = o.getString("content");
                }
                String err;
                synchronized (lock) {
                    if (handle == 0) { call.reject("No model loaded"); return; }
                    err = LlamaNative.generate(handle, roles, contents, temp, topK, topP, minP, rep,
                            nPredict, seed, bytes -> {
                                JSObject ev = new JSObject();
                                ev.put("token", new String(bytes, StandardCharsets.UTF_8));
                                notifyListeners("token", ev);
                            });
                }
                if (err != null) call.reject(err); else call.resolve();
            } catch (Throwable e) {
                call.reject("Generate failed: " + e.getMessage());
            }
        });
    }

    // ------------------------------------------------------------- download

    @PluginMethod
    public void cancelDownload(PluginCall call) {
        dlCancel = true;
        call.resolve();
    }

    @PluginMethod
    public void downloadModel(PluginCall call) {
        final String url = call.getString("url");
        final String name = call.getString("name");
        final String token = call.getString("token", "");
        dlCancel = false;
        io.execute(() -> {
            DocumentFile dir = tree();
            if (dir == null) { call.reject("No folder connected"); return; }
            if (dir.findFile(name) != null) { call.reject("A file with this name already exists in the folder"); return; }
            DocumentFile out = null;
            HttpURLConnection c = null;
            try {
                c = open(url, token);
                long total = c.getContentLengthLong();
                out = dir.createFile("application/octet-stream", name);
                if (out == null) throw new IOException("Could not create the file in the folder");
                if (!name.equals(out.getName())) out.renameTo(name);   // some providers add an extension
                long done = 0, last = 0;
                try (InputStream in = c.getInputStream();
                     OutputStream os = getContext().getContentResolver().openOutputStream(out.getUri(), "w")) {
                    if (os == null) throw new IOException("Cannot write to the folder");
                    byte[] buf = new byte[1 << 16];
                    int n;
                    while ((n = in.read(buf)) > 0) {
                        if (dlCancel) throw new IOException("Cancelled");
                        os.write(buf, 0, n);
                        done += n;
                        long now = System.currentTimeMillis();
                        if (now - last > 300) { last = now; emitProgress(done, total); }
                    }
                    os.flush();
                }
                emitProgress(done, total);
                call.resolve();
            } catch (Exception e) {
                if (out != null) out.delete();
                call.reject(e.getMessage() != null ? e.getMessage() : "Download failed");
            } finally {
                if (c != null) c.disconnect();
            }
        });
    }

    private void emitProgress(long bytes, long total) {
        JSObject ev = new JSObject();
        ev.put("bytes", bytes);
        ev.put("total", total);
        notifyListeners("download", ev);
    }

    /** Follows redirects by hand so the HF token is only ever sent to huggingface.co. */
    private HttpURLConnection open(String url, String token) throws IOException {
        String cur = url;
        for (int i = 0; i < 6; i++) {
            URL u = new URL(cur);
            if (!"https".equals(u.getProtocol())) throw new IOException("Only https links are allowed");
            HttpURLConnection c = (HttpURLConnection) u.openConnection();
            c.setInstanceFollowRedirects(false);
            c.setConnectTimeout(20000);
            c.setReadTimeout(30000);
            c.setRequestProperty("User-Agent", "LocalLLM/1.0");
            if (token != null && !token.isEmpty() && u.getHost().endsWith("huggingface.co")) {
                c.setRequestProperty("Authorization", "Bearer " + token);
            }
            int code = c.getResponseCode();
            if (code >= 300 && code < 400) {
                String loc = c.getHeaderField("Location");
                c.disconnect();
                if (loc == null) throw new IOException("Bad redirect");
                cur = new URL(u, loc).toString();
                continue;
            }
            if (code == 401 || code == 403) { c.disconnect(); throw new IOException("Access denied (gated model? add an access token)"); }
            if (code != 200) { c.disconnect(); throw new IOException("HTTP " + code); }
            return c;
        }
        throw new IOException("Too many redirects");
    }

    @Override
    protected void handleOnDestroy() {
        unloadInternal();
        super.handleOnDestroy();
    }
}
