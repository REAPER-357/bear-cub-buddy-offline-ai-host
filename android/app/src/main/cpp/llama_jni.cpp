// JNI glue between LlamaNative.java and llama.cpp.
// Written against the current llama.cpp C API (llama_model_load_from_file, llama_init_from_model,
// llama_memory_clear, llama_sampler_*). If your llama.cpp checkout renamed something, the compiler
// error will name it - it is a one-line fix.
#include <jni.h>
#include <android/log.h>
#include <atomic>
#include <algorithm>
#include <climits>
#include <string>
#include <vector>
#include "llama.h"

#define TAG "LlamaBridge"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)

struct Engine {
    llama_model*       model;
    llama_context*     ctx;
    const llama_vocab* vocab;
    int                n_ctx;
};

static std::atomic<bool> g_stop{false};
static bool g_backend = false;

// Length of the longest prefix of s that does not end in a half-finished UTF-8 character.
static size_t utf8_complete_len(const std::string& s) {
    size_t n = s.size();
    if (n == 0) return 0;
    size_t i = n;
    int back = 0;
    while (i > 0 && back < 4) {
        i--; back++;
        unsigned char ch = (unsigned char) s[i];
        if ((ch & 0xC0) == 0x80) continue;                 // continuation byte, keep looking back
        int need = ch >= 0xF0 ? 4 : ch >= 0xE0 ? 3 : ch >= 0xC0 ? 2 : 1;
        return back >= need ? n : i;                       // incomplete -> cut before the lead byte
    }
    return n;
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_reaper357_localllm_LlamaNative_loadModel(JNIEnv* env, jclass, jstring jpath, jint nCtx, jint nThreads) {
    if (!g_backend) { llama_backend_init(); g_backend = true; }

    const char* path = env->GetStringUTFChars(jpath, nullptr);
    llama_model_params mp = llama_model_default_params();
    mp.use_mmap = true;
    llama_model* model = llama_model_load_from_file(path, mp);
    env->ReleaseStringUTFChars(jpath, path);
    if (!model) { LOGI("model load failed"); return 0; }

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx           = (uint32_t) nCtx;
    cp.n_batch         = 512;
    cp.n_ubatch        = 512;
    cp.n_threads       = nThreads;
    cp.n_threads_batch = nThreads;
    llama_context* ctx = llama_init_from_model(model, cp);
    if (!ctx) { llama_model_free(model); LOGI("context init failed"); return 0; }

    Engine* e = new Engine{model, ctx, llama_model_get_vocab(model), (int) llama_n_ctx(ctx)};
    LOGI("model ready, n_ctx=%d", e->n_ctx);
    return (jlong) e;
}

extern "C" JNIEXPORT void JNICALL
Java_com_reaper357_localllm_LlamaNative_stop(JNIEnv*, jclass) { g_stop = true; }

extern "C" JNIEXPORT void JNICALL
Java_com_reaper357_localllm_LlamaNative_unload(JNIEnv*, jclass, jlong h) {
    Engine* e = (Engine*) h;
    if (!e) return;
    llama_free(e->ctx);
    llama_model_free(e->model);
    delete e;
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_reaper357_localllm_LlamaNative_generate(JNIEnv* env, jclass, jlong h,
        jobjectArray jroles, jobjectArray jcontents,
        jfloat temp, jint topK, jfloat topP, jfloat minP, jfloat repeat,
        jint nPredict, jint seed, jobject cb) {
    Engine* e = (Engine*) h;
    if (!e) return env->NewStringUTF("No model loaded");
    g_stop = false;

    // ---- build the chat prompt with the model's own template
    jsize n = env->GetArrayLength(jroles);
    std::vector<std::string> roles(n), contents(n);
    for (jsize i = 0; i < n; i++) {
        jstring r = (jstring) env->GetObjectArrayElement(jroles, i);
        jstring c = (jstring) env->GetObjectArrayElement(jcontents, i);
        const char* rs = env->GetStringUTFChars(r, nullptr);
        const char* cs = env->GetStringUTFChars(c, nullptr);
        roles[i] = rs; contents[i] = cs;
        env->ReleaseStringUTFChars(r, rs); env->ReleaseStringUTFChars(c, cs);
        env->DeleteLocalRef(r); env->DeleteLocalRef(c);
    }
    std::vector<llama_chat_message> msgs(n);
    for (jsize i = 0; i < n; i++) msgs[i] = { roles[i].c_str(), contents[i].c_str() };

    const char* tmpl = llama_model_chat_template(e->model, nullptr);
    std::vector<char> buf(16384);
    int len = llama_chat_apply_template(tmpl ? tmpl : "chatml", msgs.data(), msgs.size(), true, buf.data(), (int) buf.size());
    if (len < 0) len = llama_chat_apply_template("chatml", msgs.data(), msgs.size(), true, buf.data(), (int) buf.size());
    if (len > (int) buf.size()) {
        buf.resize(len);
        len = llama_chat_apply_template(tmpl ? tmpl : "chatml", msgs.data(), msgs.size(), true, buf.data(), (int) buf.size());
    }
    if (len < 0) return env->NewStringUTF("Chat template not supported for this model");
    std::string prompt(buf.data(), len);

    // ---- tokenize
    int nTok = -llama_tokenize(e->vocab, prompt.c_str(), (int) prompt.size(), nullptr, 0, true, true);
    if (nTok <= 0) return env->NewStringUTF("Tokenization failed");
    std::vector<llama_token> toks(nTok);
    if (llama_tokenize(e->vocab, prompt.c_str(), (int) prompt.size(), toks.data(), nTok, true, true) < 0)
        return env->NewStringUTF("Tokenization failed");
    if (nTok >= e->n_ctx - 8) return env->NewStringUTF("Conversation is longer than the context size. Start a new chat or raise Context size.");

    // fresh start every turn (history is re-sent by the UI): simple and always consistent
    llama_memory_clear(llama_get_memory(e->ctx), true);

    // ---- sampler chain
    llama_sampler* smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(smpl, llama_sampler_init_penalties(64, repeat, 0.0f, 0.0f));
    if (temp <= 0.0f) {
        llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
    } else {
        if (topK > 0)    llama_sampler_chain_add(smpl, llama_sampler_init_top_k(topK));
        if (topP < 1.0f) llama_sampler_chain_add(smpl, llama_sampler_init_top_p(topP, 1));
        if (minP > 0.0f) llama_sampler_chain_add(smpl, llama_sampler_init_min_p(minP, 1));
        llama_sampler_chain_add(smpl, llama_sampler_init_temp(temp));
        llama_sampler_chain_add(smpl, llama_sampler_init_dist((uint32_t) seed));   // -1 -> default random seed
    }
    for (int i = std::max(0, nTok - 64); i < nTok; i++) llama_sampler_accept(smpl, toks[i]);

    // ---- callback plumbing
    jclass cbCls = env->FindClass("com/reaper357/localllm/LlamaNative$TokenCallback");
    jmethodID onToken = env->GetMethodID(cbCls, "onToken", "([B)V");
    auto emit = [&](const std::string& s) {
        if (s.empty()) return;
        jbyteArray arr = env->NewByteArray((jsize) s.size());
        env->SetByteArrayRegion(arr, 0, (jsize) s.size(), (const jbyte*) s.data());
        env->CallVoidMethod(cb, onToken, arr);
        env->DeleteLocalRef(arr);
    };

    // ---- prompt processing in chunks
    const int nBatch = 512;
    for (int i = 0; i < nTok; i += nBatch) {
        if (g_stop) { llama_sampler_free(smpl); return nullptr; }
        int cnt = std::min(nBatch, nTok - i);
        llama_batch b = llama_batch_get_one(toks.data() + i, cnt);
        if (llama_decode(e->ctx, b) != 0) { llama_sampler_free(smpl); return env->NewStringUTF("Prompt processing failed"); }
    }

    // ---- token loop
    std::string pending;
    int pos = nTok;
    int limit = nPredict > 0 ? nPredict : INT_MAX;
    const char* err = nullptr;
    for (int i = 0; i < limit && !g_stop; i++) {
        llama_token t = llama_sampler_sample(smpl, e->ctx, -1);
        if (llama_vocab_is_eog(e->vocab, t)) break;

        char piece[256];
        int pn = llama_token_to_piece(e->vocab, t, piece, sizeof(piece), 0, true);
        if (pn > 0) pending.append(piece, pn);
        size_t cut = utf8_complete_len(pending);
        if (cut > 0) { emit(pending.substr(0, cut)); pending.erase(0, cut); }

        if (pos + 1 >= e->n_ctx) break;                 // context full
        llama_batch b = llama_batch_get_one(&t, 1);
        if (llama_decode(e->ctx, b) != 0) { err = "Generation failed"; break; }
        pos++;
    }
    llama_sampler_free(smpl);
    return err ? env->NewStringUTF(err) : nullptr;
}
