package com.reaper357.localllm;

import android.os.Bundle;
import com.getcapacitor.BridgeActivity;

public class MainActivity extends BridgeActivity {
    @Override
    public void onCreate(Bundle savedInstanceState) {
        registerPlugin(LlamaBridgePlugin.class);
        super.onCreate(savedInstanceState);
    }
}
