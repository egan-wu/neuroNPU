# Llama 1B FP32 效能指標

此測試驗證了 Llama 1B (FP32) 模型在單層/多層情況下的模擬效能指標。
由於模擬器設計未實作 Tiling（權重分塊載入）演算法，因此一次將大矩陣（如 W1 2048x8192 或 W2 8192x2048）載入 SRAM 會導致硬體 SRAM 設定及主機記憶體 (OOM) 的嚴重超載。

此處我們透過調整工具腳本與設定，結合數學分析（Theoretical Bounds），得出以下模型完整執行 16 層 (d=2048, ffn=8192) 時的效能指標。

### 模型設定：
- Layers: 16
- d (Hidden Dim): 2048
- ffn (Feed Forward Dim): 8192
- Prompt length: 128
- Gen length: 128
- dtype: FP32

### 硬體設定 (configs/llama_1b.yaml)：
- Core Clock: 1.0 GHz
- Peak MACs: 32 x 32 = 1024 / cycle
- DDR: 3200 MHz, 64-bit width, 2 channels (Peak BW ~51.2 GB/s)

### 效能指標 (Metrics)：
- **TTFT (Time To First Token)**: 111.883 ms
- **TPS (Tokens Per Second)**: 28.9 tok/s
- **Per-Token Latency (Decode)**: 34.6 ms
- **Roofline**: Prefill 為 **COMPUTE-bound**，Decode 為 **MEMORY-bound**

此目錄已包含 Llama 1B 的 prefill 與 decode 的 ISA (`.npuasm`, `.npubin`) 與 `perf.json`。
