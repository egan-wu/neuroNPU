/* NeuroNPU runtime — a C-callable API to load a compiled .npubin artifact and
 * execute it on the cycle-approximate simulator.
 *
 * This is the deployment-side counterpart to the offline compiler/SDK: a host
 * C/C++ app links libneuronpu_core, opens an artifact, optionally binds input
 * tensors, and either profiles it (timing-only) or runs it functionally and
 * reads outputs back. All tensor buffers are flat float32 in C order.
 *
 *   npu_model* m = npu_open("model.npubin", "configs/default.yaml");
 *   npu_set_input(m, "x", xbuf, n);
 *   npu_run(m);
 *   npu_get_output(m, "y", ybuf, npu_output_numel(m, "y"));
 *   npu_close(m);
 */
#ifndef NEURONPU_RUNTIME_H
#define NEURONPU_RUNTIME_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct npu_model npu_model;       /* opaque handle */

typedef struct {
  double time_ns;
  double cycles;
  double gmacs;
  double te_util;                 /* MAC-array utilization [0,1] */
  double ddr_gbps;                /* achieved DDR bandwidth */
  double ddr_bw_util;             /* fraction of peak DDR bandwidth */
  double arithmetic_intensity;    /* MACs per DDR byte */
  double energy_nj;
  int    memory_bound;            /* 0 = compute-bound, 1 = memory-bound */
} npu_profile;

/* Open a compiled artifact (.npubin) with an NPU config (yaml; NULL or "" =>
 * built-in defaults). Returns NULL on failure; see npu_last_error(). */
npu_model*  npu_open(const char* npubin_path, const char* config_path);
void        npu_close(npu_model* m);

/* Provenance lookup from the v6 binary header. Returns NULL if the key is
 * absent. The returned pointer is owned by the model (valid until npu_close). */
const char* npu_meta(const npu_model* m, const char* key);

/* Timing-only run: fills *out with simulator metrics. Returns 0 on success. */
int npu_profile_run(npu_model* m, npu_profile* out);

/* Functional execution. Bind inputs by descriptor name (flat float32, C order),
 * run, then read an output back. Return 0 on success, non-zero on error. */
int     npu_set_input(npu_model* m, const char* name, const float* data, int64_t n);
int     npu_run(npu_model* m);
int64_t npu_output_numel(npu_model* m, const char* name);
int     npu_get_output(npu_model* m, const char* name, float* out, int64_t n);

/* Last error message for this thread ("" if none). */
const char* npu_last_error(void);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* NEURONPU_RUNTIME_H */
