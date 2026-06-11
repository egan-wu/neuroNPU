/* Example: load a compiled .npubin and run it via the NeuroNPU C runtime API.
 *
 *   runtime_demo <model.npubin> [config.yaml]
 *
 * Prints the provenance header, runs a timing-only profile, then (if the model
 * has an "x" input / "y" output) does a functional run and prints a few values.
 * This is the deployment-side counterpart to the offline compiler. */
#include <stdio.h>
#include <stdlib.h>

#include "neuronpu/runtime.h"

int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: %s <model.npubin> [config.yaml]\n", argv[0]);
    return 2;
  }
  const char* cfg = argc > 2 ? argv[2] : "configs/default.yaml";
  npu_model* m = npu_open(argv[1], cfg);
  if (!m) {
    fprintf(stderr, "open failed: %s\n", npu_last_error());
    return 1;
  }

  printf("provenance: name=%s sdk=%s opt=%s\n",
         npu_meta(m, "name") ? npu_meta(m, "name") : "?",
         npu_meta(m, "sdk_version") ? npu_meta(m, "sdk_version") : "?",
         npu_meta(m, "opt") ? npu_meta(m, "opt") : "?");

  npu_profile p;
  if (npu_profile_run(m, &p) != 0) {
    fprintf(stderr, "profile failed: %s\n", npu_last_error());
    npu_close(m);
    return 1;
  }
  printf("profile: %.1f us  %.4f GMACs  MACu %.1f%%  %s-bound  %.2f mJ\n",
         p.time_ns / 1e3, p.gmacs, p.te_util * 100.0,
         p.memory_bound ? "memory" : "compute", p.energy_nj / 1e6);

  /* Optional functional run if the model exposes an x -> y_out interface.
   * Artifacts saved with embed=True bake their weights into the binary, so this
   * is numerically correct standalone; the host only binds runtime inputs. */
  int64_t nx = npu_output_numel(m, "x");
  if (nx > 0) {
    float* xb = (float*)calloc((size_t)nx, sizeof(float));
    for (int64_t i = 0; i < nx; ++i) xb[i] = 0.1f * (float)(i % 7);
    npu_set_input(m, "x", xb, nx);
    if (npu_run(m) == 0) {
      int64_t ny = npu_output_numel(m, "y_out");
      if (ny > 0) {
        float* yb = (float*)calloc((size_t)ny, sizeof(float));
        npu_get_output(m, "y_out", yb, ny);
        printf("functional: y[0..3] =");
        for (int64_t i = 0; i < ny && i < 4; ++i) printf(" %g", yb[i]);
        printf("  (numel=%lld)\n", (long long)ny);
        free(yb);
      }
    } else {
      printf("functional run skipped: %s\n", npu_last_error());
    }
    free(xb);
  }

  npu_close(m);
  return 0;
}
