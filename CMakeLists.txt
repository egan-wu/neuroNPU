cmake_minimum_required(VERSION 3.16)
project(neuronpu C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
add_compile_options(-Wall -Wextra)

add_library(neuronpu_core
  src/types.cpp
  src/config.cpp
  src/isa.cpp
  src/memory.cpp
  src/logger.cpp
  src/perf.cpp
  src/core.cpp
  src/runtime.cpp)
target_include_directories(neuronpu_core PUBLIC include)

add_executable(neuronpu src/main.cpp)
target_link_libraries(neuronpu PRIVATE neuronpu_core)

# Example host app using the C runtime API (deployment-side).
add_executable(runtime_demo examples/runtime_demo.c)
target_link_libraries(runtime_demo PRIVATE neuronpu_core)
set_target_properties(runtime_demo PROPERTIES LINKER_LANGUAGE CXX)  # C app, C++ lib

enable_testing()
add_test(NAME matmul_slice
         COMMAND neuronpu run ${CMAKE_SOURCE_DIR}/tests/isa_programs/matmul.npuasm
                 --config ${CMAKE_SOURCE_DIR}/configs/default.yaml
                 --out ${CMAKE_BINARY_DIR}/logs)
add_test(NAME vector_ops
         COMMAND neuronpu run ${CMAKE_SOURCE_DIR}/tests/isa_programs/vector_ops.npuasm
                 --config ${CMAKE_SOURCE_DIR}/configs/default.yaml
                 --out ${CMAKE_BINARY_DIR}/logs)
add_test(NAME conv
         COMMAND neuronpu run ${CMAKE_SOURCE_DIR}/tests/isa_programs/conv.npuasm
                 --config ${CMAKE_SOURCE_DIR}/configs/default.yaml
                 --out ${CMAKE_BINARY_DIR}/logs)
add_test(NAME dma_contention
         COMMAND neuronpu run ${CMAKE_SOURCE_DIR}/tests/isa_programs/dma_contention.npuasm
                 --config ${CMAKE_SOURCE_DIR}/configs/default.yaml
                 --out ${CMAKE_BINARY_DIR}/logs)
add_test(NAME rope
         COMMAND neuronpu run ${CMAKE_SOURCE_DIR}/tests/isa_programs/rope.npuasm
                 --config ${CMAKE_SOURCE_DIR}/configs/default.yaml
                 --out ${CMAKE_BINARY_DIR}/logs)
add_test(NAME sram_contention
         COMMAND neuronpu run ${CMAKE_SOURCE_DIR}/tests/isa_programs/matmul.npuasm
                 --config ${CMAKE_SOURCE_DIR}/configs/sram_constrained.yaml
                 --out ${CMAKE_BINARY_DIR}/logs)
add_test(NAME loop
         COMMAND neuronpu run ${CMAKE_SOURCE_DIR}/tests/isa_programs/loop.npuasm
                 --config ${CMAKE_SOURCE_DIR}/configs/default.yaml
                 --out ${CMAKE_BINARY_DIR}/logs)
add_test(NAME multicore
         COMMAND neuronpu run ${CMAKE_SOURCE_DIR}/tests/isa_programs/multicore.npuasm
                 --config ${CMAKE_SOURCE_DIR}/configs/dual_core.yaml
                 --out ${CMAKE_BINARY_DIR}/logs)
add_test(NAME nested_loop
         COMMAND neuronpu run ${CMAKE_SOURCE_DIR}/tests/isa_programs/nested_loop.npuasm
                 --config ${CMAKE_SOURCE_DIR}/configs/default.yaml
                 --out ${CMAKE_BINARY_DIR}/logs)
add_test(NAME transformer_block
         COMMAND neuronpu run ${CMAKE_SOURCE_DIR}/tests/isa_programs/transformer_block.npuasm
                 --config ${CMAKE_SOURCE_DIR}/configs/default.yaml
                 --out ${CMAKE_BINARY_DIR}/logs)
add_test(NAME elementwise
         COMMAND neuronpu run ${CMAKE_SOURCE_DIR}/tests/isa_programs/elementwise.npuasm
                 --config ${CMAKE_SOURCE_DIR}/configs/default.yaml
                 --out ${CMAKE_BINARY_DIR}/logs)
add_test(NAME causal_softmax
         COMMAND neuronpu run ${CMAKE_SOURCE_DIR}/tests/isa_programs/causal_softmax.npuasm
                 --config ${CMAKE_SOURCE_DIR}/configs/default.yaml
                 --out ${CMAKE_BINARY_DIR}/logs)
add_test(NAME gather
         COMMAND neuronpu run ${CMAKE_SOURCE_DIR}/tests/isa_programs/gather.npuasm
                 --config ${CMAKE_SOURCE_DIR}/configs/default.yaml
                 --out ${CMAKE_BINARY_DIR}/logs)
add_test(NAME timing_only
         COMMAND neuronpu run ${CMAKE_SOURCE_DIR}/tests/isa_programs/transformer_block.npuasm
                 --config ${CMAKE_SOURCE_DIR}/configs/default.yaml
                 --out ${CMAKE_BINARY_DIR}/logs --timing-only)

find_program(PYTHON3 python3)
if(PYTHON3)
  add_test(NAME compiler_validate
           COMMAND ${PYTHON3} -m compiler.validate
                   --neuronpu $<TARGET_FILE:neuronpu>
                   --config ${CMAKE_SOURCE_DIR}/configs/default.yaml
                   --workdir ${CMAKE_BINARY_DIR}/compiler
           WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
  add_test(NAME compiler_validate_tiled
           COMMAND ${PYTHON3} -m compiler.validate
                   --neuronpu $<TARGET_FILE:neuronpu>
                   --config ${CMAKE_SOURCE_DIR}/configs/default.yaml
                   --workdir ${CMAKE_BINARY_DIR}/compiler_t
                   --fuse --tile --tile-budget 16 --nbuf 2
           WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
  add_test(NAME llama_profile
           COMMAND ${PYTHON3} -m compiler.profile_llama
                   --config-json ${CMAKE_SOURCE_DIR}/tests/llama_tiny_config.json
                   --neuronpu $<TARGET_FILE:neuronpu>
                   --config ${CMAKE_SOURCE_DIR}/configs/default.yaml
                   --workdir ${CMAKE_BINARY_DIR}/llama --prompt 16 --gen 4 --kv 16 64
           WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
  add_test(NAME trace_analyze
           COMMAND ${PYTHON3} -c "import subprocess,sys; subprocess.run([sys.executable,'-m','compiler.profile_llama','--config-json','tests/llama_tiny_config.json','--neuronpu','$<TARGET_FILE:neuronpu>','--config','configs/default.yaml','--workdir','${CMAKE_BINARY_DIR}/llama_t','--prompt','16','--gen','2','--kv','16','--tile'],check=True); subprocess.run([sys.executable,'-m','compiler.analyze','${CMAKE_BINARY_DIR}/llama_t/prefill'],check=True)"
           WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
  add_test(NAME int8_functional
           COMMAND ${PYTHON3} -c "import numpy as np; from compiler.models import tiny_llama_layer; from compiler.driver import compile_and_run; from compiler import reference, passes; g=tiny_llama_layer(8,32,64); x=(np.random.default_rng(1).standard_normal((8,32))*0.5).astype(np.float32); gq=passes.quantize(g,{'x':x}); rq=reference.execute(gq,{'x':x})['y']; got,_=compile_and_run(gq,{'x':x},'$<TARGET_FILE:neuronpu>','${CMAKE_SOURCE_DIR}/configs/default.yaml','${CMAKE_BINARY_DIR}/q8',opt={'reuse':True}); exit(0 if np.max(np.abs(got['y']-rq))<1e-3 else 1)"
           WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
  add_test(NAME quant_profile
           COMMAND ${PYTHON3} -m compiler.profile_quant
                   --config-json ${CMAKE_SOURCE_DIR}/tests/llama_tiny_config.json
                   --neuronpu $<TARGET_FILE:neuronpu>
                   --config ${CMAKE_SOURCE_DIR}/configs/default.yaml
                   --workdir ${CMAKE_BINARY_DIR}/quant --kv 16 --layers 2
           WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
  add_test(NAME yolo_block
           COMMAND ${PYTHON3} -c "import numpy as np; from compiler.models import yolo_block; from compiler.driver import validate; g=yolo_block(); x=(np.random.default_rng(2).standard_normal((3,16,16))*0.5).astype(np.float32); ok,_=validate(g,{'x':x},'$<TARGET_FILE:neuronpu>','${CMAKE_SOURCE_DIR}/configs/default.yaml','${CMAKE_BINARY_DIR}/yolo',opt={'reuse':True,'fuse':True}); exit(0 if ok else 1)"
           WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
  add_test(NAME sdk_api
           COMMAND ${PYTHON3} -c "import numpy as np; import compiler.api as npu; from compiler.models import tiny_llama_layer; g=tiny_llama_layer(4,8,16); x=(np.random.default_rng(1).standard_normal((4,8))*0.5).astype(np.float32); m=npu.compile(g,opt='O1',neuronpu='$<TARGET_FILE:neuronpu>',config='${CMAKE_SOURCE_DIR}/configs/default.yaml',workdir='${CMAKE_BINARY_DIR}/sdk'); assert m.validate({'x':x}); p=m.profile(); bp='${CMAKE_BINARY_DIR}/sdk/tiny.npubin'; m.save(bp); import os; os.remove(bp+'.meta.json'); a=npu.load(bp); assert a.meta['name']==g.name, 'in-band provenance lost'; assert a.meta['sdk_version']==npu.SDK_VERSION; assert abs(a.profile(out_dir='${CMAKE_BINARY_DIR}/sdk/ld').time_ns-p.time_ns)<1; mq=npu.compile(g,opt='O0',quant=npu.Int8({'x':x}),neuronpu='$<TARGET_FILE:neuronpu>',config='${CMAKE_SOURCE_DIR}/configs/default.yaml',workdir='${CMAKE_BINARY_DIR}/sdkq'); assert mq.validate({'x':x}); print('SDK API OK')"
           WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
  add_test(NAME cli_tool
           COMMAND ${PYTHON3} -c "import subprocess,sys; import compiler.api as npu; from compiler.models import tiny_llama_layer; g=tiny_llama_layer(4,8,16); npu.compile(g,opt='O1',neuronpu='$<TARGET_FILE:neuronpu>',config='${CMAKE_SOURCE_DIR}/configs/default.yaml',workdir='${CMAKE_BINARY_DIR}/cli').save('${CMAKE_BINARY_DIR}/cli/m.npubin'); subprocess.run([sys.executable,'-m','compiler.cli','profile','${CMAKE_BINARY_DIR}/cli/m.npubin','--neuronpu','$<TARGET_FILE:neuronpu>','--config','${CMAKE_SOURCE_DIR}/configs/default.yaml','--workdir','${CMAKE_BINARY_DIR}/cli'],check=True); subprocess.run([sys.executable,'-m','compiler.cli','--version'],check=True)"
           WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
  add_test(NAME runtime_api
           COMMAND ${PYTHON3} -c "import numpy as np,subprocess,re,compiler.api as npu; from compiler.models import tiny_llama_layer; from compiler import reference; g=tiny_llama_layer(4,8,16); x=(0.1*(np.arange(32)%7)).reshape(4,8).astype('float32'); exp=reference.execute(g,{'x':x})['y'].ravel(); m=npu.compile(g,opt='O1',neuronpu='$<TARGET_FILE:neuronpu>',config='${CMAKE_SOURCE_DIR}/configs/default.yaml',workdir='${CMAKE_BINARY_DIR}/rtapi'); m.save('${CMAKE_BINARY_DIR}/rtapi/m.npubin'); out=subprocess.run(['$<TARGET_FILE:runtime_demo>','${CMAKE_BINARY_DIR}/rtapi/m.npubin','${CMAKE_SOURCE_DIR}/configs/default.yaml'],capture_output=True,text=True,check=True).stdout; assert 'name=tiny_llama_layer' in out, out; nums=[float(v) for v in re.search(r'y.0..3. =([^(]*)',out).group(1).split()]; assert np.allclose(nums,exp[:4],atol=1e-4), (nums,exp[:4]); print('RUNTIME API OK (weights baked in, standalone == numpy)')"
           WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
  add_test(NAME llm_demo
           COMMAND ${PYTHON3} ${CMAKE_SOURCE_DIR}/tools/llm_demo.py
                   --neuronpu $<TARGET_FILE:neuronpu>
                   --config ${CMAKE_SOURCE_DIR}/configs/default.yaml
                   --outdir ${CMAKE_BINARY_DIR}/llm
                   --layers 2 --d 32 --ffn 64 --prompt 16 --gen 4)
endif()
