cmake_minimum_required(VERSION 3.16)
project(neuronpu CXX)

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
  src/core.cpp)
target_include_directories(neuronpu_core PUBLIC include)

add_executable(neuronpu src/main.cpp)
target_link_libraries(neuronpu PRIVATE neuronpu_core)

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
  add_test(NAME llm_demo
           COMMAND ${PYTHON3} ${CMAKE_SOURCE_DIR}/tools/llm_demo.py
                   --neuronpu $<TARGET_FILE:neuronpu>
                   --config ${CMAKE_SOURCE_DIR}/configs/default.yaml
                   --outdir ${CMAKE_BINARY_DIR}/llm
                   --layers 2 --d 32 --ffn 64 --prompt 16 --gen 4)
endif()
