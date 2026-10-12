include(cc_binary)
include(cc_library)

cc_library(
  NAME
    kv_cache_events
  HDRS
    kv_cache_event_journal.h
    kv_cache_event_publisher.h
  SRCS
    kv_cache_event_journal.cpp
    kv_cache_event_publisher.cpp
  DEPS
    glog::glog
    nlohmann_json::nlohmann_json
    Threads::Threads
)
if(ZMQ_INCLUDE_DIR AND ZMQ_LIBRARY)
  # Explicit development-library paths remain available for offline builds.
  target_include_directories(kv_cache_events SYSTEM PRIVATE ${ZMQ_INCLUDE_DIR})
  target_link_libraries(kv_cache_events PRIVATE ${ZMQ_LIBRARY})
else()
  find_package(ZeroMQ CONFIG REQUIRED)
  if(TARGET libzmq-static)
    target_link_libraries(kv_cache_events PRIVATE libzmq-static)
  else()
    target_link_libraries(kv_cache_events PRIVATE libzmq)
  endif()
endif()

cc_library(
  NAME
    block_hasher
  HDRS
    block_hasher.h
  SRCS
    block_hasher.cpp
  DEPS
    :config
    :multimodal
    glog::glog
    torch
    xxHash
)

cc_library(
  NAME
    prefix_cache
  HDRS
    prefix_cache.h
    linear_state_prefix_cache.h
    prefix_cache_factory.h
  SRCS
    prefix_cache.cpp
    linear_state_prefix_cache.cpp
    prefix_cache_factory.cpp
  DEPS
    :kv_cache_events
    :block_hasher
    :config
    :kv_cache_block_types
    :multimodal
    :common
    glog::glog
    absl::strings
    absl::time
    Boost::serialization
    SMHasherSupport
    torch
    xxHash
)




cc_binary(
  NAME
    prefix_cache_benchmark
  SRCS
    prefix_cache_benchmark.cpp
  DEPS
    torch_python
    :kv_cache
    :prefix_cache
    :kv_cache_block
    benchmark::benchmark
    benchmark::benchmark_main
    :xllm_server
    SMHasherSupport
    xxHash
)

target_link_libraries(prefix_cache_benchmark PRIVATE brpc OpenSSL::SSL OpenSSL::Crypto)
add_dependencies(prefix_cache_benchmark brpc-static)
