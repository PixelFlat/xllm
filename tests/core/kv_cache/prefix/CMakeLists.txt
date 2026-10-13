include(cc_test)

cc_test(
  NAME
    kv_cache_event_journal_test
  SRCS
    kv_cache_event_journal_test.cpp
  DEPS
    :kv_cache_events
    GTest::gtest_main
)
target_link_options(kv_cache_event_journal_test PRIVATE
  "-Wl,--wrap=zmq_msg_recv")
if(ZMQ_INCLUDE_DIR AND ZMQ_LIBRARY)
  target_include_directories(kv_cache_event_journal_test SYSTEM PRIVATE
    ${ZMQ_INCLUDE_DIR})
  target_link_libraries(kv_cache_event_journal_test PRIVATE ${ZMQ_LIBRARY})
else()
  find_package(ZeroMQ CONFIG REQUIRED)
  if(TARGET libzmq-static)
    target_link_libraries(kv_cache_event_journal_test PRIVATE libzmq-static)
  else()
    target_link_libraries(kv_cache_event_journal_test PRIVATE libzmq)
  endif()
endif()

include(cc_binary)
cc_binary(
  NAME
    kv_cache_event_fixture
  SRCS
    kv_cache_event_fixture.cpp
  DEPS
    :block
    :kv_cache_events
    :xllm_server
)
target_include_directories(kv_cache_event_fixture PRIVATE
  ${PROJECT_SOURCE_DIR}/xllm)
target_link_libraries(kv_cache_event_fixture PRIVATE
  "$<LINK_GROUP:RESCAN,xtensor,xllm_server>"
  brpc OpenSSL::SSL OpenSSL::Crypto Folly::folly c_sec)
add_dependencies(kv_cache_event_fixture brpc-static)

cc_test(
  NAME
    prefix_test
  SRCS
    prefix_cache_test.cpp
  DEPS
    :config
    :kv_cache
    :prefix_cache
    :block
    absl::random_random
    Boost::serialization
    GTest::gtest_main
)
target_link_libraries(prefix_test PRIVATE brpc OpenSSL::SSL OpenSSL::Crypto Folly::folly :xllm_server)
add_dependencies(prefix_test brpc-static)

cc_test(
  NAME
    linear_state_prefix_cache_test
  SRCS
    linear_state_prefix_cache_test.cpp
  DEPS
    :block
    GTest::gtest_main
)
