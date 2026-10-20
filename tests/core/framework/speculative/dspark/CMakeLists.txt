include(cc_test)

cc_test(
  NAME dspark_startup_calibration_test
  NO_NPU_RUNTIME
  SRCS
    dspark_startup_calibration_test.cpp
    "${PROJECT_SOURCE_DIR}/xllm/core/framework/speculative/dspark/dspark_startup_calibration.cpp"
    "${PROJECT_SOURCE_DIR}/xllm/core/framework/speculative/dspark/dspark_sps_cost_model.cpp"
    "${PROJECT_SOURCE_DIR}/xllm/core/framework/speculative/dspark/dspark_window_collector.cpp"
  DEPS
    nlohmann_json::nlohmann_json
    GTest::gtest_main
)

cc_test(
  NAME dspark_window_collector_test
  NO_NPU_RUNTIME
  SRCS
    dspark_window_collector_test.cpp
    "${PROJECT_SOURCE_DIR}/xllm/core/framework/speculative/dspark/dspark_budget_planner.cpp"
    "${PROJECT_SOURCE_DIR}/xllm/core/framework/speculative/dspark/dspark_window_collector.cpp"
  DEPS GTest::gtest_main
)

cc_test(
  NAME dspark_sps_cost_model_test
  NO_NPU_RUNTIME
  SRCS
    dspark_sps_cost_model_test.cpp
    "${PROJECT_SOURCE_DIR}/xllm/core/framework/speculative/dspark/dspark_sps_cost_model.cpp"
    "${PROJECT_SOURCE_DIR}/xllm/core/framework/speculative/dspark/dspark_budget_planner.cpp"
  DEPS
    nlohmann_json::nlohmann_json
    GTest::gtest_main
)

cc_test(
  NAME dspark_adaptive_cost_cache_test
  NO_NPU_RUNTIME
  SRCS
    dspark_adaptive_cost_cache_test.cpp
    "${PROJECT_SOURCE_DIR}/xllm/core/framework/speculative/dspark/dspark_adaptive_cost_cache.cpp"
  DEPS GTest::gtest_main
)

cc_test(
  NAME dspark_budget_planner_test
  NO_NPU_RUNTIME
  SRCS
    dspark_budget_planner_test.cpp
    "${PROJECT_SOURCE_DIR}/xllm/core/framework/speculative/dspark/dspark_budget_planner.cpp"
  DEPS GTest::gtest_main
)
