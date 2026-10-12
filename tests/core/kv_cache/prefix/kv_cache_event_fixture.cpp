/* Copyright 2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/xLLM-AI/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// CPU-only protocol fixture using the production cache and publisher. This
// exercises metadata/residency, not a model forward or a physical NPU copy.
#include <glog/logging.h>

#include <array>
#include <iostream>
#include <memory>
#include <string>

#include "core/kv_cache/block/block_manager_impl.h"
#include "core/kv_cache/prefix/kv_cache_event_publisher.h"

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  CHECK_EQ(argc, 2) << "Expected publisher configuration JSON";
  auto journal = std::make_shared<xllm::KvCacheEventJournal>();
  auto publisher = std::make_unique<xllm::KvCacheEventPublisher>(
      journal, nlohmann::json::parse(argv[1]), "fixture", /*dp_size=*/2);
  std::array<std::unique_ptr<xllm::BlockManagerImpl>, 2> devices;
  std::array<std::unique_ptr<xllm::BlockManagerImpl>, 2> hosts;
  for (int32_t rank = 0; rank < 2; ++rank) {
    xllm::BlockManager::Options options;
    options.num_blocks(64)
        .block_size(2)
        .enable_prefix_cache(true)
        .event_journal(journal)
        .event_scope({"fixture", rank, "gpu"});
    devices[rank] = std::make_unique<xllm::BlockManagerImpl>(options);
    options.event_scope({"fixture", rank, "cpu"});
    hosts[rank] = std::make_unique<xllm::BlockManagerImpl>(options);
  }
  std::string line;
  std::cout << "ready" << std::endl;
  while (std::getline(std::cin, line)) {
    const nlohmann::json command = nlohmann::json::parse(line);
    const std::string operation = command.at("op").get<std::string>();
    if (operation == "stop") {
      publisher.reset();
      std::cout << "stopped" << std::endl;
      break;
    }
    if (operation == "state") {
      std::cout << journal->read("", 0).dump() << std::endl;
      continue;
    }
    if (operation == "status") {
      std::cout << nlohmann::json({{"publisher_healthy", publisher->healthy()},
                                   {"journal_enabled", journal->enabled()}})
                       .dump()
                << std::endl;
      continue;
    }
    if (operation == "disable_events") {
      journal->disable();
      std::cout << "ok" << std::endl;
      continue;
    }
    const int32_t rank = command.at("rank").get<int32_t>();
    CHECK_GE(rank, 0);
    CHECK_LT(rank, 2);
    if (operation == "store") {
      const auto tokens = command.at("tokens").get<std::vector<int32_t>>();
      std::vector<xllm::Block> blocks =
          devices[rank]->allocate(tokens.size() / 2);
      CHECK_EQ(blocks.size(), tokens.size() / 2);
      devices[rank]->cache(tokens, blocks);
      devices[rank]->deallocate(blocks);
    } else if (operation == "copy") {
      const auto tokens = command.at("tokens").get<std::vector<int32_t>>();
      std::vector<xllm::Block> source = devices[rank]->allocate_shared(tokens);
      CHECK_EQ(source.size(), tokens.size() / 2);
      std::vector<xllm::Block> blocks = hosts[rank]->allocate(source.size());
      CHECK_EQ(blocks.size(), source.size());
      for (size_t index = 0; index < blocks.size(); ++index) {
        blocks[index].set_hash_value(source[index].get_immutable_hash_value());
      }
      hosts[rank]->cache(blocks);
      hosts[rank]->deallocate(blocks);
      devices[rank]->deallocate(source);
    } else if (operation == "clear") {
      const std::string medium = command.at("medium").get<std::string>();
      CHECK(medium == "gpu" || medium == "cpu");
      xllm::BlockManagerImpl* manager =
          (medium == "gpu" ? devices[rank] : hosts[rank]).get();
      const size_t capacity = manager->num_total_blocks();
      std::vector<xllm::Block> blocks = manager->allocate(capacity);
      CHECK_EQ(blocks.size(), capacity);
      manager->deallocate(blocks);
    } else {
      LOG(FATAL) << "Unknown fixture operation: " << operation;
    }
    std::cout << "ok" << std::endl;
  }
  return 0;
}
