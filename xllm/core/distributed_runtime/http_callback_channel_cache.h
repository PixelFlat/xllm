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

#pragma once

#include <cstddef>
#include <deque>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <utility>

namespace brpc {
class Channel;
}  // namespace brpc

namespace xllm {

// Callback endpoints come from requests, rather than service discovery. Bound
// their cache independently of channels owned by management RPC stubs.
class HttpCallbackChannelCache final {
 public:
  static constexpr size_t kCapacity = 128;

  std::shared_ptr<brpc::Channel> find(const std::string& address) const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    const auto it = channels_.find(address);
    return it == channels_.end() ? nullptr : it->second;
  }

  std::shared_ptr<brpc::Channel> insert(
      std::string address,
      std::shared_ptr<brpc::Channel> channel) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    const auto existing = channels_.find(address);
    if (existing != channels_.end()) {
      return existing->second;
    }
    if (channels_.size() == kCapacity) {
      // In-flight calls retain shared ownership after FIFO eviction.
      channels_.erase(order_.front());
      order_.pop_front();
    }
    order_.emplace_back(address);
    return channels_.emplace(std::move(address), std::move(channel))
        .first->second;
  }

  void erase(const std::string& address) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    if (channels_.erase(address) > 0) {
      std::erase(order_, address);
    }
  }

  size_t size() const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    return channels_.size();
  }

 private:
  mutable std::shared_mutex mutex_;
  std::unordered_map<std::string, std::shared_ptr<brpc::Channel>> channels_;
  std::deque<std::string> order_;
};

}  // namespace xllm
