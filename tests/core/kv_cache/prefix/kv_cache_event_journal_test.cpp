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

#include "core/kv_cache/prefix/kv_cache_event_journal.h"

#include <glog/logging.h>
#include <gtest/gtest.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <thread>

#include "core/kv_cache/prefix/kv_cache_event_publisher.h"

namespace xllm {
namespace {
const std::string kFirstHash(32, '1');
const std::string kSecondHash(32, '2');
const KvCacheEventScope kDevice{"model", 0, "gpu"};
const KvCacheEventScope kHost{"model", 0, "cpu"};

class BoundPort final {
 public:
  BoundPort() : fd_(socket(AF_INET, SOCK_STREAM, 0)) {
    CHECK_GE(fd_, 0) << "test socket: " << errno;
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    CHECK_EQ(bind(fd_, reinterpret_cast<sockaddr*>(&address), sizeof(address)),
             0)
        << "test bind: " << errno;
    socklen_t length = sizeof(address);
    CHECK_EQ(getsockname(fd_, reinterpret_cast<sockaddr*>(&address), &length),
             0)
        << "test port: " << errno;
    endpoint_ = "tcp://127.0.0.1:" + std::to_string(ntohs(address.sin_port));
  }
  ~BoundPort() { release(); }
  BoundPort(const BoundPort&) = delete;
  BoundPort& operator=(const BoundPort&) = delete;
  const std::string& endpoint() const { return endpoint_; }
  void release() {
    if (fd_ >= 0) {
      close(fd_);
      fd_ = -1;
    }
  }

 private:
  int32_t fd_;
  std::string endpoint_;
};
}  // namespace

TEST(KvCacheEventJournalTest, HostCopySurvivesDeviceEviction) {
  KvCacheEventJournal journal;
  journal.store(kDevice, kFirstHash, "", {11, 12}, /*block_index=*/0);
  journal.store_copy(kHost, kFirstHash);
  journal.remove(kDevice, kFirstHash);
  const nlohmann::json state = journal.read("", 0);
  ASSERT_EQ(state["events"].size(), 1);
  EXPECT_EQ(state["events"][0]["medium"], "cpu");
  EXPECT_EQ(state["events"][0]["token_ids"], nlohmann::json({11, 12}));
  journal.store_copy(kDevice, kFirstHash);
  EXPECT_EQ(journal.read("", 0)["events"].size(), 2);
  journal.remove(kHost, kFirstHash);
  EXPECT_EQ(journal.read("", 0)["events"][0]["medium"], "gpu");
}

TEST(KvCacheEventJournalTest, DuplicateStoreDoesNotProduceAnotherEvent) {
  KvCacheEventJournal journal;
  const nlohmann::json initial = journal.read("", 0);
  journal.store(kDevice, kFirstHash, "", {1, 2}, /*block_index=*/0);
  journal.store(kDevice, kFirstHash, "", {1, 2}, /*block_index=*/0);
  const nlohmann::json page = journal.read(initial["epoch"], 0);
  ASSERT_FALSE(page["reset"].get<bool>());
  EXPECT_EQ(page["events"].size(), 1);
  EXPECT_EQ(page["cursor"], 1);
}

TEST(KvCacheEventJournalTest, OverflowAndRestartReturnCurrentSnapshot) {
  KvCacheEventJournal journal(/*capacity=*/2);
  const nlohmann::json initial = journal.read("", 0);
  journal.store(kDevice, kFirstHash, "", {1, 2}, /*block_index=*/0);
  journal.store(kDevice, kSecondHash, kFirstHash, {3, 4}, /*block_index=*/1);
  journal.remove(kDevice, kFirstHash);
  const nlohmann::json page = journal.read(initial["epoch"], 0);
  ASSERT_TRUE(page["reset"].get<bool>());
  ASSERT_EQ(page["events"].size(), 1);
  EXPECT_EQ(page["events"][0]["block_hash"], kSecondHash);
  EXPECT_EQ(page["events"][0]["parent_hash"], kFirstHash);
  EXPECT_EQ(page["cursor"], 3);
  EXPECT_TRUE(journal.read("previous-process", 3)["reset"].get<bool>());
  EXPECT_TRUE(journal.read(initial["epoch"], 100)["reset"].get<bool>());
}

TEST(KvCacheEventJournalTest, RankAndModelResidenciesAreIndependent) {
  KvCacheEventJournal journal;
  journal.store(kDevice, kFirstHash, "", {1, 2}, /*block_index=*/0);
  journal.store({"model", 1, "gpu"}, kFirstHash, "", {1, 2}, /*block_index=*/0);
  journal.store({"other", 0, "gpu"}, kFirstHash, "", {1, 2}, /*block_index=*/0);
  journal.remove(kDevice, kFirstHash);
  EXPECT_EQ(journal.read("", 0)["events"].size(), 2);
}

TEST(KvCacheEventJournalTest, IncrementalPaginationDoesNotSkipEvents) {
  KvCacheEventJournal journal;
  const nlohmann::json initial = journal.read("", 0);
  journal.store(kDevice, kFirstHash, "", {1, 2}, /*block_index=*/0);
  journal.store_copy(kHost, kFirstHash);
  journal.remove(kDevice, kFirstHash);
  const nlohmann::json first = journal.read(initial["epoch"], 0, /*limit=*/1);
  const nlohmann::json rest = journal.read(first["epoch"], first["cursor"]);
  EXPECT_EQ(first["cursor"], 1);
  EXPECT_EQ(rest["events"].size(), 2);
  EXPECT_EQ(rest["events"][1]["type"], "removed");
}

TEST(KvCacheEventJournalTest, HostSnapshotTracksEvictionAndDeviceOnlyUpdates) {
  KvCacheEventJournal journal;
  journal.store(kDevice, kFirstHash, "", {1, 2}, /*block_index=*/0);
  journal.store_copy(kHost, kFirstHash);
  const nlohmann::json host =
      journal.read("", 0, /*limit=*/1024, /*medium=*/"cpu");
  ASSERT_EQ(host["events"].size(), 1);
  EXPECT_EQ(host["events"][0]["medium"], "cpu");
  EXPECT_EQ(host["events"][0]["token_ids"], nlohmann::json({1, 2}));

  journal.remove(kHost, kFirstHash);
  const nlohmann::json removed = journal.read(
      host["epoch"], host["cursor"], /*limit=*/1, /*medium=*/"cpu");
  ASSERT_EQ(removed["events"].size(), 1);
  EXPECT_EQ(removed["events"][0]["type"], "removed");
  EXPECT_TRUE(
      journal.read("", 0, /*limit=*/1024, /*medium=*/"cpu")["events"].empty());
  EXPECT_EQ(journal.read("", 0)["events"].size(), 1);

  journal.store(kDevice, kSecondHash, kFirstHash, {3, 4}, /*block_index=*/1);
  const nlohmann::json filtered = journal.read(
      removed["epoch"], removed["cursor"], /*limit=*/1, /*medium=*/"cpu");
  EXPECT_TRUE(filtered["events"].empty());
  EXPECT_GT(filtered["cursor"], removed["cursor"]);
}

TEST(KvCacheEventJournalTest, PendingOverflowClearsClaimsAndStopsCollection) {
  KvCacheEventJournal journal(/*capacity=*/4096, /*pending_capacity=*/2);
  const nlohmann::json initial = journal.read("", 0);
  journal.store(kDevice, kFirstHash, "", {1, 2}, /*block_index=*/0);
  journal.store(kDevice, kSecondHash, kFirstHash, {3, 4}, /*block_index=*/1);
  journal.remove(kDevice, kFirstHash);
  EXPECT_FALSE(journal.enabled());
  const nlohmann::json reset = journal.read(initial["epoch"], 0);
  EXPECT_TRUE(reset["reset"].get<bool>());
  EXPECT_NE(reset["epoch"], initial["epoch"]);
  EXPECT_TRUE(reset["events"].empty());
  journal.store(kDevice, kFirstHash, "", {1, 2}, /*block_index=*/0);
  EXPECT_TRUE(journal.read("", 0)["events"].empty());
}

TEST(KvCacheEventJournalTest, InvalidMetadataClearsPreviouslyPublishedClaims) {
  KvCacheEventJournal journal;
  journal.store(kDevice, kFirstHash, "", {1, 2}, /*block_index=*/0);
  const nlohmann::json initial = journal.read("", 0);
  ASSERT_EQ(initial["events"].size(), 1);
  journal.store(kDevice, "invalid-hash", "", {3, 4}, /*block_index=*/0);
  const nlohmann::json reset =
      journal.read(initial["epoch"], initial["cursor"]);
  EXPECT_FALSE(journal.enabled());
  EXPECT_TRUE(reset["reset"].get<bool>());
  EXPECT_TRUE(reset["events"].empty());
}

TEST(KvCacheEventJournalTest,
     ConcurrentProducersPreserveResidencyDuringSnapshots) {
  KvCacheEventJournal journal;
  std::atomic<bool> done{false};
  std::thread reader([&]() {
    while (!done.load()) {
      journal.read("", 0);
    }
  });
  std::vector<std::thread> writers;
  writers.reserve(4);
  for (int32_t rank = 0; rank < 4; ++rank) {
    writers.emplace_back([&, rank]() {
      const KvCacheEventScope scope{"model", rank, "gpu"};
      for (int32_t index = 0; index < 500; ++index) {
        const std::string suffix = std::to_string(index);
        const std::string hash = std::string(32 - suffix.size(), '0') + suffix;
        journal.store(scope, hash, "", {index, index + 1}, /*block_index=*/0);
      }
    });
  }
  for (auto& writer : writers) {
    writer.join();
  }
  done.store(true);
  reader.join();
  EXPECT_TRUE(journal.enabled());
  EXPECT_EQ(journal.read("", 0)["events"].size(), 2000);
}

TEST(KvCacheEventPublisherTest,
     InvalidConfigTypeDisablesExporterWithoutTerminatingCaller) {
  auto journal = std::make_shared<KvCacheEventJournal>();
  KvCacheEventPublisher publisher(
      journal, {{"endpoint", 123}}, "model", /*dp_size=*/1);
  EXPECT_FALSE(publisher.healthy());
  EXPECT_FALSE(journal->enabled());
}

TEST(KvCacheEventPublisherTest,
     EmptyEndpointDisablesExporterWithoutTerminatingCaller) {
  auto journal = std::make_shared<KvCacheEventJournal>();
  KvCacheEventPublisher publisher(
      journal, {{"endpoint", ""}}, "model", /*dp_size=*/1);
  EXPECT_FALSE(publisher.healthy());
  EXPECT_FALSE(journal->enabled());
}

TEST(KvCacheEventPublisherTest,
     BindFailureDisablesExporterWithoutTerminatingCaller) {
  BoundPort occupied;
  auto journal = std::make_shared<KvCacheEventJournal>();
  KvCacheEventPublisher publisher(
      journal, {{"endpoint", occupied.endpoint()}}, "model", /*dp_size=*/1);
  EXPECT_FALSE(publisher.healthy());
  EXPECT_FALSE(journal->enabled());
  journal->store(kDevice, kFirstHash, "", {1, 2}, /*block_index=*/0);
  EXPECT_TRUE(journal->read("", 0)["events"].empty());
}

TEST(KvCacheEventPublisherTest,
     BackgroundMetadataFailureDoesNotTerminateCaller) {
  BoundPort reserved;
  const std::string endpoint = reserved.endpoint();
  reserved.release();
  auto journal = std::make_shared<KvCacheEventJournal>();
  KvCacheEventPublisher publisher(
      journal, {{"endpoint", endpoint}}, "model", /*dp_size=*/1);
  ASSERT_TRUE(publisher.healthy());
  journal->store(kDevice, kFirstHash, "", {1, 2}, /*block_index=*/0);
  journal->store(kDevice, kFirstHash, "", {3, 4}, /*block_index=*/0);
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(2);
  while (journal->enabled() && std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  EXPECT_FALSE(journal->enabled());
  EXPECT_FALSE(publisher.healthy());
}

TEST(KvCacheEventPublisherTest, StoredRemovedAndParentUseSameWireIdentity) {
  KvCacheEventJournal journal;
  const nlohmann::json initial = journal.read("", 0);
  journal.store(kDevice, kFirstHash, "", {1, 2}, /*block_index=*/0);
  journal.store(kDevice, kSecondHash, kFirstHash, {3, 4}, /*block_index=*/1);
  journal.remove(kDevice, kFirstHash);
  const auto events = journal.read(initial["epoch"], 0)["events"];
  const auto first = KvCacheEventPublisher::engine_event(events[0]);
  const auto second = KvCacheEventPublisher::engine_event(events[1]);
  const auto removed = KvCacheEventPublisher::engine_event(events[2]);
  EXPECT_EQ(first["block_hashes"], removed["block_hashes"]);
  EXPECT_EQ(first["block_hashes"][0], second["parent_block_hash"]);
  EXPECT_TRUE(first["parent_block_hash"].is_null());
  EXPECT_EQ(second["token_ids"], nlohmann::json({3, 4}));
  EXPECT_EQ(KvCacheEventPublisher::pool_event(events[1])["seq_hashes"],
            second["block_hashes"]);
  EXPECT_EQ(KvCacheEventPublisher::rank_endpoint("tcp://*:5557", 2),
            "tcp://*:5559");
}

}  // namespace xllm
