#pragma once

#include <functional>
#include <shared_mutex>
#include <thread>
#include "chess/callbacks.h"
#include "chess/uciloop.h"
#include "mcts/search.h"
#include "mcts/node.h"
#include "mcts/params.h"
#include "neural/cache.h"
#include "neural/network.h"
#include "syzygy/syzygy.h"
#include "utils/logging.h"
#include "utils/mutex.h"
#include "utils/optional.h"

namespace lczero {


enum TTScoreType : uint8_t {
  kScoreExact = 0,
  kScoreGE = 1,
  kScoreLE = 2,
};

struct TTEntry {
  union {
    uint64_t hashKey;
    struct {
      // 16 LSB's are not important as the hash table size is at least > 64k
      // entries
      uint16_t age;
      uint8_t hashPart[6];  // most significant bits of the hash key
    };
  };  // 64 bits

  union {
    uint64_t otherInfo;
    struct {
      Move bestMove;          // 16 bits
      TTScoreType scoreType;  // 8 bits
      uint8_t depth;          // 8 bits
      float score;            // 32 bits
    };
  };
};

static_assert(sizeof(TTEntry) == 16,
              "Check size of TT Entry, assumed to be 16 bytes!");

class TranspositionTable {
 private:
  static const int kDefaultTTSize = 256 * 1024 * 1024;
  TTEntry* tt_;  // the transposition table

  uint64_t size_;       // size in elements (should be a power of 2)
  uint64_t indexBits_;  // size-1
  uint64_t hashBits_;   // ALLSET ^ indexBits;

 public:
  void init(uint64_t byteSize = kDefaultTTSize);
  void destroy();
  void reset();

  bool lookup(uint64_t hash, int searchDepth, float* score,
              TTScoreType* scoreType, int* foundDepth, Move* bestMove);
  void update(uint64_t hash, float score, TTScoreType scoreType, Move bestMove,
              int depth, int age);
};


// Ankan - design idea : maybe make this (and MCTS search) inherit from some common Search class, that has virtual functions
class AlphaBetaSearch {
 public:
  AlphaBetaSearch(NodeTree& tree, Network* network,
                  BestMoveInfo::Callback best_move_callback,
                  ThinkingInfo::Callback info_callback, const SearchLimits& limits,
                  const OptionsDict& options, NNCache* cache,
                  SyzygyTablebase* syzygy_tb,
                  TranspositionTable& tt);

  ~AlphaBetaSearch();

  // Starts worker threads and returns immediately.
  void StartThreads(size_t how_many);

  // Stops search. At the end bestmove will be returned. The function is not
  // blocking, so it returns before search is actually done.
  void Stop();
  // Stops search, but does not return bestmove. The function is not blocking.
  void Abort();
  // Blocks until all worker thread finish.
  void Wait();
  // Returns whether search is active. Workers check that to see whether another
  // search iteration is needed.
  bool IsSearchActive() const;

  // Returns best move
  std::pair<Move, Move> GetBestMove();

  // Returns the evaluation of the best move
  float GetBestEval() const;

  // Returns the total number of nodes in the search (alpha beta + MCTS)
  std::int64_t GetTotalNodes() const;

  // Returns the search parameters.
  const SearchParams& GetParams() const { return params_; }

 private:

  float AlphaBeta(int depth, float alpha, float beta, Move *bestMove);

  void SortMoves(MoveList &list, std::vector<float> &pValues);
  int nodes_;

  void OnBestMove(const BestMoveInfo& move);
  void OnInfo(const std::vector<ThinkingInfo>& infos);

  std::atomic<bool> stop_{false};

  // TODO: get rid of these!
  NodeTree* tree_;
  const OptionsDict* options_;

  Node* root_node_;

  // not used by alpha-beta search, but passed on to MCTS search at nodes
  NNCache* cache_;
  SyzygyTablebase* syzygy_tb_;
  const PositionHistory& played_history_;


  PositionHistory history_;

  Network* const network_;
  const SearchLimits limits_;

  const std::chrono::steady_clock::time_point start_time_;

  const int64_t initial_visits_;    // TODO: figure out how is this used?

  BestMoveInfo::Callback best_move_callback_;
  ThinkingInfo::Callback info_callback_;
  const SearchParams params_;

  std::vector<std::thread> threads_;

  void SendUciInfo();

  int64_t GetTimeSinceStart() const;
  int64_t GetTimeToDeadline() const;

  // Function which runs in a separate thread and runs the search
  void SearchThread();

  bool timeout_;
  TranspositionTable &tt_;

}; // class AlphaBetaSearch

}  // namespace lczero
