#include "alphaBeta/alphaBetaSearch.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <sstream>
#include <thread>

#include "mcts/node.h"
#include "neural/cache.h"
#include "neural/encoder.h"
#include "utils/random.h"

namespace lczero {

void AlphaBetaSearch::SortMoves(MoveList& list, std::vector<float>& pValues) {
  // simple bubble sort
  bool swapped = true;
  while (swapped) {
    swapped = false;
    for (std::vector<float>::size_type i = 0; i < pValues.size() - 1; i++) {
      if (pValues[i + 1] > pValues[i]) {
        float tempVal = pValues[i];
        pValues[i] = pValues[i + 1];
        pValues[i + 1] = tempVal;

        Move tempMove = list[i];
        list[i] = list[i + 1];
        list[i + 1] = tempMove;

        swapped = true;
      }
    }
  }
}

float AlphaBetaSearch::AlphaBeta(int depth, float alpha, float beta,
                                 Move* bestMove) {
  const auto& board = history_.Last().GetBoard();

  // check for timeout
  if (limits_.search_deadline && GetTimeToDeadline() < 0) {
    // timeout_ = true;
    // return 0;
  }

  auto legal_moves = board.GenerateLegalMoves();

  // detect terminal nodes
  if (legal_moves.empty()) {
    // Could be a checkmate or a stalemate
    if (board.IsUnderCheck()) {
      return -1.0f;
    } else {
      return 0;
    }
  }

  if (!board.HasMatingMaterial()) {
    return 0;
  }

  if (history_.Last().GetNoCaptureNoPawnPly() >= 100) {
    return 0;
  }

  if (history_.Last().GetRepetitions() >= 2) {
    return 0;
  }

  // lookup in the transposition table
  uint64_t hash = board.Hash();
  int hashDepth = 0;
  float hashScore = 0;
  TTScoreType scoreType;
  Move ttMove = {};
  bool foundInTT =
      tt_.lookup(hash, depth, &hashScore, &scoreType, &hashDepth, &ttMove);
  if (foundInTT && hashDepth >= depth) {
    *bestMove = ttMove;

    // exact score at same or better depth => done, return TT value
    if (scoreType == kScoreExact) {
      return hashScore;
    }

    // score at same or better depth causes beta cutoff - again return TT
    // value
    if (scoreType == kScoreGE && hashScore >= beta) {
      return hashScore;
    }

    // score causes alpha-cutoff
    if (scoreType == kScoreLE && hashScore <= alpha) {
      return hashScore;
    }
  }

  float currentMax = -1;
  int movesSearched = 0;
  bool improvedAlpha = false;

  // try hash move first - before calling NN
  if ((ttMove.as_packed_int() != Move().as_packed_int()) && (depth != 0)) {
    history_.Append(ttMove);
    tree_->MakeMove(ttMove);

    Move nextBest;
    float curScore = -AlphaBeta(depth - 1, -beta, -alpha, &nextBest);

    tree_->PopHistory();
    tree_->TrimTreeAtHead();
    history_.Pop();

    if (timeout_) return 0;

    if (curScore >= beta) {
      tt_.update(hash, curScore, kScoreGE, ttMove, depth, history_.GetLength());
      return curScore;
    }

    if (curScore > currentMax) {
      currentMax = curScore;
      *bestMove = ttMove;
      if (currentMax > alpha) {
        alpha = currentMax;
        improvedAlpha = true;
      }
    }
  }

  // TODO: support syzygy_tb_

  nodes_++;

  float v;

  if (depth == 0) {
  // TODO: call into MCTS here
#if 1
    SearchLimits limits;
    limits.visits = 800;

    auto search = std::make_unique<Search>(
        *tree_, network_,
        std::bind(&AlphaBetaSearch::OnBestMove, this, std::placeholders::_1),
        std::bind(&AlphaBetaSearch::OnInfo, this, std::placeholders::_1),
        limits, *options_, cache_, syzygy_tb_);

    search->StartThreads(1);
    search->Wait();

    v = search->GetBestEval();
    //if (history_.IsBlackToMove()) v = -v;
#endif

#if 0
    auto computation = network_->NewComputation();
    auto planes = EncodePositionForNN(history_, 8, params_.GetHistoryFill());
    computation->AddInput(std::move(planes));
    computation->ComputeBlocking();

    v = computation->GetQVal(0);
#endif

    tt_.update(hash, v, kScoreExact, Move(), depth, history_.GetLength());

    return v;  // TODO: see if the score returned needs any adjustment (e.g,
               // relative to black or white?)
  }

  // NN eval to get policy and value
  auto computation = network_->NewComputation();
  auto planes = EncodePositionForNN(history_, 8, params_.GetHistoryFill());
  computation->AddInput(std::move(planes));
  computation->ComputeBlocking();

  v = computation->GetQVal(0);
  // printf("\nvalue head score: %f\n", v);

  // sort move list based on output of NN policy head
  std::vector<float> pVals;
  for (auto move : legal_moves) {
    pVals.emplace_back(computation->GetPVal(0, move.as_nn_index()));
  }
  SortMoves(legal_moves, pVals);

  for (auto move : legal_moves) {
    if (move == ttMove) continue;  // already tried before

    // history is now redundant!
    history_.Append(move);
    tree_->MakeMove(move);

    Move nextBest;
    float curScore = -AlphaBeta(depth - 1, -beta, -alpha, &nextBest);

    tree_->PopHistory();
    tree_->TrimTreeAtHead();
    history_.Pop();

    if (timeout_) return 0;

    if (curScore >= beta) {
      tt_.update(hash, curScore, kScoreGE, move, depth, history_.GetLength());
      return curScore;
    }

    if (curScore > currentMax) {
      currentMax = curScore;
      *bestMove = move;
      if (currentMax > alpha) {
        alpha = currentMax;
        improvedAlpha = true;
      }
    }
  }

  // default node type is ALL node and the score returned is a upper bound on
  // the score of the node
  if (improvedAlpha) {
    scoreType = kScoreExact;
  } else {
    // ALL node
    scoreType = kScoreLE;
  }
  tt_.update(hash, currentMax, scoreType, *bestMove, depth,
             history_.GetLength());

  return currentMax;
}

AlphaBetaSearch::AlphaBetaSearch(NodeTree& tree, Network* network,
                                 BestMoveInfo::Callback best_move_callback,
                                 ThinkingInfo::Callback info_callback,
                                 const SearchLimits& limits,
                                 const OptionsDict& options, NNCache* cache,
                                 SyzygyTablebase* syzygy_tb,
                                 TranspositionTable& tt)
    : root_node_(tree.GetCurrentHead()),
      cache_(cache),
      syzygy_tb_(syzygy_tb),
      played_history_(tree.GetPositionHistory()),
      history_(played_history_),
      network_(network),
      limits_(limits),
      start_time_(std::chrono::steady_clock::now()),
      initial_visits_(root_node_->GetN()),
      best_move_callback_(best_move_callback),
      info_callback_(info_callback),
      tree_(&tree),
      tt_(tt),
      options_(&options),
      params_(options) {}

void AlphaBetaSearch::StartThreads(size_t how_many) {
  // Start the search thread
  threads_.emplace_back([this]() { SearchThread(); });
}

void AlphaBetaSearch::OnBestMove(const BestMoveInfo& move) {
  // std::cout << "bestmove " << move.bestmove.as_string() << std::endl;
  // printf("\nbest move from backend!: %s\n",
  // move.bestmove.as_string().c_str());
}

void AlphaBetaSearch::OnInfo(const std::vector<ThinkingInfo>& infos) {
  // infos[0].score;
  // ignore info
}

void AlphaBetaSearch::SearchThread() {
  printf("starting search thread\n");

  // NodeTree tree;
  // tree.ResetToPosition(option_dict.Get<std::string>(kFenId.GetId()), {});

#if 0
  auto search = std::make_unique<Search>(
      *tree_, network_,
      std::bind(&AlphaBetaSearch::OnBestMove, this, std::placeholders::_1),
      std::bind(&AlphaBetaSearch::OnInfo, this, std::placeholders::_1), limits_,
      *options_, cache_, syzygy_tb_);

  search->StartThreads(1);
  search->Wait();
#endif

  nodes_ = 0;
  float alpha = -1.0f, beta = 1.0f;
  Move bestMove;

  // iterative deepening
  int depth = 0;
  float val = 0;
  timeout_ = false;
  while (true) {
    depth++;

    Move currentBest;
    val = AlphaBeta(depth, alpha, beta, &currentBest);

    if (timeout_) break;

    bestMove = currentBest;

    // send uci info,
    std::vector<ThinkingInfo> uci_infos;
    ThinkingInfo common_info;
    common_info.depth = depth;
    common_info.seldepth = depth;
    common_info.time = GetTimeSinceStart();
    common_info.nodes = nodes_;  // Ankan - test
    common_info.hashfull = 0;
    common_info.nps = nodes_ * 1000.0 / GetTimeSinceStart();
    common_info.tb_hits = 0;

    common_info.score = 290.680623072 * tan(1.548090806 * val);

    common_info.pv.push_back(bestMove);
    uci_infos.emplace_back(common_info);
    info_callback_(uci_infos);
  }

  // printf("\nval: %f, bestMove: %s\n", val, bestMove.as_string().c_str());
  // printf("\nNodes evaluated: %d, time: %llu ms\n", nodes_,
  // GetTimeSinceStart());

  // send and best move
  best_move_callback_({bestMove});

#if 0
  //while (1) {
  for (int i=0;i<10;i++) {
    printf("loop iter\n");
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));

    SendUciInfo();

    // exit if stop was called
    if (stop_.load(std::memory_order_acquire)) break;
  }
#endif
}

void AlphaBetaSearch::SendUciInfo() /*REQUIRES(nodes_mutex_)*/ {
  std::vector<ThinkingInfo> uci_infos;

  // Info common for all multipv variants.
  ThinkingInfo common_info;
  common_info.depth = 7;  // Ankan - test!
  common_info.seldepth = 11;
  common_info.time = GetTimeSinceStart();
  common_info.nodes = 777;  // Ankan - test
  common_info.hashfull = 0;
  common_info.nps = 1111;
  common_info.tb_hits = 0;

  common_info.score = 256;

  common_info.pv.push_back(Move(BoardSquare("a1"), BoardSquare("a5")));
  common_info.pv.push_back(Move(BoardSquare("b2"), BoardSquare("b3")));
  common_info.pv.push_back(Move(BoardSquare("c7"), BoardSquare("c5")));

  uci_infos.emplace_back(common_info);

  info_callback_(uci_infos);

  // best_move_callback_(
  // {final_bestmove_.GetMove(played_history_.IsBlackToMove()),
  // final_pondermove_.GetMove(!played_history_.IsBlackToMove())});
}

int64_t AlphaBetaSearch::GetTimeSinceStart() const {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::steady_clock::now() - start_time_)
      .count();
}

int64_t AlphaBetaSearch::GetTimeToDeadline() const {
  if (!limits_.search_deadline) return 0;
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             *limits_.search_deadline - std::chrono::steady_clock::now())
      .count();
}

void AlphaBetaSearch::Stop() {
  stop_.store(true, std::memory_order_release);
  timeout_ = true;
  printf("Stopping search\n");
}

void AlphaBetaSearch::Abort() {
  Stop();
  Wait();
  printf("Aborting search, if it is still active.\n");
}

void AlphaBetaSearch::Wait() {
  while (!threads_.empty()) {
    threads_.back().join();
    threads_.pop_back();
  }
  printf("Wait done, all threads finished.\n");
}

AlphaBetaSearch::~AlphaBetaSearch() {
  Abort();
  Wait();
  // tt_.destroy(); - owned by engine class!
}

void TranspositionTable::init(uint64_t byteSize) {
  size_ = byteSize / sizeof(TTEntry);
  tt_ = (TTEntry*)malloc(byteSize);

  // size must be a power of 2
  indexBits_ = size_ - 1;
  hashBits_ = 0xFFFFFFFFFFFFFFFFull ^ indexBits_;
  reset();
}

void TranspositionTable::reset() { memset(tt_, 0, size_ * sizeof(TTEntry)); }

void TranspositionTable::destroy() { free(tt_); }

bool TranspositionTable::lookup(uint64_t hash, int searchDepth, float* score,
                                TTScoreType* scoreType, int* foundDepth,
                                Move* bestMove) {
  TTEntry entry = tt_[hash & indexBits_];
  if (entry.hashKey == hash) {
  // Ankan - TODO!
#if 0
    // convert distance to mate to absolute score for mates
    if (abs(entry.score) >= MATE_SCORE_BASE / 2) {
      if (entry.score < 0)
        entry.score = entry.score - searchDepth;
      else
        entry.score = entry.score + searchDepth;
    }
#endif
    *score = entry.score;
    *scoreType = entry.scoreType;
    *foundDepth = entry.depth;
    *bestMove = entry.bestMove;

    return true;
  } else {
    return false;
  }
}
void TranspositionTable::update(uint64_t hash, float score,
                                TTScoreType scoreType, Move bestMove, int depth,
                                int age) {
// Ankan - TODO!
#if 0
  // hack: fix mate score. Always store mate as distance to mate from the
  // current position normally we return -(MATE_SCORE_BASE + depth);
  if (abs(score) >= MATE_SCORE_BASE / 2) {
    if (score < 0)
      score = score + depth;
    else
      score = score - depth;
  }
#endif

  // TODO: better replacement strategy
  TTEntry* oldentry = &tt_[hash & indexBits_];
  if (age - oldentry->age > 32 || depth >= oldentry->depth) {
    oldentry->age = age;
    oldentry->bestMove = bestMove;
    oldentry->depth = depth;
    oldentry->hashKey = hash;
    oldentry->score = score;
    oldentry->scoreType = scoreType;
  }
}

}  // namespace lczero
