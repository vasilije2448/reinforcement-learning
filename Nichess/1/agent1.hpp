#pragma once

#include "nichess/nichess.hpp"
#include "nichess_wrapper.hpp"

#include <vector>
#include <chrono>

using namespace nichess;

namespace agent1 {

class Agent1 {
  public:
    std::vector<std::vector<float>> pieceTypeToIndexToSquareValue;
    int numNodesSearched = 0;
    const int numNodesBeforeTimeCheck = 10000; // used for iterative deepening
    int maxThinkingTime = 3000; // in milliseconds
	  std::chrono::time_point<std::chrono::system_clock> startTime;
		bool abortSearch = false; // set to true when time runs out

    Agent1();
    PlayerAction computeAction(nichess_wrapper::GameWrapper& gameWrapper, int maxThinkingTime);
		PlayerAction runAlphaBetaSearch(nichess_wrapper::GameWrapper& gameWrapper, int searchDepth);
    float alphabeta(nichess_wrapper::GameWrapper& gameWrapper, float alpha, float beta, int depth, bool maximizingPlayer, Player startingPlayer);
    float positionValue(nichess_wrapper::GameWrapper& gameWrapper, Player player);
    float quiescenceSearch(nichess_wrapper::GameWrapper& gameWrapper, bool maximizingPlayer, Player startingPlayer);
    float positionValueFromString(std::string position, Player player);
};

} // namespace agent1
