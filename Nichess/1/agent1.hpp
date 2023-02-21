#pragma once

#include "nichess/nichess.hpp"
#include "nichess_wrapper.hpp"

#include <vector>

using namespace nichess;

namespace agent1 {

class Agent1 {
  public:
    std::vector<std::vector<float>> pieceTypeToIndexToSquareValue;

    Agent1();
    PlayerAction computeAction(nichess_wrapper::GameWrapper& gameWrapper, int searchDepth);
    float minimax(nichess_wrapper::GameWrapper& gameWrapper, int depth, bool maximizingPlayer, Player startingPlayer);
    float positionValue(nichess_wrapper::GameWrapper& gameWrapper, Player player);
};

} // namespace agent1
