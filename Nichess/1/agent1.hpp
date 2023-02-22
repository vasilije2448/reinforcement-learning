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
    float alphabeta(nichess_wrapper::GameWrapper& gameWrapper, float alpha, float beta, int depth, bool maximizingPlayer, Player startingPlayer);
    float positionValue(nichess_wrapper::GameWrapper& gameWrapper, Player player);
};

} // namespace agent1
