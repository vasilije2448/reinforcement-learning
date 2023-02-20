#pragma once

#include "nichess/nichess.hpp"

#include <vector>

using namespace nichess;

namespace agent1 {

class Agent1 {
  public:
    std::vector<std::vector<float>> pieceTypeToIndexToSquareValue;

    Agent1();
    PlayerAction computeAction(Game& game, int searchDepth);
    float minimax(Game& game, int depth, bool maximizingPlayer, Player startingPlayer);
    float positionValue(Game& game, Player player);
};

} // namespace agent1
