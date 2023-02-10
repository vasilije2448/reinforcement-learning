#pragma once

#include "nichess/nichess.hpp"

#include <map>

using namespace nichess;

namespace agent1 {

class Agent1 {
  public:
    std::map<int, float> indexToP1SquareValueMap;
    std::map<int, float> indexToP2SquareValueMap;

    Agent1();
    PlayerAction computeAction(Game& game, int searchDepth);
    float minimax(Game& game, int depth, bool maximizingPlayer, Player startingPlayer);
    float positionValue(Game& game, Player player);
};

} // namespace agent1
