#pragma once

#include "nichess/nichess.hpp"

#include <vector>

using namespace nichess;

namespace nichess_wrapper {

class GameWrapper {
  public:
    Game game;

    GameWrapper();
    std::vector<PlayerAction> usefulLegalActionsWithoutWalls();
    std::vector<PlayerAction> usefulLegalActionsWithoutMovesAndWalls();
};

} // namespace nichess_wrapper
