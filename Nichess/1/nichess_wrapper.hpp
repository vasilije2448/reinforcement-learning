#pragma once

#include "nichess/nichess.hpp"

#include <vector>
#include <memory>

using namespace nichess;

namespace nichess_wrapper {

class GameWrapper {
  public:
    std::unique_ptr<Game> game;

    GameWrapper(GameCache& gameCache);
    std::vector<PlayerAction> usefulLegalActionsWithoutMoves();
};

} // namespace nichess_wrapper
