#include <iostream>
#include <tuple>
#include <cmath>
#include <random>
#include <iterator>
#include <algorithm>
#include <limits>
#include <map>

#include "nichess/nichess.hpp"
#include "agent1.hpp"

using namespace nichess;

void myMove(Game& game) {
    int x1, y1, x2, y2, x3, y3, x4, y4;
    std::cout << "Enter move's source x coordinate (-1 for MOVE_SKIP): ";
    std::cin >> x1;
    std::cout << "Enter move's source y coordinate (-1 for MOVE_SKIP): ";
    std::cin >> y1;
    std::cout << "Enter move's destination x coordinate (-1 for MOVE_SKIP): ";
    std::cin >> x2;
    std::cout << "Enter move's destination y coordinate (-1 for MOVE_SKIP): ";
    std::cin >> y2;
    std::cout << "Enter ability's source x coordinate (-1 for ABILITY_SKIP): ";
    std::cin >> x3;
    std::cout << "Enter ability's source y coordinate (-1 for ABILITY_SKIP): ";
    std::cin >> y3;
    std::cout << "Enter ability's destination x coordinate (-1 for ABILITY_SKIP): ";
    std::cin >> x4;
    std::cout << "Enter ability's destination y coordinate (-1 for ABILITY_SKIP): ";
    std::cin >> y4;

    int moveSrcIdx;
    int moveDstIdx;
    int abilitySrcIdx;
    int abilityDstIdx;
    if(x1 == MOVE_SKIP || x2 == MOVE_SKIP || y1 == MOVE_SKIP || y2 == MOVE_SKIP) {
      moveSrcIdx = MOVE_SKIP;
      moveDstIdx = MOVE_SKIP;
    } else {
      moveSrcIdx = coordinatesToBoardIndex(x1, y1);
      moveDstIdx = coordinatesToBoardIndex(x2, y2);
    }
    if(x3 == ABILITY_SKIP || x4 == ABILITY_SKIP || y3 == ABILITY_SKIP || y4 == ABILITY_SKIP) {
      abilitySrcIdx = ABILITY_SKIP;
      abilityDstIdx = ABILITY_SKIP;
    } else {
      abilitySrcIdx = coordinatesToBoardIndex(x3, y3);
      abilityDstIdx = coordinatesToBoardIndex(x4, y4);
    }

    if(game.isActionLegal(moveSrcIdx, moveDstIdx, abilitySrcIdx, abilityDstIdx)) {
      game.makeAction(moveSrcIdx, moveDstIdx, abilitySrcIdx, abilityDstIdx);
    } else {
      std::cout << "Illegal action. Try again.\n";
      game.print();
      myMove(game);
    }
}

int main() {
  bool gameOver = false;
  Game game = Game();
  agent1::Agent1 opponent = agent1::Agent1();
  game.print();
  while(!gameOver) {
    if(game.getCurrentPlayer() == PLAYER_1) {
      std::cout << "Player to move: PLAYER_1 (upper-case letters)\n";
    } else {
      std::cout << "Player to move: PLAYER_2 (lower-case letters)\n";
    }
    myMove(game);
    gameOver = game.gameOver();
    if(!gameOver) {
      PlayerAction oa = opponent.computeAction(game, 2);
      game.makeAction(oa.moveSrcIdx, oa.moveDstIdx, oa.abilitySrcIdx, oa.abilityDstIdx);
      gameOver = game.gameOver();
    }
    game.print();
  }
  std::optional<Player> winner = game.winner();
  if(winner) {
    if(winner.value() == PLAYER_1) {
      std::cout << "Player 1 won\n";
    } else {
      std::cout << "Player 2 won\n";
    }
  }
  return 0;
}
