#include <iostream>
#include <tuple>
#include <cmath>
#include <random>
#include <iterator>
#include <algorithm>
#include <limits>
#include <map>
#include <chrono>

#include "nichess/nichess.hpp"
#include "nichess_wrapper.hpp"
#include "agent1.hpp"

using namespace nichess;

void myMove(nichess_wrapper::GameWrapper& gameWrapper) {
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

    if(gameWrapper.game.isActionLegal(moveSrcIdx, moveDstIdx, abilitySrcIdx, abilityDstIdx)) {
      gameWrapper.game.makeAction(moveSrcIdx, moveDstIdx, abilitySrcIdx, abilityDstIdx);
    } else {
      std::cout << "Illegal action. Try again.\n";
      gameWrapper.game.print();
      myMove(gameWrapper);
    }
}

int main() {
  bool gameOver = false;
  nichess_wrapper::GameWrapper gameWrapper = nichess_wrapper::GameWrapper();
  agent1::Agent1 opponent = agent1::Agent1();
  gameWrapper.game.print();
  while(!gameOver) {
    if(gameWrapper.game.getCurrentPlayer() == PLAYER_1) {
      std::cout << "Player to move: PLAYER_1 (upper-case letters)\n";
    } else {
      std::cout << "Player to move: PLAYER_2 (lower-case letters)\n";
    }
    //myMove(gameWrapper);
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    PlayerAction oa = opponent.computeAction(gameWrapper, 5000);
    oa.print();
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Calculating time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;
    gameWrapper.game.makeAction(oa.moveSrcIdx, oa.moveDstIdx, oa.abilitySrcIdx, oa.abilityDstIdx);
    gameWrapper.game.print();

    gameOver = gameWrapper.game.gameOver();
    if(!gameOver) {
      std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
      PlayerAction oa = opponent.computeAction(gameWrapper, 5000);
      oa.print();
      std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
      std::cout << "Calculating time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;
      gameWrapper.game.makeAction(oa.moveSrcIdx, oa.moveDstIdx, oa.abilitySrcIdx, oa.abilityDstIdx);
      gameOver = gameWrapper.game.gameOver();
    }
    gameWrapper.game.print();
  }
  std::optional<Player> winner = gameWrapper.game.winner();
  if(winner) {
    if(winner.value() == PLAYER_1) {
      std::cout << "Player 1 won\n";
    } else {
      std::cout << "Player 2 won\n";
    }
  }
  /*
  agent1::Agent1 ag = agent1::Agent1();
  std::string position;
  std::cout << "Enter encoded position: ";
  std::cin >> position; // get user input from the keyboard
  Player p = PLAYER_2;
  float val = ag.positionValueFromString(position, p);
  std::cout << val;
  */
  return 0;
}
