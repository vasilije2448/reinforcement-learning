#include <iostream>
#include <tuple>
#include <cmath>
#include <random>
#include <iterator>
#include <algorithm>
#include <limits>
#include <map>
#include <assert.h>

#include "nichess/nichess.hpp"
#include "agent1.hpp"

using namespace nichess;

float pieceTypeToValueMultiplier(PieceType pt) {
  switch(pt) {
    case P1_KING:
      return 1000;
    case P1_MAGE:
      return  10; 
    case P1_PAWN:
      return 1;
    case P1_WARRIOR:
      return 5;
    case P1_WALL:
      return 0.01;
    case P1_ASSASSIN:
      return 20;
    case P2_KING:
      return 1000;
    case P2_MAGE:
      return 10;
    case P2_PAWN:
      return 1;
    case P2_WARRIOR:
      return 5;
    case P2_WALL:
      return 0.01;
    case P2_ASSASSIN:
      return 20;
    case NO_PIECE:
      return 0;
    default:
      return 0;
  }
}

/*
 * Square value represents how important is the square for PLAYER_1.
 * (Encourages moving pieces forward)
 */
std::map<int, float> createIndexToP1SquareValueMap() {
  std::map<int, float> retval;
  // want pieces to go close to p2's king
  int mostValuableX = NUM_COLUMNS - 1;
  int mostValuableY = NUM_ROWS - 1;
  int currentSquareIndex, dx, dy;
  for(int y = 0; y < NUM_ROWS; y++) {
    for(int x = 0; x < NUM_COLUMNS; x++) {
      currentSquareIndex = coordinatesToBoardIndex(x, y);
      dx = mostValuableX - x;
      dy = mostValuableY - y;
      if(dx == 0 && dy == 0) {
        retval[currentSquareIndex] = 1;
      } else {
        retval[currentSquareIndex] = 1 / (std::pow(dx, 2) + std::pow(dy, 2));
      }
    }
  }
  return retval;
}

/*
 * Square value represents how important is the square for PLAYER_2.
 * (Encourages moving pieces forward)
 */
std::map<int, float> createIndexToP2SquareValueMap() {
  std::map<int, float> retval;
  // want pieces to go close to p1's king
  int mostValuableX = 0;
  int mostValuableY = 0;
  int currentSquareIndex, dx, dy;
  for(int y = 0; y < NUM_ROWS; y++) {
    for(int x = 0; x < NUM_COLUMNS; x++) {
      currentSquareIndex = coordinatesToBoardIndex(x, y);
      dx = mostValuableX - x;
      dy = mostValuableY - y;
      if(dx == 0 && dy == 0) {
        retval[currentSquareIndex] = 1;
      } else {
        retval[currentSquareIndex] = 1 / (std::pow(dx, 2) + std::pow(dy, 2));
      }
    }
  }
  return retval;
}

/*
 * Returns value of the current position, relative to the player.
 */
float agent1::Agent1::positionValue(Game& game, Player player) {
  float retval = 0;
  std::vector<Piece*> p1Pieces = game.getAllPiecesByPlayer(PLAYER_1);
  for(Piece* p : p1Pieces) {
    if(p->healthPoints <= 0) {
      retval -= pieceTypeToValueMultiplier(p->type) * 100;
      continue;
    }
    retval += pieceTypeToValueMultiplier(p->type) * p->healthPoints;
    if(p->type == P1_KING) continue;
    retval += indexToP1SquareValueMap[p->squareIndex] * 1000 * pieceTypeToValueMultiplier(p->type);
  }
  std::vector<Piece*> p2Pieces = game.getAllPiecesByPlayer(PLAYER_2);
  for(Piece* p : p2Pieces) {
    if(p->healthPoints <= 0) {
      retval += pieceTypeToValueMultiplier(p->type) * 100;
      continue;
    }
    retval -= pieceTypeToValueMultiplier(p->type) * p->healthPoints;
    if(p->type == P2_KING) continue;
    retval -= indexToP2SquareValueMap[p->squareIndex] * 1000 * pieceTypeToValueMultiplier(p->type);
  }
  float m = 1;
  if(player == PLAYER_2) m = -1;
  return m * retval;
}

float agent1::Agent1::minimax(Game& game, int depth, bool maximizingPlayer, Player startingPlayer) {
  if(depth == 0 || game.gameOver()) {
    return positionValue(game, startingPlayer);
  }
  if(maximizingPlayer) {
    std::vector<PlayerAction> ala = game.usefulLegalActions();
    float value = -std::numeric_limits<float>::max();
    for(PlayerAction pa : ala) {
      game.makeAction(pa.moveSrcIdx, pa.moveDstIdx, pa.abilitySrcIdx, pa.abilityDstIdx);
      value = std::max(value, minimax(game, depth - 1, false, startingPlayer));
      game.undoLastAction();
    }
    return value;
  } else {
    std::vector<PlayerAction> ala = game.usefulLegalActions();
    float value = std::numeric_limits<float>::max();
    for(PlayerAction pa : ala) {
      game.makeAction(pa.moveSrcIdx, pa.moveDstIdx, pa.abilitySrcIdx, pa.abilityDstIdx);
      value = std::min(value, minimax(game, depth - 1, true, startingPlayer));
      game.undoLastAction();
    }
    return value;

  }
}

agent1::Agent1::Agent1() {
  indexToP1SquareValueMap = createIndexToP1SquareValueMap();
  indexToP2SquareValueMap = createIndexToP2SquareValueMap();
}

PlayerAction agent1::Agent1::computeAction(Game& game, int searchDepth) {
  assert(searchDepth > 0);
  std::vector<PlayerAction> ala = game.usefulLegalActions();
  float bestValue = -std::numeric_limits<float>::max();
  PlayerAction bestAction = ala[0];
  float value;
  Player startingPlayer = game.getCurrentPlayer();
  for(PlayerAction pa : ala) {
    game.makeAction(pa.moveSrcIdx, pa.moveDstIdx, pa.abilitySrcIdx, pa.abilityDstIdx);
    value = minimax(game, searchDepth - 1, false, startingPlayer);
    if(value > bestValue) {
      bestValue = value;
      bestAction = pa;
    }
    game.undoLastAction();
    /*
    game.makeAction(pa.moveSrcIdx, pa.moveDstIdx, pa.abilitySrcIdx, pa.abilityDstIdx);
    value = positionValue(game, ~startingPlayer);
    if(value > bestValue) {
      bestValue = value;
      bestAction = pa;
    }
    game.undoLastAction();
    */
  }
  std::cout << "\n Enemy action:\n";
  bestAction.print();
  std::cout << "Value (from enemy's POV): " << bestValue << "\n\n";
  return bestAction;
}
