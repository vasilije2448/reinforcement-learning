#include <iostream>
#include <tuple>
#include <cmath>
#include <random>
#include <iterator>
#include <algorithm>
#include <limits>
#include <assert.h>

#include "nichess/nichess.hpp"
#include "nichess_wrapper.hpp"
#include "agent1.hpp"

using namespace nichess;

int numNodes = 0;

float pieceTypeToValueMultiplier(PieceType pt) {
  switch(pt) {
    case P1_KING:
      return 30;
    case P1_MAGE:
      return  10; 
    case P1_PAWN:
      return 1;
    case P1_WARRIOR:
      return 5;
    case P1_WALL:
      return 0.01;
    case P1_ASSASSIN:
      return 15;
    case P2_KING:
      return 30;
    case P2_MAGE:
      return 10;
    case P2_PAWN:
      return 1;
    case P2_WARRIOR:
      return 5;
    case P2_WALL:
      return 0.01;
    case P2_ASSASSIN:
      return 15;
    case NO_PIECE:
      return 0;
    default:
      return 0;
  }
}

std::vector<std::vector<float>> createPieceTypeToIndexToSquareValue() {
  std::vector<std::vector<float>> pieceTypeToIndexToSquareValue(NUM_PIECE_TYPE);
  int mostValuableX, mostValuableY, dx, dy, currentSquareIndex;

  // p1 king
  mostValuableX = 0;
  mostValuableY = 0;
  std::vector<float> indexToP1KingSquareValue(NUM_ROWS * NUM_COLUMNS);
  for(int y = 0; y < NUM_ROWS; y++) {
    for(int x = 0; x < NUM_COLUMNS; x++) {
      currentSquareIndex = coordinatesToBoardIndex(x, y);
      dx = mostValuableX - x;
      dy = mostValuableY - y;
      if(dx == 0 && dy == 0) {
        indexToP1KingSquareValue[currentSquareIndex] = 1;
      } else {
        indexToP1KingSquareValue[currentSquareIndex] = 1 / (std::pow(dx, 2) + std::pow(dy, 2));
      }
    }
  }
  pieceTypeToIndexToSquareValue[P1_KING] = indexToP1KingSquareValue;

  // p1 mage
  mostValuableX = 4;
  mostValuableY = 4;
  std::vector<float> indexToP1MageSquareValue(NUM_ROWS * NUM_COLUMNS);
  for(int y = 0; y < NUM_ROWS; y++) {
    for(int x = 0; x < NUM_COLUMNS; x++) {
      currentSquareIndex = coordinatesToBoardIndex(x, y);
      dx = mostValuableX - x;
      dy = mostValuableY - y;
      if(dx == 0 && dy == 0) {
        indexToP1MageSquareValue[currentSquareIndex] = 1;
      } else {
        indexToP1MageSquareValue[currentSquareIndex] = 1 / (std::pow(dx, 2) + std::pow(dy, 2));
      }
    }
  }
  pieceTypeToIndexToSquareValue[P1_MAGE] = indexToP1MageSquareValue;

  // p1 pawn
  mostValuableX = 2;
  mostValuableY = 2;
  std::vector<float> indexToP1PawnSquareValue(NUM_ROWS * NUM_COLUMNS);
  for(int y = 0; y < NUM_ROWS; y++) {
    for(int x = 0; x < NUM_COLUMNS; x++) {
      currentSquareIndex = coordinatesToBoardIndex(x, y);
      dx = mostValuableX - x;
      dy = mostValuableY - y;
      if(dx == 0 && dy == 0) {
        indexToP1PawnSquareValue[currentSquareIndex] = 1;
      } else {
        indexToP1PawnSquareValue[currentSquareIndex] = 1 / (std::pow(dx, 2) + std::pow(dy, 2));
      }
    }
  }
  pieceTypeToIndexToSquareValue[P1_PAWN] = indexToP1PawnSquareValue;

  // p1 warrior
  mostValuableX = 5;
  mostValuableY = 5;
  std::vector<float> indexToP1WarriorSquareValue(NUM_ROWS * NUM_COLUMNS);
  for(int y = 0; y < NUM_ROWS; y++) {
    for(int x = 0; x < NUM_COLUMNS; x++) {
      currentSquareIndex = coordinatesToBoardIndex(x, y);
      dx = mostValuableX - x;
      dy = mostValuableY - y;
      if(dx == 0 && dy == 0) {
        indexToP1WarriorSquareValue[currentSquareIndex] = 1;
      } else {
        indexToP1WarriorSquareValue[currentSquareIndex] = 1 / (std::pow(dx, 2) + std::pow(dy, 2));
      }
    }
  }
  pieceTypeToIndexToSquareValue[P1_WARRIOR] = indexToP1WarriorSquareValue;

  // p1 walls
  // not used anywhere, but added for completeness
  std::vector<float> indexToP1WallSquareValue(NUM_ROWS * NUM_COLUMNS);
  for(int y = 0; y < NUM_ROWS; y++) {
    for(int x = 0; x < NUM_COLUMNS; x++) {
      currentSquareIndex = coordinatesToBoardIndex(x, y);
      dx = mostValuableX - x;
      dy = mostValuableY - y;
      indexToP1WallSquareValue[currentSquareIndex] = 1;
    }
  }
  pieceTypeToIndexToSquareValue[P1_WALL] = indexToP1WallSquareValue;

  // p1 assassin
  mostValuableX = 7;
  mostValuableY = 2;
  std::vector<float> indexToP1AssassinSquareValue(NUM_ROWS * NUM_COLUMNS);
  for(int y = 0; y < NUM_ROWS; y++) {
    for(int x = 0; x < NUM_COLUMNS; x++) {
      currentSquareIndex = coordinatesToBoardIndex(x, y);
      dx = mostValuableX - x;
      dy = mostValuableY - y;
      if(dx == 0 && dy == 0) {
        indexToP1AssassinSquareValue[currentSquareIndex] = 1;
      } else {
        indexToP1AssassinSquareValue[currentSquareIndex] = 1 / (std::pow(dx, 2) + std::pow(dy, 2));
      }
    }
  }
  pieceTypeToIndexToSquareValue[P1_ASSASSIN] = indexToP1AssassinSquareValue;

  // p2 king
  mostValuableX = 7;
  mostValuableY = 7;
  std::vector<float> indexToP2KingSquareValue(NUM_ROWS * NUM_COLUMNS);
  for(int y = 0; y < NUM_ROWS; y++) {
    for(int x = 0; x < NUM_COLUMNS; x++) {
      currentSquareIndex = coordinatesToBoardIndex(x, y);
      dx = mostValuableX - x;
      dy = mostValuableY - y;
      if(dx == 0 && dy == 0) {
        indexToP2KingSquareValue[currentSquareIndex] = 1;
      } else {
        indexToP2KingSquareValue[currentSquareIndex] = 1 / (std::pow(dx, 2) + std::pow(dy, 2));
      }
    }
  }
  pieceTypeToIndexToSquareValue[P2_KING] = indexToP2KingSquareValue;

  // p2 mage
  mostValuableX = 3;
  mostValuableY = 3;
  std::vector<float> indexToP2MageSquareValue(NUM_ROWS * NUM_COLUMNS);
  for(int y = 0; y < NUM_ROWS; y++) {
    for(int x = 0; x < NUM_COLUMNS; x++) {
      currentSquareIndex = coordinatesToBoardIndex(x, y);
      dx = mostValuableX - x;
      dy = mostValuableY - y;
      if(dx == 0 && dy == 0) {
        indexToP2MageSquareValue[currentSquareIndex] = 1;
      } else {
        indexToP2MageSquareValue[currentSquareIndex] = 1 / (std::pow(dx, 2) + std::pow(dy, 2));
      }
    }
  }
  pieceTypeToIndexToSquareValue[P2_MAGE] = indexToP2MageSquareValue;

  // p2 pawn
  mostValuableX = 5;
  mostValuableY = 5;
  std::vector<float> indexToP2PawnSquareValue(NUM_ROWS * NUM_COLUMNS);
  for(int y = 0; y < NUM_ROWS; y++) {
    for(int x = 0; x < NUM_COLUMNS; x++) {
      currentSquareIndex = coordinatesToBoardIndex(x, y);
      dx = mostValuableX - x;
      dy = mostValuableY - y;
      if(dx == 0 && dy == 0) {
        indexToP2PawnSquareValue[currentSquareIndex] = 1;
      } else {
        indexToP2PawnSquareValue[currentSquareIndex] = 1 / (std::pow(dx, 2) + std::pow(dy, 2));
      }
    }
  }
  pieceTypeToIndexToSquareValue[P2_PAWN] = indexToP2PawnSquareValue;

  // p2 warrior
  mostValuableX = 2;
  mostValuableY = 2;
  std::vector<float> indexToP2WarriorSquareValue(NUM_ROWS * NUM_COLUMNS);
  for(int y = 0; y < NUM_ROWS; y++) {
    for(int x = 0; x < NUM_COLUMNS; x++) {
      currentSquareIndex = coordinatesToBoardIndex(x, y);
      dx = mostValuableX - x;
      dy = mostValuableY - y;
      if(dx == 0 && dy == 0) {
        indexToP2WarriorSquareValue[currentSquareIndex] = 1;
      } else {
        indexToP2WarriorSquareValue[currentSquareIndex] = 1 / (std::pow(dx, 2) + std::pow(dy, 2));
      }
    }
  }
  pieceTypeToIndexToSquareValue[P2_WARRIOR] = indexToP2WarriorSquareValue;

  // p2 walls
  // not used anywhere, but added for completeness
  std::vector<float> indexToP2WallSquareValue(NUM_ROWS * NUM_COLUMNS);
  for(int y = 0; y < NUM_ROWS; y++) {
    for(int x = 0; x < NUM_COLUMNS; x++) {
      currentSquareIndex = coordinatesToBoardIndex(x, y);
      dx = mostValuableX - x;
      dy = mostValuableY - y;
      indexToP2WallSquareValue[currentSquareIndex] = 1;
    }
  }
  pieceTypeToIndexToSquareValue[P2_WALL] = indexToP2WallSquareValue;

  // p2 assassin
  mostValuableX = 0;
  mostValuableY = 5;
  std::vector<float> indexToP2AssassinSquareValue(NUM_ROWS * NUM_COLUMNS);
  for(int y = 0; y < NUM_ROWS; y++) {
    for(int x = 0; x < NUM_COLUMNS; x++) {
      currentSquareIndex = coordinatesToBoardIndex(x, y);
      dx = mostValuableX - x;
      dy = mostValuableY - y;
      if(dx == 0 && dy == 0) {
        indexToP2AssassinSquareValue[currentSquareIndex] = 1;
      } else {
        indexToP2AssassinSquareValue[currentSquareIndex] = 1 / (std::pow(dx, 2) + std::pow(dy, 2));
      }
    }
  }
  pieceTypeToIndexToSquareValue[P2_ASSASSIN] = indexToP2AssassinSquareValue;

  return pieceTypeToIndexToSquareValue;
}

/*
 * Returns value of the current position, relative to the player.
 */
float agent1::Agent1::positionValue(nichess_wrapper::GameWrapper& gameWrapper, Player player) {
  float retval = 0;
  std::vector<Piece*> p1Pieces = gameWrapper.game.getAllPiecesByPlayer(PLAYER_1);
  for(Piece* p : p1Pieces) {
    if(p->healthPoints <= 0) {
      retval -= pieceTypeToValueMultiplier(p->type) * 100;
      continue;
    }
    retval += pieceTypeToValueMultiplier(p->type) * p->healthPoints;
    //if(p->type == P1_KING) continue;
    //retval += indexToP1SquareValueMap[p->squareIndex] * pieceTypeToValueMultiplier(p->type) * p->healthPoints;
    retval += 0.02 * pieceTypeToIndexToSquareValue[p->type][p->squareIndex] * pieceTypeToValueMultiplier(p->type) * p->healthPoints;
  }
  std::vector<Piece*> p2Pieces = gameWrapper.game.getAllPiecesByPlayer(PLAYER_2);
  for(Piece* p : p2Pieces) {
    if(p->healthPoints <= 0) {
      retval += pieceTypeToValueMultiplier(p->type) * 100;
      continue;
    }
    retval -= pieceTypeToValueMultiplier(p->type) * p->healthPoints;
    //if(p->type == P2_KING) continue;
    //retval -= indexToP2SquareValueMap[p->squareIndex] * pieceTypeToValueMultiplier(p->type) * p->healthPoints;
    retval -= 0.02 * pieceTypeToIndexToSquareValue[p->type][p->squareIndex] * pieceTypeToValueMultiplier(p->type) * p->healthPoints;
  }
  float m = 1;
  if(player == PLAYER_2) m = -1;
  return m * retval;
}

float agent1::Agent1::alphabeta(nichess_wrapper::GameWrapper& gameWrapper, float alpha, float beta, int depth, bool maximizingPlayer, Player startingPlayer) {
  if(depth == 0 || gameWrapper.game.gameOver()) {
    return positionValue(gameWrapper, startingPlayer);
  }
  if(maximizingPlayer) {
    std::vector<PlayerAction> ala = gameWrapper.usefulLegalActionsWithoutWalls();
    float value = -std::numeric_limits<float>::max();
    for(PlayerAction pa : ala) {
      gameWrapper.game.makeAction(pa.moveSrcIdx, pa.moveDstIdx, pa.abilitySrcIdx, pa.abilityDstIdx);
      value = std::max(value, alphabeta(gameWrapper, alpha, beta, depth - 1, false, startingPlayer));
      gameWrapper.game.undoLastAction();
      numNodes++;
      if(value > beta) {
        break;
      }
      alpha = std::max(alpha, value);
    }
    return value;
  } else {
    std::vector<PlayerAction> ala = gameWrapper.usefulLegalActionsWithoutWalls();
    float value = std::numeric_limits<float>::max();
    for(PlayerAction pa : ala) {
      gameWrapper.game.makeAction(pa.moveSrcIdx, pa.moveDstIdx, pa.abilitySrcIdx, pa.abilityDstIdx);
      value = std::min(value, alphabeta(gameWrapper, alpha, beta, depth - 1, true, startingPlayer));
      gameWrapper.game.undoLastAction();
      numNodes++;
      if(value < alpha) {
        break;
      }
      beta = std::min(beta, value);
    }
    return value;

  }
}

agent1::Agent1::Agent1() {
  pieceTypeToIndexToSquareValue = createPieceTypeToIndexToSquareValue();
}

PlayerAction agent1::Agent1::computeAction(nichess_wrapper::GameWrapper& gameWrapper, int searchDepth) {
  assert(searchDepth > 0);
  numNodes = 0;
  std::vector<PlayerAction> ala = gameWrapper.usefulLegalActionsWithoutWalls();
  float bestValue = -std::numeric_limits<float>::max();
  PlayerAction bestAction = ala[0];
  float value;
  Player startingPlayer = gameWrapper.game.getCurrentPlayer();
  for(PlayerAction pa : ala) {
    gameWrapper.game.makeAction(pa.moveSrcIdx, pa.moveDstIdx, pa.abilitySrcIdx, pa.abilityDstIdx);
    float alpha = -std::numeric_limits<float>::max();
    float beta = std::numeric_limits<float>::max();
    value = alphabeta(gameWrapper, alpha, beta, searchDepth - 1, false, startingPlayer);
    if(value > bestValue) {
      bestValue = value;
      bestAction = pa;
    }
    gameWrapper.game.undoLastAction();
    numNodes++;
  }
  std::cout << "Number of nodes explored: " << numNodes << "\n";
  return bestAction;
}
