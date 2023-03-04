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
      return 15;
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
  double t;

  // p1 king
  mostValuableX = 0;
  mostValuableY = 0;
  std::vector<float> indexToP1KingSquareValue(NUM_ROWS * NUM_COLUMNS);
  for(int y = 0; y < NUM_ROWS; y++) {
    for(int x = 0; x < NUM_COLUMNS; x++) {
      currentSquareIndex = coordinatesToBoardIndex(x, y);
      dx = std::abs(mostValuableX - x);
      dy = std::abs(mostValuableY - y);
      if(dx == 0 && dy == 0) {
        indexToP1KingSquareValue[currentSquareIndex] = 1;
      } else {
        t = std::max(dx, dy);
        t = 1 - 0.1 * t;
        t = std::max(0.5, t);
        indexToP1KingSquareValue[currentSquareIndex] = t;
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
      dx = std::abs(mostValuableX - x);
      dy = std::abs(mostValuableY - y);
      if(dx == 0 && dy == 0) {
        indexToP1MageSquareValue[currentSquareIndex] = 1;
      } else {
        t = std::max(dx, dy);
        t = 1 - 0.1 * t;
        t = std::max(0.5, t);
        indexToP1MageSquareValue[currentSquareIndex] = t;
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
      dx = std::abs(mostValuableX - x);
      dy = std::abs(mostValuableY - y);
      if(dx == 0 && dy == 0) {
        indexToP1PawnSquareValue[currentSquareIndex] = 1;
      } else {
        t = std::max(dx, dy);
        t = 1 - 0.1 * t;
        t = std::max(0.5, t);
        indexToP1PawnSquareValue[currentSquareIndex] = t;
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
      dx = std::abs(mostValuableX - x);
      dy = std::abs(mostValuableY - y);
      if(dx == 0 && dy == 0) {
        indexToP1WarriorSquareValue[currentSquareIndex] = 1;
      } else {
        t = std::max(dx, dy);
        t = 1 - 0.1 * t;
        t = std::max(0.5, t);
        indexToP1WarriorSquareValue[currentSquareIndex] = t;
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
      dx = std::abs(mostValuableX - x);
      dy = std::abs(mostValuableY - y);
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
      dx = std::abs(mostValuableX - x);
      dy = std::abs(mostValuableY - y);
      if(dx == 0 && dy == 0) {
        indexToP1AssassinSquareValue[currentSquareIndex] = 1;
      } else {
        t = std::max(dx, dy);
        t = 1 - 0.1 * t;
        t = std::max(0.5, t);
        indexToP1AssassinSquareValue[currentSquareIndex] = t;
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
      dx = std::abs(mostValuableX - x);
      dy = std::abs(mostValuableY - y);
      if(dx == 0 && dy == 0) {
        indexToP2KingSquareValue[currentSquareIndex] = 1;
      } else {
        t = std::max(dx, dy);
        t = 1 - 0.1 * t;
        t = std::max(0.5, t);
        indexToP2KingSquareValue[currentSquareIndex] = t;
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
      dx = std::abs(mostValuableX - x);
      dy = std::abs(mostValuableY - y);
      if(dx == 0 && dy == 0) {
        indexToP2MageSquareValue[currentSquareIndex] = 1;
      } else {
        t = std::max(dx, dy);
        t = 1 - 0.1 * t;
        t = std::max(0.5, t);
        indexToP2MageSquareValue[currentSquareIndex] = t;
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
      dx = std::abs(mostValuableX - x);
      dy = std::abs(mostValuableY - y);
      if(dx == 0 && dy == 0) {
        indexToP2PawnSquareValue[currentSquareIndex] = 1;
      } else {
        t = std::max(dx, dy);
        t = 1 - 0.1 * t;
        t = std::max(0.5, t);
        indexToP2PawnSquareValue[currentSquareIndex] = t;
      }
    }
  }
  pieceTypeToIndexToSquareValue[P2_PAWN] = indexToP2PawnSquareValue;

  // p2 warrior
  mostValuableX = 5;
  mostValuableY = 5;
  std::vector<float> indexToP2WarriorSquareValue(NUM_ROWS * NUM_COLUMNS);
  for(int y = 0; y < NUM_ROWS; y++) {
    for(int x = 0; x < NUM_COLUMNS; x++) {
      currentSquareIndex = coordinatesToBoardIndex(x, y);
      dx = std::abs(mostValuableX - x);
      dy = std::abs(mostValuableY - y);
      if(dx == 0 && dy == 0) {
        indexToP2WarriorSquareValue[currentSquareIndex] = 1;
      } else {
        t = std::max(dx, dy);
        t = 1 - 0.1 * t;
        t = std::max(0.5, t);
        indexToP2WarriorSquareValue[currentSquareIndex] = t;
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
      dx = std::abs(mostValuableX - x);
      dy = std::abs(mostValuableY - y);
      indexToP2WallSquareValue[currentSquareIndex] = 1;
    }
  }
  pieceTypeToIndexToSquareValue[P2_WALL] = indexToP2WallSquareValue;

  // p2 assassin
  mostValuableX = 0;
  mostValuableY = 0;
  std::vector<float> indexToP2AssassinSquareValue(NUM_ROWS * NUM_COLUMNS);
  for(int y = 0; y < NUM_ROWS; y++) {
    for(int x = 0; x < NUM_COLUMNS; x++) {
      currentSquareIndex = coordinatesToBoardIndex(x, y);
      dx = std::abs(mostValuableX - x);
      dy = std::abs(mostValuableY - y);
      if(dx == 0 && dy == 0) {
        indexToP2AssassinSquareValue[currentSquareIndex] = 1;
      } else {
        t = std::max(dx, dy);
        t = 1 - 0.1 * t;
        t = std::max(0.5, t);
        indexToP2AssassinSquareValue[currentSquareIndex] = t;
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
    retval += pieceTypeToIndexToSquareValue[p->type][p->squareIndex] * pieceTypeToValueMultiplier(p->type) * p->healthPoints;
    //if(p->type == P1_KING) continue;
    //retval += indexToP1SquareValueMap[p->squareIndex] * pieceTypeToValueMultiplier(p->type) * p->healthPoints;
    //retval += 0.02 * pieceTypeToIndexToSquareValue[p->type][p->squareIndex] * pieceTypeToValueMultiplier(p->type) * p->healthPoints;
  }
  std::vector<Piece*> p2Pieces = gameWrapper.game.getAllPiecesByPlayer(PLAYER_2);
  for(Piece* p : p2Pieces) {
    if(p->healthPoints <= 0) {
      retval += pieceTypeToValueMultiplier(p->type) * 100;
      continue;
    }
    retval -= pieceTypeToIndexToSquareValue[p->type][p->squareIndex] * pieceTypeToValueMultiplier(p->type) * p->healthPoints;
    //if(p->type == P2_KING) continue;
    //retval -= indexToP2SquareValueMap[p->squareIndex] * pieceTypeToValueMultiplier(p->type) * p->healthPoints;
    //retval -= 0.02 * pieceTypeToIndexToSquareValue[p->type][p->squareIndex] * pieceTypeToValueMultiplier(p->type) * p->healthPoints;
  }
  float m = 1;
  if(player == PLAYER_2) m = -1;
  return m * retval;
}

float agent1::Agent1::positionValueFromString(std::string encodedPosition, Player player) {
  nichess_wrapper::GameWrapper gw = nichess_wrapper::GameWrapper();
  gw.game.boardFromString(encodedPosition);
  gw.game.print();
  return positionValue(gw, player);
}

float agent1::Agent1::quiescenceSearch(nichess_wrapper::GameWrapper& gameWrapper, bool maximizingPlayer, Player startingPlayer) {
  if(this->numNodesSearched % this->numNodesBeforeTimeCheck == 0 &&
      std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - this->startTime).count() >= this->maxThinkingTime) {
    this->abortSearch = true;
    return 0;
  }

  std::vector<PlayerAction> actions = gameWrapper.usefulLegalActionsWithoutMovesAndWalls();
  if(actions.size() == 0) return positionValue(gameWrapper, startingPlayer);
  
  float value;
  if(maximizingPlayer) {
      value = -std::numeric_limits<float>::max();
      for(PlayerAction pa : actions) {
        gameWrapper.game.makeAction(pa.moveSrcIdx, pa.moveDstIdx, pa.abilitySrcIdx, pa.abilityDstIdx);
        value = std::max(value, quiescenceSearch(gameWrapper, false, startingPlayer));
        gameWrapper.game.undoLastAction();
        numNodesSearched++;
        if(this->abortSearch) return 0;
      }
    } else {
      value = std::numeric_limits<float>::max();
      for(PlayerAction pa : actions) {
        gameWrapper.game.makeAction(pa.moveSrcIdx, pa.moveDstIdx, pa.abilitySrcIdx, pa.abilityDstIdx);
        value = std::min(value, quiescenceSearch(gameWrapper, true, startingPlayer));
        gameWrapper.game.undoLastAction();
        numNodesSearched++;
        if(this->abortSearch) return 0;
      }
    }
  return value;
}

float agent1::Agent1::alphabeta(nichess_wrapper::GameWrapper& gameWrapper, float alpha, float beta, int depth, bool maximizingPlayer, Player startingPlayer) {
  if(this->numNodesSearched % this->numNodesBeforeTimeCheck == 0 &&
      std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - this->startTime).count() >= this->maxThinkingTime) {
    this->abortSearch = true;
    return 0;
  }
			
  if(depth == 0 || gameWrapper.game.gameOver()) {
    //return positionValue(gameWrapper, startingPlayer);
    return quiescenceSearch(gameWrapper, !maximizingPlayer, startingPlayer);
  }
  if(maximizingPlayer) {
    std::vector<PlayerAction> ala = gameWrapper.usefulLegalActionsWithoutWalls();
    float value = -std::numeric_limits<float>::max();
    for(PlayerAction pa : ala) {
      gameWrapper.game.makeAction(pa.moveSrcIdx, pa.moveDstIdx, pa.abilitySrcIdx, pa.abilityDstIdx);
      value = std::max(value, alphabeta(gameWrapper, alpha, beta, depth - 1, false, startingPlayer));
      gameWrapper.game.undoLastAction();
      numNodesSearched++;
      if(this->abortSearch) return 0;
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
      numNodesSearched++;
      if(this->abortSearch) return 0;
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

class ActionValue {
  public:
    PlayerAction action;
    float value;

    ActionValue() { }
    ActionValue(PlayerAction action, float value): action(action), value(value) { }
};

bool compareActionValue(ActionValue av1, ActionValue av2) {
  return av1.value > av2.value;
}

PlayerAction agent1::Agent1::runAlphaBetaSearch(nichess_wrapper::GameWrapper& gameWrapper, int searchDepth) {
  std::vector<PlayerAction> ala = gameWrapper.usefulLegalActionsWithoutWalls();
  float bestValue = -std::numeric_limits<float>::max();
  PlayerAction bestAction = ala[0];
  float value;
  Player startingPlayer = gameWrapper.game.getCurrentPlayer();
  std::vector<ActionValue> bestActionValues;
  for(PlayerAction pa : ala) {
    gameWrapper.game.makeAction(pa.moveSrcIdx, pa.moveDstIdx, pa.abilitySrcIdx, pa.abilityDstIdx);
    float alpha = -std::numeric_limits<float>::max();
    float beta = std::numeric_limits<float>::max();
    value = alphabeta(gameWrapper, alpha, beta, searchDepth - 1, false, startingPlayer);
    gameWrapper.game.undoLastAction();
    numNodesSearched++;
    if(this->abortSearch) return bestAction;
    if(value > bestValue) {
      bestValue = value;
      bestAction = pa;
    }

    if(bestActionValues.size() < 5) {
      ActionValue av = ActionValue(pa, value);
      bestActionValues.push_back(av);
      std::sort(bestActionValues.begin(), bestActionValues.end(), compareActionValue);
    } else {
      ActionValue currentAv;
      for(int i = 0; i < bestActionValues.size(); i++) {
        currentAv = bestActionValues[i];
        if(value > currentAv.value) {
          ActionValue newAv = ActionValue(pa, value);
          bestActionValues.insert(bestActionValues.begin() + i, newAv);
          bestActionValues.pop_back();
          break;
        }
      }
    }
  }
  std::cout << "Number of nodes explored: " << numNodesSearched << "\n";
  /*
  std::cout << "Best actions and their values:\n";
  for(ActionValue av: bestActionValues) {
    std::cout << "Action:\n";
    av.action.print();
    std::cout << "Value:\n";
    std::cout << av.value << "\n";
  }
  */
  return bestAction;
}

PlayerAction agent1::Agent1::computeAction(nichess_wrapper::GameWrapper& gameWrapper, int maxThinkingTime) {
  assert(maxThinkingTime > 300);
  this->startTime = std::chrono::system_clock::now();
  this->numNodesSearched = 0;
  this->abortSearch = false;
  this->maxThinkingTime = maxThinkingTime;
  PlayerAction allTimeBestAction, currentBestAction;
  allTimeBestAction = PlayerAction(MOVE_SKIP, MOVE_SKIP, ABILITY_SKIP, ABILITY_SKIP);
  int i = 1;
  while(true) {
    std::cout << "Searching with max depth " << i << "\n";
    currentBestAction = runAlphaBetaSearch(gameWrapper, i);
    if(!this->abortSearch) { // only save best action if search was completed
      allTimeBestAction = currentBestAction;
    } else {
      std::cout << "Total number of nodes explored: " << this->numNodesSearched << "\n";
      std::cout << "Search with depth " << i << " not completed. Using result from depth " << i-1 << ".\n";
      break;
    }
    i++;
  }
  return allTimeBestAction;
}
