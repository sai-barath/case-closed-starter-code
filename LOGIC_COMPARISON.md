# Game Logic Comparison: Python vs Go

## ✅ VERIFIED: Both implementations are IDENTICAL

### Critical Move Order (The Bug)

**Python (case_closed_game.py:218-219)**
```python
agent_one_alive = self.agent1.move(dir1, other_agent=self.agent2, use_boost=boost1)
agent_two_alive = self.agent2.move(dir2, other_agent=self.agent1, use_boost=boost2)
```

**Go (game_logic.go:373-374)**
```go
agentOneAlive := g.Agent1.Move(dir1, g.Agent2, boost1)
agentTwoAlive := g.Agent2.Move(dir2, g.Agent1, boost2)
```

**Result**: Agent 1 ALWAYS moves first in both implementations. This creates the move-order bias.

---

## Key Components Verified

### 1. Board Configuration
- **Size**: 18x20 (Height x Width) ✅
- **Torus wraparound**: Both use modulo arithmetic ✅
- **Cell states**: EMPTY=0, AGENT=1 ✅

### 2. Agent Initialization
- **Start positions**: 
  - Agent1: (1, 2) facing RIGHT ✅
  - Agent2: (17, 15) facing LEFT ✅
- **Trail length**: 2 cells initially ✅
- **Boosts**: 3 per agent ✅

### 3. Move Logic (Agent.move/Move)

**Collision Detection Order** (IDENTICAL):
1. Check if not alive → return false
2. Check boost availability
3. For each move (1 or 2 if boosted):
   - Check if direction is opposite (invalid) → skip
   - Calculate new head position
   - Apply torus wraparound
   - Get cell state
   - Update direction
   - **If cell has AGENT**:
     - Check if own trail → die (return false)
     - Check if other agent's trail:
       - If other's HEAD → head-on collision → BOTH die
       - Else → hit trail → self dies
   - Add new head to trail
   - Increment length
   - Set board cell to AGENT
4. Return true if survived

### 4. Game.step/Step Logic (IDENTICAL)

1. Check if turns >= 200 → compare lengths
2. **Agent1 moves first** ← THE BUG
3. **Agent2 moves second** ← THE BUG  
4. Check results:
   - Both dead → DRAW
   - Agent1 dead → AGENT2_WIN
   - Agent2 dead → AGENT1_WIN
   - Both alive → increment turns, return nil/None

---

## Critical Implementation Details

### Python uses `deque` and `in` operator for trail checking
```python
self.trail = deque([start_pos, second])
if new_head in self.trail:  # O(n) linear search
```

### Go uses `[]Position` + `map[Position]bool` for O(1) trail checking
```go
Trail:    []Position{startPos, second}
TrailSet: map[Position]bool{startPos: true, second: true}
if a.ContainsPosition(newHead) {  // O(1) map lookup
```

**This is a valid optimization - both are functionally identical.**

---

## The Move-Order Bug Explained

Since Agent1 always moves first:

**Scenario**: Agent1 attacks position P, Agent2 currently at P wants to escape

**Turn execution**:
1. Agent1 moves to P
2. Agent1's new head is placed at P
3. Agent2 tries to move
4. But P now contains Agent1's head → head-on collision OR trail collision
5. Result depends on exact timing

**If positions swapped** (Agent2 at attacking position, Agent1 at escape position):
1. Agent1 escapes first (gets priority)
2. Agent2 tries to attack the now-empty position
3. Agent1 survives

**Conclusion**: This is a fundamental game design issue, not an implementation bug. Both Python and Go implement it identically.
