import json
import random
from typing import Dict, List, Tuple, Optional

class BattlesnakeLogic:
    def __init__(self):
        # Base weights for different move factors
        self.base_weights = {
            'avoid_walls': 100.0,
            'avoid_body': 100.0,
            'avoid_enemies': 80.0,
            'food_attraction': 30.0,
            'center_attraction': 5.0,
            'space_control': 15.0,
            'head_to_head': -50.0,
        }
        
        # Strategy thresholds
        self.strategy_thresholds = {
            'low_health': 30,
            'high_health': 70,
            'short_length': 4,
            'long_length': 8,
        }
        
        # Current active weights
        self.weights = self.base_weights.copy()
        
        # Move directions
        self.directions = ['up', 'down', 'left', 'right']
        self.direction_vectors = {
            'up': (0, 1),
            'down': (0, -1),
            'left': (-1, 0),
            'right': (1, 0)
        }

    def get_move(self, game_state: Dict) -> str:
        """Main function to determine the next move"""
        my_snake = game_state['you']
        board = game_state['board']
        head = my_snake['head']
        
        # Debug logging
        print(f"\n=== TURN {game_state.get('turn', 0)} ===")
        print(f"Head position: ({head['x']}, {head['y']})")
        print(f"Board size: {board['width']}x{board['height']}")
        
        # FIRST: Get all safe moves (no walls, no immediate death)
        safe_moves = self.get_safe_moves(head, my_snake, board)
        
        print(f"Safe moves available: {safe_moves}")
        
        # If no safe moves, we're dead anyway - pick randomly
        if not safe_moves:
            print("❌ NO SAFE MOVES - choosing randomly")
            return random.choice(self.directions)
        
        # If only one safe move, take it
        if len(safe_moves) == 1:
            print(f"✓ Only one safe move: {safe_moves[0]}")
            return safe_moves[0]
        
        # Dynamically adjust strategy based on snake state
        self.adjust_strategy(my_snake, board)
        
        # Calculate scores ONLY for safe moves
        move_scores = {}
        for direction in safe_moves:
            new_pos = self.get_new_position(head, direction)
            score = 0.0
            
            # CRITICAL: Extra penalty for risky equal-length scenarios
            equal_length_penalty = self.calculate_equal_length_danger(new_pos, my_snake, board)
            score += equal_length_penalty
            
            # Add wall proximity penalty
            wall_proximity_score = self.calculate_wall_proximity_score(new_pos, board)
            score += wall_proximity_score
            
            # Check for traps and dead ends
            trap_penalty = self.calculate_trap_score(new_pos, my_snake, board)
            score += trap_penalty
            
            # Skip heavy calculations if trapped badly
            if trap_penalty > -500:
                # Calculate escape routes
                escape_score = self.calculate_escape_route_score(new_pos, my_snake, board)
                score += escape_score
                
                # Look ahead for future traps (important for long snakes)
                future_space_score = self.calculate_future_space_score(new_pos, my_snake, board)
                score += future_space_score
                
                # Smart tail chasing
                tail_score = self.calculate_tail_chase_score(new_pos, my_snake, board)
                score += tail_score
                
                # Strategic factors
                score += self.calculate_food_score(new_pos, board['food'])
                score += self.calculate_space_score(new_pos, my_snake, board)
                score += self.calculate_center_score(new_pos, board)
                score += self.calculate_enemy_avoidance_score(new_pos, board['snakes'], my_snake)
                score += self.calculate_head_to_head_score(new_pos, board['snakes'], my_snake)
            
            move_scores[direction] = score
        
        # Sort by score
        sorted_moves = sorted(move_scores.items(), key=lambda x: x[1], reverse=True)
        best_move = sorted_moves[0][0]
        
        strategy = self.get_current_strategy(my_snake)
        print(f"Strategy: {strategy}")
        print(f"Move scores: {dict(sorted_moves)}")
        print(f"✓ Choosing: {best_move}\n")
        
        return best_move
    
    def get_safe_moves(self, head: Dict, my_snake: Dict, board: Dict) -> List[str]:
        """Get all moves that don't immediately kill us (walls, snake bodies)"""
        safe = []
        
        for direction in self.directions:
            new_pos = self.get_new_position(head, direction)
            x, y = new_pos
            
            # Check walls - THIS IS CRITICAL
            if x < 0 or x >= board['width'] or y < 0 or y >= board['height']:
                print(f"  {direction} -> OUT OF BOUNDS ({x}, {y})")
                continue
            
            # Check own body (excluding tail)
            hit_self = False
            for i, segment in enumerate(my_snake['body'][:-1]):
                if new_pos == (segment['x'], segment['y']):
                    print(f"  {direction} -> HIT OWN BODY at ({x}, {y})")
                    hit_self = True
                    break
            
            if hit_self:
                continue
            
            # Check enemy bodies
            hit_enemy = False
            for snake in board['snakes']:
                if snake['id'] == my_snake['id']:
                    continue
                for i, segment in enumerate(snake['body'][:-1]):
                    if new_pos == (segment['x'], segment['y']):
                        print(f"  {direction} -> HIT ENEMY BODY at ({x}, {y})")
                        hit_enemy = True
                        break
                if hit_enemy:
                    break
            
            if hit_enemy:
                continue
            
            # This move is safe!
            print(f"  {direction} -> SAFE ({x}, {y})")
            safe.append(direction)
        
        return safe

    def get_new_position(self, head: Dict, direction: str) -> Tuple[int, int]:
        """Calculate new head position for a given direction"""
        dx, dy = self.direction_vectors[direction]
        return (head['x'] + dx, head['y'] + dy)

    def is_out_of_bounds(self, pos: Tuple[int, int], board: Dict) -> bool:
        """Check if position is outside board boundaries"""
        x, y = pos
        return x < 0 or x >= board['width'] or y < 0 or y >= board['height']

    def calculate_equal_length_danger(self, pos: Tuple[int, int], my_snake: Dict, board: Dict) -> float:
        """Extra penalty for moving near equal-length snakes to avoid tie deaths"""
        my_length = len(my_snake['body'])
        penalty = 0.0
        
        for snake in board['snakes']:
            if snake['id'] == my_snake['id']:
                continue
            
            enemy_length = len(snake['body'])
            
            # Only care about equal-length snakes (tie scenarios)
            if enemy_length == my_length:
                enemy_head = (snake['head']['x'], snake['head']['y'])
                distance = self.manhattan_distance(pos, enemy_head)
                
                # Very close to equal-length enemy - dangerous!
                if distance == 1:
                    penalty -= 300.0
                    print(f"  ⚠️⚠️ EXTREME TIE DANGER: Distance 1 from equal snake!")
                elif distance == 2:
                    penalty -= 150.0
                    print(f"  ⚠️ HIGH TIE RISK: Distance 2 from equal snake")
                elif distance == 3:
                    penalty -= 50.0
        
        return penalty

    def calculate_wall_proximity_score(self, pos: Tuple[int, int], board: Dict) -> float:
        """Penalize positions near walls"""
        x, y = pos
        width, height = board['width'], board['height']
        
        # Calculate distance from each wall
        dist_from_left = x
        dist_from_right = width - 1 - x
        dist_from_bottom = y
        dist_from_top = height - 1 - y
        
        min_wall_dist = min(dist_from_left, dist_from_right, dist_from_bottom, dist_from_top)
        
        # Heavy penalty for being right next to a wall
        if min_wall_dist == 0:
            return -50.0
        elif min_wall_dist == 1:
            return -10.0
        elif min_wall_dist == 2:
            return -5.0
        else:
            return 0.0

    def calculate_trap_score(self, pos: Tuple[int, int], my_snake: Dict, board: Dict) -> float:
        """Detect if a move leads to a trap/dead end using flood fill - ENHANCED for long snakes"""
        body_length = len(my_snake['body'])
        
        # Use larger limit for longer snakes to properly assess space
        search_limit = max(body_length * 3, 50)
        accessible = self.limited_flood_fill(pos, my_snake, board, limit=search_limit)
        
        accessible_count = len(accessible)
        
        # For long snakes, we need MUCH more space to survive
        if body_length >= 10:
            # Long snake - need lots of space
            if accessible_count < body_length:
                return -2000.0
            elif accessible_count < body_length * 1.5:
                return -800.0
            elif accessible_count < body_length * 2:
                return -400.0
            elif accessible_count < body_length * 2.5:
                return -150.0
            else:
                return 0.0
        elif body_length >= 6:
            # Medium snake
            if accessible_count < body_length:
                return -1500.0
            elif accessible_count < body_length * 1.5:
                return -500.0
            elif accessible_count < body_length * 2:
                return -200.0
            elif accessible_count < body_length * 2.5:
                return -50.0
            else:
                return 0.0
        else:
            # Short snake
            if accessible_count < body_length:
                return -1000.0
            elif accessible_count < body_length * 1.5:
                return -300.0
            elif accessible_count < body_length * 2:
                return -100.0
            else:
                return 0.0

    def limited_flood_fill(self, start_pos: Tuple[int, int], my_snake: Dict, board: Dict, limit: int = 50) -> set:
        """Fast flood fill with depth limit for trap detection"""
        visited = set()
        stack = [start_pos]
        
        # Get all occupied positions
        occupied = set()
        for snake in board['snakes']:
            for i, segment in enumerate(snake['body']):
                if i < len(snake['body']) - 1:
                    occupied.add((segment['x'], segment['y']))
        
        while stack and len(visited) < limit:
            x, y = stack.pop()
            
            if (x, y) in visited:
                continue
            
            # Check bounds
            if x < 0 or x >= board['width'] or y < 0 or y >= board['height']:
                continue
            
            # Check if occupied
            if (x, y) in occupied:
                continue
            
            visited.add((x, y))
            
            # Add adjacent cells
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                stack.append((x + dx, y + dy))
        
        return visited

    def calculate_future_space_score(self, pos: Tuple[int, int], my_snake: Dict, board: Dict) -> float:
        """Look ahead to see if this move leads to progressively tighter spaces"""
        body_length = len(my_snake['body'])
        
        # Only do this expensive check for longer snakes
        if body_length < 6:
            return 0.0
        
        # Simulate moving to this position and check future escape options
        future_escape_count = 0
        tight_futures = 0
        
        for direction in self.directions:
            future_pos = self.get_new_position({'x': pos[0], 'y': pos[1]}, direction)
            
            if not self.is_position_safe(future_pos, my_snake, board):
                continue
            
            future_escape_count += 1
            
            # Check how many moves we'd have from THAT position
            future_future_count = 0
            for direction2 in self.directions:
                future_future_pos = self.get_new_position({'x': future_pos[0], 'y': future_pos[1]}, direction2)
                if self.is_position_safe(future_future_pos, my_snake, board):
                    future_future_count += 1
            
            # If any future position has 0-1 escapes, that's a warning sign
            if future_future_count <= 1:
                tight_futures += 1
        
        # If most of our future options lead to tight spaces, heavily penalize
        if future_escape_count > 0:
            tight_ratio = tight_futures / future_escape_count
            
            if tight_ratio >= 0.75:
                return -300.0
            elif tight_ratio >= 0.5:
                return -150.0
            elif tight_ratio >= 0.25:
                return -50.0
        
        return 0.0

    def calculate_tail_chase_score(self, pos: Tuple[int, int], my_snake: Dict, board: Dict) -> float:
        """Smart tail chasing - follow tail when safe, CRITICAL for long snakes"""
        tail = my_snake['body'][-1]
        tail_pos = (tail['x'], tail['y'])
        body_length = len(my_snake['body'])
        
        # Calculate distance to our tail
        distance_to_tail = self.manhattan_distance(pos, tail_pos)
        
        # Check if tail will move this turn
        health = my_snake['health']
        tail_will_move = health < 100
        
        if not tail_will_move:
            return 0.0
        
        # Check accessible space
        accessible = len(self.limited_flood_fill(pos, my_snake, board, limit=50))
        space_ratio = accessible / body_length if body_length > 0 else 1.0
        
        # For long snakes in tight spaces, tail chasing becomes ESSENTIAL
        if body_length >= 10:
            if space_ratio < 2.5:
                if distance_to_tail == 1:
                    return 200.0
                elif distance_to_tail == 2:
                    return 100.0
                elif distance_to_tail <= 4:
                    return 50.0
            elif space_ratio < 3.5:
                if distance_to_tail <= 2:
                    return 80.0
                elif distance_to_tail <= 3:
                    return 30.0
        elif body_length >= 6:
            if space_ratio < 2.0:
                if distance_to_tail <= 2:
                    return 100.0
                elif distance_to_tail <= 4:
                    return 40.0
        else:
            if space_ratio < 2.0:
                if distance_to_tail <= 2:
                    return 50.0
                elif distance_to_tail <= 4:
                    return 20.0
        
        return 0.0

    def calculate_escape_route_score(self, pos: Tuple[int, int], my_snake: Dict, board: Dict) -> float:
        """Score based on number of escape routes - CRITICAL for long snakes"""
        escape_count = 0
        body_length = len(my_snake['body'])
        
        for direction in self.directions:
            future_pos = self.get_new_position({'x': pos[0], 'y': pos[1]}, direction)
            
            if self.is_position_safe(future_pos, my_snake, board):
                escape_count += 1
        
        # For long snakes, having multiple escape routes is CRITICAL
        if body_length >= 10:
            if escape_count == 0:
                return -1500.0
            elif escape_count == 1:
                return -600.0
            elif escape_count == 2:
                return -100.0
            elif escape_count == 3:
                return 150.0
            else:
                return 300.0
        elif body_length >= 6:
            if escape_count == 0:
                return -1000.0
            elif escape_count == 1:
                return -400.0
            elif escape_count == 2:
                return 0.0
            elif escape_count == 3:
                return 100.0
            else:
                return 200.0
        else:
            if escape_count == 0:
                return -800.0
            elif escape_count == 1:
                return -200.0
            elif escape_count == 2:
                return 0.0
            elif escape_count == 3:
                return 50.0
            else:
                return 100.0
    
    def is_position_safe(self, pos: Tuple[int, int], my_snake: Dict, board: Dict) -> bool:
        """Quick check if a position is safe"""
        x, y = pos
        
        # Check bounds
        if x < 0 or x >= board['width'] or y < 0 or y >= board['height']:
            return False
        
        # Check if occupied by any snake body
        for snake in board['snakes']:
            for segment in snake['body'][:-1]:
                if pos == (segment['x'], segment['y']):
                    return False
        
        return True

    def calculate_food_score(self, pos: Tuple[int, int], food: List[Dict]) -> float:
        """Calculate score based on proximity to food"""
        if not food:
            return 0.0
        
        min_distance = min(self.manhattan_distance(pos, (f['x'], f['y'])) for f in food)
        return self.weights['food_attraction'] * (10.0 / (min_distance + 1))

    def calculate_space_score(self, pos: Tuple[int, int], my_snake: Dict, board: Dict) -> float:
        """Calculate score based on available space"""
        accessible_spaces = self.limited_flood_fill(pos, my_snake, board, limit=50)
        return self.weights['space_control'] * len(accessible_spaces) * 0.1

    def calculate_center_score(self, pos: Tuple[int, int], board: Dict) -> float:
        """Calculate score for staying near the center"""
        center_x, center_y = board['width'] // 2, board['height'] // 2
        distance_from_center = self.manhattan_distance(pos, (center_x, center_y))
        max_distance = max(board['width'], board['height'])
        
        return self.weights['center_attraction'] * (1.0 - distance_from_center / max_distance)

    def calculate_enemy_avoidance_score(self, pos: Tuple[int, int], snakes: List[Dict], my_snake: Dict) -> float:
        """Calculate score for avoiding enemy snake bodies"""
        score = 0.0
        
        for snake in snakes:
            if snake['id'] == my_snake['id']:
                continue
            
            for segment in snake['body']:
                if pos == (segment['x'], segment['y']):
                    score -= self.weights['avoid_enemies']
        
        return score

    def calculate_head_to_head_score(self, pos: Tuple[int, int], snakes: List[Dict], my_snake: Dict) -> float:
        """Calculate score for head-to-head encounters with aggressive winning strategy"""
        score = 0.0
        my_length = len(my_snake['body'])
        my_head = (my_snake['head']['x'], my_snake['head']['y'])
        
        for snake in snakes:
            if snake['id'] == my_snake['id']:
                continue
            
            enemy_head = (snake['head']['x'], snake['head']['y'])
            enemy_length = len(snake['body'])
            length_advantage = my_length - enemy_length
            
            # Calculate all possible enemy next positions
            enemy_possible_positions = []
            for direction in self.directions:
                enemy_next_pos = self.get_new_position(snake['head'], direction)
                x, y = enemy_next_pos
                if not (x < 0 or x >= 11 or y < 0 or y >= 11):
                    hit_body = False
                    for seg in snake['body'][:-1]:
                        if enemy_next_pos == (seg['x'], seg['y']):
                            hit_body = True
                            break
                    if not hit_body:
                        enemy_possible_positions.append(enemy_next_pos)
            
            # Check if we're moving to a position where enemy could also move
            if pos in enemy_possible_positions:
                distance_between_heads = self.manhattan_distance(my_head, enemy_head)
                
                if distance_between_heads == 2:
                    if length_advantage >= 2:
                        score += 600.0
                        print(f"  ⚔️ DOMINATING HEAD-TO-HEAD: Huge advantage ({my_length} vs {enemy_length})")
                    elif my_length > enemy_length:
                        score += 400.0
                        print(f"  💪 HEAD-TO-HEAD ADVANTAGE: We're bigger ({my_length} vs {enemy_length})")
                    elif my_length == enemy_length:
                        score -= 800.0
                        print(f"  ☠️ HEAD-TO-HEAD TIE RISK: Equal size ({my_length})")
                    else:
                        score -= 600.0
                        print(f"  ⚠️ HEAD-TO-HEAD DANGER: We're smaller ({my_length} vs {enemy_length})")
                else:
                    if length_advantage >= 2:
                        score += 300.0
                    elif my_length > enemy_length:
                        score += 200.0
                    elif my_length == enemy_length:
                        score -= 400.0
                    else:
                        score -= 300.0
            
            # Check if we're moving adjacent to enemy head
            distance_to_enemy = self.manhattan_distance(pos, enemy_head)
            if distance_to_enemy == 1 and pos not in enemy_possible_positions:
                if length_advantage >= 2:
                    score += 150.0
                    print(f"  ✂️ CUTTING OFF much smaller enemy ({my_length} vs {enemy_length})")
                elif my_length > enemy_length:
                    score += 100.0
                    print(f"  ✂️ CUTTING OFF smaller enemy ({my_length} vs {enemy_length})")
                elif my_length == enemy_length:
                    score -= 150.0
                else:
                    score -= 100.0
            
            # Predict future collision course
            if distance_to_enemy == 2:
                for enemy_move in enemy_possible_positions:
                    if self.manhattan_distance(pos, enemy_move) == 1:
                        if length_advantage >= 2:
                            score += 120.0
                        elif my_length > enemy_length:
                            score += 80.0
                        elif my_length == enemy_length:
                            score -= 200.0
                        else:
                            score -= 120.0
        
        return score

    def manhattan_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
        """Calculate Manhattan distance between two positions"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def adjust_strategy(self, my_snake: Dict, board: Dict):
        """Dynamically adjust weights based on snake health, length, and positioning"""
        health = my_snake['health']
        length = len(my_snake['body'])
        
        self.weights = self.base_weights.copy()
        
        # VERY LONG SNAKE: Survival and space management is priority
        if length >= 15:
            self.apply_survival_strategy()
            print(f"  🐍 SURVIVAL MODE: Very long snake (L:{length})")
        
        # LONG SNAKE: Balance between space and strategy
        elif length >= 10:
            self.apply_long_snake_strategy()
            print(f"  📏 LONG SNAKE MODE: Need lots of space (L:{length})")
        
        # High health + Short length = AGGRESSIVE
        elif (health >= self.strategy_thresholds['high_health'] and 
            length <= self.strategy_thresholds['short_length']):
            self.apply_aggressive_strategy()
            
        # Low health + Short length = FOOD PRIORITY
        elif (health <= self.strategy_thresholds['low_health'] and 
              length <= self.strategy_thresholds['short_length']):
            self.apply_food_priority_strategy()
            
        # Low health + Long length = DEFENSIVE
        elif (health <= self.strategy_thresholds['low_health'] and 
              length >= self.strategy_thresholds['long_length']):
            self.apply_defensive_strategy()
            
        # High health + Long length = TERRITORIAL
        elif (health >= self.strategy_thresholds['high_health'] and 
              length >= self.strategy_thresholds['long_length']):
            self.apply_territorial_strategy()
            
        # Default: BALANCED
        else:
            self.apply_balanced_strategy()

    def apply_aggressive_strategy(self):
        self.weights.update({
            'avoid_enemies': 40.0,
            'head_to_head': -20.0,
            'food_attraction': 35.0,
            'space_control': 25.0,
            'center_attraction': 10.0,
        })

    def apply_food_priority_strategy(self):
        self.weights.update({
            'food_attraction': 60.0,
            'avoid_enemies': 120.0,
            'head_to_head': -120.0,
            'space_control': 5.0,
            'center_attraction': 2.0,
        })

    def apply_defensive_strategy(self):
        self.weights.update({
            'avoid_enemies': 150.0,
            'head_to_head': -100.0,
            'food_attraction': 40.0,
            'space_control': 30.0,
            'center_attraction': 2.0,
        })

    def apply_territorial_strategy(self):
        self.weights.update({
            'avoid_enemies': 60.0,
            'head_to_head': -30.0,
            'food_attraction': 15.0,
            'space_control': 35.0,
            'center_attraction': 15.0,
        })

    def apply_long_snake_strategy(self):
        """For long snakes (10-14 length) - prioritize space"""
        self.weights.update({
            'avoid_enemies': 90.0,
            'head_to_head': -60.0,
            'food_attraction': 20.0,
            'space_control': 50.0,
            'center_attraction': 5.0,
        })

    def apply_survival_strategy(self):
        """For very long snakes (15+) - all about not self-trapping"""
        self.weights.update({
            'avoid_enemies': 100.0,
            'head_to_head': -80.0,
            'food_attraction': 15.0,
            'space_control': 60.0,
            'center_attraction': 2.0,
        })

    def apply_balanced_strategy(self):
        pass

    def get_current_strategy(self, my_snake: Dict) -> str:
        """Get a description of the current strategy"""
        health = my_snake['health']
        length = len(my_snake['body'])
        
        if length >= 15:
            return f"SURVIVAL (H:{health}, L:{length})"
        
        if length >= 10:
            return f"LONG_SNAKE (H:{health}, L:{length})"
        
        if (health >= self.strategy_thresholds['high_health'] and 
            length <= self.strategy_thresholds['short_length']):
            return f"AGGRESSIVE (H:{health}, L:{length})"
            
        elif (health <= self.strategy_thresholds['low_health'] and 
              length <= self.strategy_thresholds['short_length']):
            return f"FOOD_PRIORITY (H:{health}, L:{length})"
            
        elif (health <= self.strategy_thresholds['low_health'] and 
              length >= self.strategy_thresholds['long_length']):
            return f"DEFENSIVE (H:{health}, L:{length})"
            
        elif (health >= self.strategy_thresholds['high_health'] and 
              length >= self.strategy_thresholds['long_length']):
            return f"TERRITORIAL (H:{health}, L:{length})"
            
        else:
            return f"BALANCED (H:{health}, L:{length})"

    def update_weights(self, new_weights: Dict[str, float]):
        """Update the weights for different factors"""
        self.weights.update(new_weights)

    def get_weights(self) -> Dict[str, float]:
        """Get current weights"""
        return self.weights.copy()


# Flask server integration
def create_battlesnake_server():
    """Create Flask server for Battlesnake"""
    try:
        from flask import Flask, request, jsonify
        app = Flask(__name__)
        snake_logic = BattlesnakeLogic()
        
        @app.route('/')
        def info():
            return jsonify({
                "apiversion": "1",
                "author": "YourUsername",
                "color": "#888888",
                "head": "default",
                "tail": "default"
            })
        
        @app.route('/start', methods=['POST'])
        def start():
            return "OK"
        
        @app.route('/move', methods=['POST'])
        def move():
            game_state = request.get_json()
            move = snake_logic.get_move(game_state)
            return jsonify({"move": move})
        
        @app.route('/end', methods=['POST'])
        def end():
            return "OK"
        
        return app
    except ImportError:
        print("Flask not available. Install with: pip install flask")
        return None


# Example usage and testing
if __name__ == "__main__":
    # Example game state for testing
    example_game_state = {
        "game": {"id": "test", "timeout": 500},
        "turn": 1,
        "board": {
            "height": 11,
            "width": 11,
            "food": [{"x": 5, "y": 5}],
            "snakes": [
                {
                    "id": "my-snake",
                    "name": "My Snake",
                    "health": 90,
                    "body": [{"x": 1, "y": 1}, {"x": 1, "y": 0}],
                    "head": {"x": 1, "y": 1}
                }
            ]
        },
        "you": {
            "id": "my-snake",
            "name": "My Snake",
            "health": 90,
            "body": [{"x": 1, "y": 1}, {"x": 1, "y": 0}],
            "head": {"x": 1, "y": 1}
        }
    }
    
    # Test the logic
    snake_logic = BattlesnakeLogic()
    
    print("=== TESTING BATTLESNAKE LOGIC ===")
    move = snake_logic.get_move(example_game_state)
    print(f"Recommended move: {move}")
    
    # Create Flask server if available
    app = create_battlesnake_server()
    if app:
        print("\nFlask server created. Run with: python battlesnake.py")
        # Uncomment to run: app.run(host='0.0.0.0', port=8080)
