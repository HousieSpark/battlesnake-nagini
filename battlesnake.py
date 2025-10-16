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
            'food_attraction': 20.0,
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
            return -20.0
        elif min_wall_dist == 2:
            return -5.0
        else:
            return 0.0

    def calculate_trap_score(self, pos: Tuple[int, int], my_snake: Dict, board: Dict) -> float:
        """Detect if a move leads to a trap/dead end using limited flood fill"""
        accessible = self.limited_flood_fill(pos, my_snake, board, limit=len(my_snake['body']) + 3)
        
        body_length = len(my_snake['body'])
        
        if len(accessible) < body_length:
            return -1000.0
        elif len(accessible) < body_length * 1.5:
            return -300.0
        elif len(accessible) < body_length * 2:
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

    def calculate_tail_chase_score(self, pos: Tuple[int, int], my_snake: Dict, board: Dict) -> float:
        """Smart tail chasing - follow tail when safe"""
        tail = my_snake['body'][-1]
        tail_pos = (tail['x'], tail['y'])
        
        distance_to_tail = self.manhattan_distance(pos, tail_pos)
        
        health = my_snake['health']
        tail_will_move = health < 100
        
        if not tail_will_move:
            return 0.0
        
        accessible = len(self.limited_flood_fill(pos, my_snake, board, limit=30))
        
        if accessible < len(my_snake['body']) * 2:
            if distance_to_tail <= 2:
                return 50.0
            elif distance_to_tail <= 4:
                return 20.0
        
        return 0.0

    def calculate_escape_route_score(self, pos: Tuple[int, int], my_snake: Dict, board: Dict) -> float:
        """Score based on number of escape routes from this position"""
        escape_count = 0
        
        for direction in self.directions:
            future_pos = self.get_new_position({'x': pos[0], 'y': pos[1]}, direction)
            
            if self.is_position_safe(future_pos, my_snake, board):
                escape_count += 1
        
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
        
        # Check bounds - CRITICAL FIX
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
        """Calculate score for head-to-head encounters"""
        score = 0.0
        my_length = len(my_snake['body'])
        
        for snake in snakes:
            if snake['id'] == my_snake['id']:
                continue
            
            enemy_head = (snake['head']['x'], snake['head']['y'])
            enemy_length = len(snake['body'])
            
            if self.manhattan_distance(pos, enemy_head) == 1:
                enemy_possible_moves = []
                for direction in self.directions:
                    enemy_next_pos = self.get_new_position(snake['head'], direction)
                    if self.is_position_safe(enemy_next_pos, snake, board={'snakes': snakes, 'width': 11, 'height': 11}):
                        enemy_possible_moves.append(enemy_next_pos)
                
                if pos in enemy_possible_moves:
                    if my_length > enemy_length:
                        score += 100.0
                    elif my_length == enemy_length:
                        score += self.weights['head_to_head'] * 0.5
                    else:
                        score += self.weights['head_to_head']
                else:
                    if my_length <= enemy_length:
                        score -= 30.0
        
        return score

    def manhattan_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
        """Calculate Manhattan distance between two positions"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def adjust_strategy(self, my_snake: Dict, board: Dict):
        """Dynamically adjust weights based on snake health and length"""
        health = my_snake['health']
        length = len(my_snake['body'])
        
        self.weights = self.base_weights.copy()
        
        if (health >= self.strategy_thresholds['high_health'] and 
            length <= self.strategy_thresholds['short_length']):
            self.apply_aggressive_strategy()
            
        elif (health <= self.strategy_thresholds['low_health'] and 
              length <= self.strategy_thresholds['short_length']):
            self.apply_food_priority_strategy()
            
        elif (health <= self.strategy_thresholds['low_health'] and 
              length >= self.strategy_thresholds['long_length']):
            self.apply_defensive_strategy()
            
        elif (health >= self.strategy_thresholds['high_health'] and 
              length >= self.strategy_thresholds['long_length']):
            self.apply_territorial_strategy()
            
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
            'head_to_head': -80.0,
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

    def apply_balanced_strategy(self):
        pass

    def get_current_strategy(self, my_snake: Dict) -> str:
        """Get a description of the current strategy"""
        health = my_snake['health']
        length = len(my_snake['body'])
        
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
