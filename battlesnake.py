import json
import random
from typing import Dict, List, Tuple, Optional

class BattlesnakeLogic:
    def __init__(self):
        # Base weights for different move factors
        self.base_weights = {
            'avoid_walls': 100.0,        # Penalty for hitting walls
            'avoid_body': 100.0,         # Penalty for hitting own body
            'avoid_enemies': 80.0,       # Penalty for hitting enemy snakes
            'food_attraction': 20.0,     # Bonus for moving toward food
            'center_attraction': 5.0,    # Bonus for staying near center
            'space_control': 15.0,       # Bonus for areas with more space
            'head_to_head': -50.0,       # Penalty for risky head-to-head encounters
        }
        
        # Strategy thresholds
        self.strategy_thresholds = {
            'low_health': 30,      # Health below this is considered low
            'high_health': 70,     # Health above this is considered high
            'short_length': 4,     # Length below this is considered short
            'long_length': 8,      # Length above this is considered long
        }
        
        # Current active weights (will be dynamically adjusted)
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
        
        # Dynamically adjust strategy based on snake state
        self.adjust_strategy(my_snake, board)
        
        # Calculate scores for all possible moves
        move_scores = self.calculate_move_scores(my_snake, board)
        
        # Sort moves by score (highest first)
        sorted_moves = sorted(move_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Return the best move, or random if all moves are equally bad
        if sorted_moves:
            best_move = sorted_moves[0][0]
            strategy = self.get_current_strategy(my_snake)
            print(f"Strategy: {strategy}, Move scores: {dict(sorted_moves)}")
            return best_move
        
        return random.choice(self.directions)

    def calculate_move_scores(self, my_snake: Dict, board: Dict) -> Dict[str, float]:
        """Calculate weighted scores for each possible move"""
        head = my_snake['head']
        scores = {}
        
        for direction in self.directions:
            new_pos = self.get_new_position(head, direction)
            score = 0.0
            
            # Check if move is valid (not hitting walls or body)
            if not self.is_valid_move(new_pos, my_snake, board):
                score -= self.weights['avoid_walls']
                scores[direction] = score
                continue
            
            # SURVIVAL CHECKS (High Priority)
            # Check for traps and dead ends
            trap_penalty = self.calculate_trap_score(new_pos, my_snake, board)
            score += trap_penalty
            
            # If trapped badly, heavily penalize this move
            if trap_penalty < -500:
                scores[direction] = score
                continue
            
            # Calculate escape routes
            escape_score = self.calculate_escape_route_score(new_pos, my_snake, board)
            score += escape_score
            
            # Smart tail chasing
            tail_score = self.calculate_tail_chase_score(new_pos, my_snake, board)
            score += tail_score
            
            # STRATEGIC FACTORS
            score += self.calculate_food_score(new_pos, board['food'])
            score += self.calculate_space_score(new_pos, my_snake, board)
            score += self.calculate_center_score(new_pos, board)
            score += self.calculate_enemy_avoidance_score(new_pos, board['snakes'], my_snake)
            score += self.calculate_head_to_head_score(new_pos, board['snakes'], my_snake)
            
            scores[direction] = score
        
        return scores

    def get_new_position(self, head: Dict, direction: str) -> Tuple[int, int]:
        """Calculate new head position for a given direction"""
        dx, dy = self.direction_vectors[direction]
        return (head['x'] + dx, head['y'] + dy)

    def is_valid_move(self, pos: Tuple[int, int], my_snake: Dict, board: Dict) -> bool:
        """Check if a position is valid (not hitting walls or own body)"""
        x, y = pos
        
        # Check walls
        if x < 0 or x >= board['width'] or y < 0 or y >= board['height']:
            return False
        
        # Check own body (excluding tail which will move, unless we just ate)
        body_positions = [(segment['x'], segment['y']) for segment in my_snake['body'][:-1]]
        if pos in body_positions:
            return False
        
        # Check if we're about to collide with enemy body
        for snake in board['snakes']:
            if snake['id'] == my_snake['id']:
                continue
            # Check enemy body (excluding their tail too)
            enemy_body = [(segment['x'], segment['y']) for segment in snake['body'][:-1]]
            if pos in enemy_body:
                return False
        
        return True

    def calculate_trap_score(self, pos: Tuple[int, int], my_snake: Dict, board: Dict) -> float:
        """Detect if a move leads to a trap/dead end using limited flood fill"""
        # Quick flood fill to check accessible spaces
        accessible = self.limited_flood_fill(pos, my_snake, board, limit=len(my_snake['body']) + 3)
        
        # If we have less accessible space than our body length, we're likely trapped
        body_length = len(my_snake['body'])
        
        if len(accessible) < body_length:
            # Severe trap - we'll definitely die
            return -1000.0
        elif len(accessible) < body_length * 1.5:
            # Risky area - might get trapped
            return -300.0
        elif len(accessible) < body_length * 2:
            # Somewhat confined
            return -100.0
        else:
            # Good amount of space
            return 0.0

    def limited_flood_fill(self, start_pos: Tuple[int, int], my_snake: Dict, board: Dict, limit: int = 50) -> set:
        """Fast flood fill with depth limit for trap detection"""
        visited = set()
        stack = [start_pos]
        
        # Get all occupied positions
        occupied = set()
        for snake in board['snakes']:
            for i, segment in enumerate(snake['body']):
                # Don't count tails as occupied (they move)
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
        
        # Calculate distance to our tail
        distance_to_tail = self.manhattan_distance(pos, tail_pos)
        
        # Check if tail will move this turn (if we just ate, tail stays)
        health = my_snake['health']
        tail_will_move = health < 100  # Tail moves unless we just ate
        
        if not tail_will_move:
            # Tail won't move, don't chase it
            return 0.0
        
        # If we're low on space, following tail is good
        accessible = len(self.limited_flood_fill(pos, my_snake, board, limit=30))
        
        if accessible < len(my_snake['body']) * 2:
            # Low space - tail chasing is a survival tactic
            if distance_to_tail <= 2:
                return 50.0  # Good move
            elif distance_to_tail <= 4:
                return 20.0  # Okay move
        
        return 0.0

    def calculate_escape_route_score(self, pos: Tuple[int, int], my_snake: Dict, board: Dict) -> float:
        """Score based on number of escape routes from this position"""
        # Count how many valid moves we'd have from this position
        escape_count = 0
        future_positions = []
        
        for direction in self.directions:
            future_pos = self.get_new_position({'x': pos[0], 'y': pos[1]}, direction)
            
            # Quick check if this future position is valid
            if self.is_position_safe(future_pos, my_snake, board):
                escape_count += 1
                future_positions.append(future_pos)
        
        # Score based on number of escape routes
        if escape_count == 0:
            return -800.0  # Death trap!
        elif escape_count == 1:
            return -200.0  # Very risky - only one way out
        elif escape_count == 2:
            return 0.0     # Acceptable
        elif escape_count == 3:
            return 50.0    # Good
        else:
            return 100.0   # Excellent - maximum mobility
    
    def is_position_safe(self, pos: Tuple[int, int], my_snake: Dict, board: Dict) -> bool:
        """Quick check if a position is safe (for escape route calculation)"""
        x, y = pos
        
        # Check bounds
        if x < 0 or x >= board['width'] or y < 0 or y >= board['height']:
            return False
        
        # Check if occupied by any snake body
        for snake in board['snakes']:
            for segment in snake['body'][:-1]:  # Exclude tails
                if pos == (segment['x'], segment['y']):
                    return False
        
        return True

    def calculate_food_score(self, pos: Tuple[int, int], food: List[Dict]) -> float:
        """Calculate score based on proximity to food"""
        if not food:
            return 0.0
        
        min_distance = min(self.manhattan_distance(pos, (f['x'], f['y'])) for f in food)
        # Closer food = higher score
        return self.weights['food_attraction'] * (10.0 / (min_distance + 1))

    def calculate_space_score(self, pos: Tuple[int, int], my_snake: Dict, board: Dict) -> float:
        """Calculate score based on available space using flood fill"""
        accessible_spaces = self.flood_fill(pos, my_snake, board)
        return self.weights['space_control'] * len(accessible_spaces)

    def calculate_center_score(self, pos: Tuple[int, int], board: Dict) -> float:
        """Calculate score for staying near the center of the board"""
        center_x, center_y = board['width'] // 2, board['height'] // 2
        distance_from_center = self.manhattan_distance(pos, (center_x, center_y))
        max_distance = max(board['width'], board['height'])
        
        # Closer to center = higher score
        return self.weights['center_attraction'] * (1.0 - distance_from_center / max_distance)

    def calculate_enemy_avoidance_score(self, pos: Tuple[int, int], snakes: List[Dict], my_snake: Dict) -> float:
        """Calculate score for avoiding enemy snake bodies"""
        score = 0.0
        
        for snake in snakes:
            if snake['id'] == my_snake['id']:
                continue
            
            # Check collision with enemy body
            for segment in snake['body']:
                if pos == (segment['x'], segment['y']):
                    score -= self.weights['avoid_enemies']
        
        return score

    def calculate_head_to_head_score(self, pos: Tuple[int, int], snakes: List[Dict], my_snake: Dict) -> float:
        """Calculate score for head-to-head encounters"""
        score = 0.0
        
        for snake in snakes:
            if snake['id'] == my_snake['id']:
                continue
            
            enemy_head = (snake['head']['x'], snake['head']['y'])
            
            # Check if we're moving adjacent to enemy head
            if self.manhattan_distance(pos, enemy_head) == 1:
                # If we're smaller or equal size, avoid head-to-head
                if len(my_snake['body']) <= len(snake['body']):
                    score += self.weights['head_to_head']
        
        return score

    def flood_fill(self, start_pos: Tuple[int, int], my_snake: Dict, board: Dict) -> set:
        """Use flood fill to calculate accessible spaces"""
        visited = set()
        stack = [start_pos]
        
        # Get all occupied positions
        occupied = set()
        for snake in board['snakes']:
            for segment in snake['body']:
                occupied.add((segment['x'], segment['y']))
        
        while stack:
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

    def adjust_strategy(self, my_snake: Dict, board: Dict):
        """Dynamically adjust weights based on snake health and length"""
        health = my_snake['health']
        length = len(my_snake['body'])
        
        # Reset to base weights
        self.weights = self.base_weights.copy()
        
        # Strategy 1: High health + Short length = AGGRESSIVE
        if (health >= self.strategy_thresholds['high_health'] and 
            length <= self.strategy_thresholds['short_length']):
            self.apply_aggressive_strategy()
            
        # Strategy 2: Low health + Short length = FOOD PRIORITY
        elif (health <= self.strategy_thresholds['low_health'] and 
              length <= self.strategy_thresholds['short_length']):
            self.apply_food_priority_strategy()
            
        # Strategy 3: Low health + Long length = DEFENSIVE
        elif (health <= self.strategy_thresholds['low_health'] and 
              length >= self.strategy_thresholds['long_length']):
            self.apply_defensive_strategy()
            
        # Strategy 4: High health + Long length = TERRITORIAL
        elif (health >= self.strategy_thresholds['high_health'] and 
              length >= self.strategy_thresholds['long_length']):
            self.apply_territorial_strategy()
            
        # Default: Moderate health/length = BALANCED
        else:
            self.apply_balanced_strategy()

    def apply_aggressive_strategy(self):
        """High health + short length: Take risks, seek confrontation"""
        self.weights.update({
            'avoid_enemies': 40.0,      # Reduced enemy avoidance
            'head_to_head': -20.0,      # Less penalty for head-to-head
            'food_attraction': 35.0,    # Increased food seeking
            'space_control': 25.0,      # More space control
            'center_attraction': 10.0,  # Stay central for opportunities
        })

    def apply_food_priority_strategy(self):
        """Low health + short length: Must find food quickly"""
        self.weights.update({
            'food_attraction': 60.0,    # Maximum food attraction
            'avoid_enemies': 120.0,     # High enemy avoidance
            'head_to_head': -80.0,      # Strong head-to-head avoidance
            'space_control': 5.0,       # Less concern for space
            'center_attraction': 2.0,   # Less concern for center
        })

    def apply_defensive_strategy(self):
        """Low health + long length: Protect investment, avoid risks"""
        self.weights.update({
            'avoid_enemies': 150.0,     # Maximum enemy avoidance
            'head_to_head': -100.0,     # Maximum head-to-head penalty
            'food_attraction': 40.0,    # Moderate food seeking
            'space_control': 30.0,      # High space control
            'center_attraction': 2.0,   # Less central positioning
        })

    def apply_territorial_strategy(self):
        """High health + long length: Control space, strategic positioning"""
        self.weights.update({
            'avoid_enemies': 60.0,      # Moderate enemy avoidance
            'head_to_head': -30.0,      # Moderate head-to-head penalty
            'food_attraction': 15.0,    # Lower food priority
            'space_control': 35.0,      # Maximum space control
            'center_attraction': 15.0,  # Strong center control
        })

    def apply_balanced_strategy(self):
        """Default balanced approach"""
        # Keep base weights - no changes needed
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


# Flask server endpoints (if using Flask framework)
def create_battlesnake_server():
    """Example of how to integrate with a web framework"""
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
    
    # Test the logic with different scenarios
    snake_logic = BattlesnakeLogic()
    
    # Test different snake states
    test_scenarios = [
        # Scenario 1: High health, short length (should be aggressive)
        {
            **example_game_state,
            "you": {
                "id": "my-snake",
                "name": "My Snake", 
                "health": 80,
                "body": [{"x": 1, "y": 1}, {"x": 1, "y": 0}],  # Length 2
                "head": {"x": 1, "y": 1}
            }
        },
        # Scenario 2: Low health, short length (should prioritize food)
        {
            **example_game_state,
            "you": {
                "id": "my-snake",
                "name": "My Snake",
                "health": 20,
                "body": [{"x": 1, "y": 1}, {"x": 1, "y": 0}, {"x": 0, "y": 0}],  # Length 3
                "head": {"x": 1, "y": 1}
            }
        },
        # Scenario 3: Low health, long length (should be defensive)
        {
            **example_game_state,
            "you": {
                "id": "my-snake",
                "name": "My Snake",
                "health": 25,
                "body": [{"x": i, "y": 1} for i in range(9)],  # Length 9
                "head": {"x": 0, "y": 1}
            }
        }
    ]
    
    print("=== TESTING ADAPTIVE STRATEGIES ===")
    for i, scenario in enumerate(test_scenarios, 1):
        # Update the board's snakes array to match
        scenario["board"]["snakes"] = [scenario["you"]]
        
        print(f"\nScenario {i}:")
        move = snake_logic.get_move(scenario)
        print(f"Recommended move: {move}")
        print(f"Active weights: {snake_logic.get_weights()}")
    
    # Example: Customize strategy thresholds
    print("\n=== CUSTOMIZING THRESHOLDS ===")
    snake_logic.update_strategy_thresholds({
        'low_health': 40,    # More conservative health threshold
        'short_length': 5,   # Adjust short length threshold
    })
    print("Updated thresholds:", snake_logic.strategy_thresholds)
    
    # Create Flask server if available
    app = create_battlesnake_server()
    if app:
        print("Flask server created. Run with: python battlesnake.py")
        # Uncomment to run: app.run(host='0.0.0.0', port=8080)
