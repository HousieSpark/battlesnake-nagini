import json
import random
from typing import Dict, List, Tuple, Optional

class BattlesnakeLogic:
    def __init__(self):
        # Adjustable weights for different move factors
        self.weights = {
            'avoid_walls': 100.0,        # Penalty for hitting walls
            'avoid_body': 100.0,         # Penalty for hitting own body
            'avoid_enemies': 80.0,       # Penalty for hitting enemy snakes
            'food_attraction': 20.0,     # Bonus for moving toward food
            'center_attraction': 5.0,    # Bonus for staying near center
            'space_control': 15.0,       # Bonus for areas with more space
            'head_to_head': -50.0,       # Penalty for risky head-to-head encounters
        }
        
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
        
        # Calculate scores for all possible moves
        move_scores = self.calculate_move_scores(my_snake, board)
        
        # Sort moves by score (highest first)
        sorted_moves = sorted(move_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Return the best move, or random if all moves are equally bad
        if sorted_moves:
            best_move = sorted_moves[0][0]
            print(f"Move scores: {dict(sorted_moves)}")
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
            
            # Calculate various factors
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
        
        # Check own body (excluding tail which will move)
        body_positions = [(segment['x'], segment['y']) for segment in my_snake['body'][:-1]]
        if pos in body_positions:
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

    def manhattan_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
        """Calculate Manhattan distance between two positions"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

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
