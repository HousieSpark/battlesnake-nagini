from flask import Flask, request, jsonify
from battlesnake import BattlesnakeLogic

# Create Flask app and snake logic
app = Flask(__name__)
snake_logic = BattlesnakeLogic()

@app.route('/')
def info():
    return jsonify({
        "apiversion": "1",
        "author": "YourUsername",  # Change this to your name
        "color": "#FF0000",        # Change snake color (hex)
        "head": "default",         # Snake head style
        "tail": "default"          # Snake tail style
    })

@app.route('/start', methods=['POST'])
def start():
    print("🐍 New game starting!")
    return "OK"

@app.route('/move', methods=['POST'])
def move():
    try:
        game_state = request.get_json()
        move = snake_logic.get_move(game_state)
        print(f"🎯 Making move: {move}")
        return jsonify({"move": move})
    except Exception as e:
        print(f"❌ Error: {e}")
        # Fallback to safe move
        return jsonify({"move": "up"})

@app.route('/end', methods=['POST'])
def end():
    print("🏁 Game ended!")
    return "OK"

# Health check endpoint
@app.route('/health')
def health():
    return jsonify({"status": "healthy"})

if __name__ == "__main__":
    print("🐍 Battlesnake server starting on Replit!")
    app.run(host='0.0.0.0', port=8080, debug=True)
