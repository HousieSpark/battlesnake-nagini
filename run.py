from battlesnake import create_battlesnake_server

if __name__ == "__main__":
    app = create_battlesnake_server()
    if app:
        print("🐍 Battlesnake server starting...")
        print("📡 Server running at: http://localhost:8080")
        print("🎮 Use this URL in Battlesnake games!")
        app.run(host='0.0.0.0', port=8080, debug=True)
