# backend/myapp/__init__.py
from flask import Flask
from myapp.routes import routes

def create_app():
    app = Flask(__name__)
    app.config['DEBUG'] = True

    # Register routes blueprint
    app.register_blueprint(routes)

    return app

if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=5000)
