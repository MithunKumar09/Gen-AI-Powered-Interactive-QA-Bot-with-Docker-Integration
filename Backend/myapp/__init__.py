# Backend/myapp/__init__.py
import logging
from flask import Flask
from myapp.routes import routes

def create_app():
    app = Flask(__name__)
    app.config['DEBUG'] = True

    # Register routes blueprint
    app.register_blueprint(routes)

    # Set up logging
    logging.basicConfig(level=logging.DEBUG)
    app.logger.debug('App initialized')

    return app
