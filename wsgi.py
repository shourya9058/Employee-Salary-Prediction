from app import app, init_model

if __name__ == "__main__":
    # Initialize the model when the WSGI server starts
    init_model()
    app.run()
