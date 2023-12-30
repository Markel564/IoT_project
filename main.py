"""
This is the main file of the project. It is the one that is run when we run the command "python main.py" in the terminal.
It creates the app and runs it.
"""
from website import create_app

app = create_app()


if __name__ == "__main__":

    app.run(debug=True)