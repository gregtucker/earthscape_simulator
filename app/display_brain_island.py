from flask import Flask, url_for

app = Flask(__name__)

html_for_island_display = """
<!DOCTYPE html>
<html>
    <head>
        <title>Welcome to Brain Island!</title>
        <meta http-equiv='refresh' content='340'>
    </head>
    <body>
        <img src='/static/current_island.png'>
    </body>
</html>
"""

@app.route("/")
def hello_world():
    return html_for_island_display

