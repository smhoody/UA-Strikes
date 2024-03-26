from flask import Flask, render_template, url_for
import folium

app = Flask(__name__)

@app.route('/')
def index():
    m = folium.Map([48.253788, 30.714256], zoom_start=6)
    folium.Circle(location=[50.072133,36.142258],
                        radius=100000,
                        color="red",
                        fill=True,
                        fill_opacity=0.5,
                        opacity=1).add_to(m)
    
    m.save("web/templates/index2.html")
    return render_template("index2.html")

if __name__ == "__main__":
    app.run(debug=True)