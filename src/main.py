from flask import Flask, request, render_template, url_for, jsonify
import folium
import geopandas
from ua_strikes import StrikeNN
from util import Util

app = Flask(__name__)
trained = False #flag for if the neural network has been trained or not

util = Util() #instantiate utility class
util.read_data() #read missile strike data
network = StrikeNN(data=util.training_data, test_data=util.testing_data) #instantiate neural network
network.train(epochs=1000) #train network
network.test()

@app.route('/', methods=['GET', 'POST'])
def index():
    input_date = "02/24/2022"
    if (request.method == "POST"): #if a POST request was made on the index page
        input_date = request.form["date"] #get date from input

    '''
    Ukraine coordinate bounds
    Latitude: [52.36586, 44.40852]  N-S
    Longitude: [22.19458, 40.15347]  W-E
    '''
    ua_geojson = geopandas.read_file("geoBoundaries-UKR.geojson") 
    #geojson data creates the outline of Ukraine on the map

    loc = 'UA Strikes' #title on the index page
    title_html = f'<h1 style="font-family: system-ui; position:absolute;z-index:100000;left:40vw;background:linear-gradient(rgba(0, 87, 183,0.6), rgba(255, 221, 0, 0.6)); border-radius: 10px; margin: 10px; padding-left: 10px; padding-right: 10px;" >{loc}</h1>' 
    date_select_element = f"""<div style="position: fixed; left: 10vw; z-index:10000;"> 
                        <form id="dateForm" method="post">
                            <input type="date" id="date" name="date" required>
                            <button type="submit" id="date-button">Predict</button>
                        </form>
                    </div>        
                """
    date_js_function = """
    <script>
        function submitDate() {
            
        }
    </script>
    """
    #zoom range 6-12 (farther - closer)
    m = folium.Map(location=[48.253788, 30.714256], 
                   max_bounds=True,
                   min_lat=42,
                   max_lat=53,
                   min_lon=20,
                   max_lon=45,
                   zoom_start=6,
                   min_zoom=5,
                   max_zoom=12)
    
    #add HTML elements to map page
    m.get_root().html.add_child(folium.Element(title_html + date_select_element + date_js_function))

    #fill Ukraine outline on map 
    folium.GeoJson(ua_geojson, style_function=lambda feature: {
        "fillColor": "#0056B9",
        "fillOpacity": 0.15,
        "color": "black",
        "weight":0
    }).add_to(m)

    # Get prediction from neural network
    coord_prediction = network.predict(input_date)

    #draw prediction area on map
    folium.Circle(location=coord_prediction,
                        radius=100000, #radius in meters
                        color="red",
                        fill=True,
                        fill_opacity=0.5,
                        opacity=1).add_to(m)

    map_html = m.get_root().render()

    return render_template('index.html', map=map_html)

if __name__ == "__main__":
    app.run(debug=True)