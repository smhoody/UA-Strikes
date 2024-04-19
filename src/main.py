from flask import Flask, request, render_template, url_for, jsonify
import folium
import geopandas
from ua_strikes import StrikeNN
from util import Util
from stats import Stats

app = Flask(__name__)
trained = False #flag for if the neural network has been trained or not

util = Util() #instantiate utility class
util.read_data() #read missile strike data
network = StrikeNN(data=util.training_data, test_data=util.testing_data) #instantiate neural network
network.train(epochs=100) #train network
network.test()

@app.route('/', methods=['GET', 'POST'])
def index():
    input_date = "02/24/2022"
    strike_marker_count = 0
    if (request.method == "POST"): #if a POST request was made on the index page
        input_date = request.form.get("date", 0) #get date from input
        strike_marker_count = int(request.form.get("strike-count", 0))

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
    strikes_slider_element = """
                    <div style="position: fixed; right: 10vw; z-index:10000;">
                        <form id="strike-markers-form" method="post">
                            <input type="range" id="strike-count" name="strike-count">
                            <button type="submit" id="marker-button">Place markers</button>
                        </form>
                    </div>
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
    m.get_root().html.add_child(folium.Element(title_html + date_select_element + strikes_slider_element))

    #fill Ukraine outline on map 
    folium.GeoJson(ua_geojson, style_function=lambda feature: {
        "fillColor": "#0056B9",
        "fillOpacity": 0.15,
        "color": "black",
        "weight":0
    }).add_to(m)

    # Get prediction from neural network
    coord_prediction_region = network.predict_m1(input_date)
    coord_prediction_precision = network.predict(input_date)

    #draw prediction area on map
    folium.Circle(location=coord_prediction_region,
                        radius=100000, #radius in meters
                        color="orange",
                        fill=True,
                        fill_opacity=0.5,
                        opacity=1).add_to(m)
    folium.Circle(location=coord_prediction_precision,
                        radius=25000,
                        color="red",
                        fill=True,
                        fill_opacity=0.8,
                        opacity=1).add_to(m)

    stats = Stats()
    all_strikes_dict = stats.get_all_strikes_by_day()
    #re-format strikes from {"{day}:[[30,40],[34,45]], ..."}
    # into => [[30,40],[34,45],...]
    strikes_list = [coord for coords_list in all_strikes_dict.values() for coord in coords_list]
    for i in range(int(len(strikes_list)*(strike_marker_count/100))):
        folium.Marker(location=strikes_list[i]).add_to(m)
    print(f"Marker count: {strike_marker_count}")
    map_html = m.get_root().render()

    return render_template('index.html', map=map_html)

if __name__ == "__main__":
    app.run(debug=True)