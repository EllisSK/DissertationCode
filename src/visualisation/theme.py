import plotly.io as pio
import plotly.graph_objects as go

def create_custom_template():
    custom_template = go.layout.Template()

    COLOURS = [
        "008dff",
        "d83034",
        "4ecb8d",
        "c701ff",
        "ff9d3a"
    ]

    custom_template.layout = {
        "font" : {
            "family" : "Aptos, sans-serif",
            "size" : 12,
            "color" : "black"
        },
        "plot_bgcolor" : "white",
        "paper_bgcolor" : "white",
        "colorway" : COLOURS,
        "xaxis": {
            "showline": True, 
            "linewidth": 1, 
            "linecolor": "black",
            "showgrid": False, 
            "ticks": "outside"
        },
        "yaxis": {
            "showline": True, 
            "linewidth": 1, 
            "linecolor": "black",
            "showgrid": False, 
            "ticks": "outside"
        }
    }

    pio.templates["custom_template"] = custom_template
    pio.templates.default = "custom_template"