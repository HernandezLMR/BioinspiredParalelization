import gradio as gr
import os
import json
from interfaces.extraction import math_extract

CONFIG_DIR = "./saved_exp"  # or whatever your config directory is

def list_config_files():
    return [f for f in os.listdir(CONFIG_DIR) if f.endswith('.json')]

def load_config(filename):
    if not filename:
        return "No file selected."
    filepath = os.path.join(CONFIG_DIR, filename)
    with open(filepath, 'r') as f:
        data = json.load(f)
    info = math_extract(data)
    return gr.Markdown(value=info, visible=True), gr.Accordion(visible=True)

with gr.Blocks() as demo:
    with gr.Row():
        config_dropdown = gr.Dropdown(choices=list_config_files(), label="Select a config file")
        refresh_button = gr.Button("🔄 Refresh")
    
    load_button = gr.Button("Load Config")
    with gr.Accordion('Config Content', open=True,visible=False,elem_id="acc1") as accordion:
        config_output = gr.Markdown(label="Config Content", visible=False,container=True, elem_id="mkd1")

    # Update dropdown options when refresh is clicked
    refresh_button.click(lambda: gr.update(choices=list_config_files()), outputs=config_dropdown)
    
    # Load config only when the button is pressed
    load_button.click(load_config, inputs=config_dropdown, outputs=[config_output, accordion])

if __name__ == "__main__":
    demo.launch()
