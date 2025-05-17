import gradio as gr
import sys
import os
sys.path.append(os.path.abspath("."))

import interfaces.run as run
import interfaces.new as new

with gr.Blocks(css="#refbtn button {height: 2.4cm !important;}", title="Optimizer") as demo:
    with gr.Tabs():
        with gr.Tab("Open"):
            run.demo.render()
        with gr.Tab("Create"):
            new.demo.render()

if __name__ == "__main__":
    demo.launch()