import gradio as gr


import interfaces.run as run
import interfaces.new as new

with gr.Blocks() as demo:
    with gr.Tabs():
        with gr.Tab("Config"):
            run.demo.render()
        with gr.Tab("Test"):
            new.demo.render()

if __name__ == "__main__":
    demo.launch()