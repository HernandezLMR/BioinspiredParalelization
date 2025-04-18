import gradio as gr


def handle_dropdown(choice):
    if choice == 1:
        return gr.update(visible=True), gr.update(visible=False)
    elif choice == 2:
        return gr.update(visible=False), gr.update(visible=True)
    else:
        return gr.update(visible=False), gr.update(visible=False)

with gr.Blocks() as demo:
    
    exp_name = gr.Textbox(label="Nombre del experimento")
    exp_type = gr.Dropdown(
        choices=["---","Ecuación Matemática", "Red Neuronal"],
        value=None,
        label="Tipo de experimento",
        interactive=True,
        type='index'
    )
    
    with gr.Column(visible=False) as menuMath:
        eval_func = gr.Textbox(label="Ecuación a evaluar")

    
    
    with gr.Column(visible=False) as menuNN:
        plc2 = gr.Textbox(value="Placeholder",interactive=False,label="Menu for Option 2")

    exp_type.change(fn=handle_dropdown, 
                    inputs=exp_type, 
                    outputs=[menuMath, menuNN])


if __name__ == "__main__":
    demo.launch()