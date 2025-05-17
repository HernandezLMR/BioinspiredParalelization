import gradio as gr
import os
import json
from algorithms.controller import run_algs

'''It would be better to split these pages into separate subpages but I think keeping them as big saucy bois is funnier'''

CONFIG_DIR = "./saved_exp"

def list_config_files():
    return [f for f in os.listdir(CONFIG_DIR) if f.endswith('.json')]

def format_ranges_for_display(ranges: dict) -> str:
    return "\n".join(f"{var}: ({bounds['min']}, {bounds['max']})" for var, bounds in ranges.items())

def format_restrictions_for_display(restrictions: list[str]) -> str:
    return "\n".join(restrictions)

def format_epoch_results(results_log):
    lines = []
    for epoch_data in results_log.get("parallel", {}).get("epochs", []):
        lines.append(f"Epoch {epoch_data['epoch']}")
        lines.append(f"Best individual: {epoch_data['best_individual']}")
        lines.append(f"Best value: {epoch_data['best_value']}")
        lines.append("")  # Line skip
    return "\n".join(lines)


def load_config(filename):
    if not filename:
        return "No file selected."
    filepath = os.path.join(CONFIG_DIR, filename)
    with open(filepath, 'r') as f:
        data = json.load(f)

    return  gr.Textbox(value=data["name"], visible=True),\
            gr.Textbox(value=data["generations"], visible=True),\
            gr.Textbox(value=data["pop_size"], visible=True),\
            gr.Textbox(value=data["task_config"]["obj"], visible=True),\
            gr.Textbox(value=data["task_config"]["eq"], visible=True),\
            gr.Textbox(value=format_ranges_for_display(data["task_config"]["ranges"]), visible=True),\
            gr.Textbox(value=format_restrictions_for_display(data["task_config"]["restrictions"]), visible=True),\
            gr.Textbox(value=data["alg_config"]["genetic"]["mutation_p"], visible=True),\
            gr.Textbox(value=data["alg_config"]["diff_ev"]["mutation_f"], visible=True),\
            gr.Textbox(value=data["alg_config"]["diff_ev"]["recomb_const"], visible=True),\
            gr.Textbox(value=data["alg_config"]["particle"]["w"], visible=True),\
            gr.Textbox(value=data["alg_config"]["particle"]["c1"], visible=True),\
            gr.Textbox(value=data["alg_config"]["particle"]["c2"], visible=True),\
            gr.Accordion(visible=True),\
            gr.Tabs(visible=True)

def run_alg(alg_type, config_path, n_processors, n_repeats):
    res = run_algs(alg_type, config_path, int(n_processors), int(n_repeats))
    try:
        val_stop = str(res["parallel"]["early_stopping_epoch"])
        vis = True
    except:
        val_stop = ""
        vis = False
    alg_dict = {"0": "Genetic", "1": "Differential Evolution", "2":"Particle"}

    return gr.Textbox(value = res["sequential"]["best_value"]),\
           gr.Textbox(value = res["sequential"]["best_individual"]),\
           gr.Textbox(value = res["sequential"]["time_taken"]),\
           gr.Textbox(value = val_stop, visible= vis),\
           gr.Textbox(value = res["parallel"]["epochs"][-1]["best_value"]),\
           gr.Textbox(value = res["parallel"]["epochs"][-1]["best_individual"]),\
           gr.Textbox(value = res["parallel"]["time_taken"]),\
           gr.Textbox(value = format_epoch_results(res)),\
           gr.Textbox(value = res["parallel"]["speed_up"]),\
           gr.Textbox(value = res["parallel"]["efficiency"]),\
           gr.Group(visible = True),\
           gr.Markdown(value=f"<center>Using {alg_dict[str(alg_type)]} algortihm</center>")
    

with gr.Blocks() as demo:
    with gr.Row():
        config_dropdown = gr.Dropdown(choices=list_config_files(), label="Select a config file", scale=11)
        refresh_button = gr.Button("🔄", scale=1, elem_id="refbtn")

    
    load_button = gr.Button("Load Config")
    with gr.Tabs(visible=False) as tabs:
        with gr.Tab("File Info"):
            with gr.Accordion('Loaded Configuration', open=True, visible=False, elem_id="acc1") as accordion:
                name = gr.Textbox(label = "Nombre")
                with gr.Row():
                    gens = gr.Textbox(label = "Generaciones")
                    pop_size = gr.Textbox(label = "Tamaño de población")
                    obj = gr.Textbox(label = "Objetivo")
                eq = gr.Textbox(label = "Ecuación")
                with gr.Row():
                    ranges = gr.Textbox(label = "Rango de variables")
                    restrictions = gr.Textbox(label = "Restricciones")


                banner = gr.Textbox("Algorithm configuration")
                with gr.Row() as algorithms:
                    
                    with gr.Group() as Genetico:
                        gr.Markdown("<center>Configuración de algoritmo genético</center>")
                        mutP = gr.Textbox(label="Probabilidad de mutación")

                    
                    with gr.Group() as Evolutivo:
                        gr.Markdown("<center>Configuración de algoritmo evolucion diferencial</center>")
                        mutF = gr.Textbox(label="Factor de mutación")
                        recombC = gr.Textbox(label="Constante de recombinacion")

                    
                    with gr.Group() as Particulas:
                        gr.Markdown("<center>Configuración de algoritmo enjambre de particulas</center>")
                        w = gr.Textbox(label="Inercia")
                        c1 = gr.Textbox(label="Coeficiente cognitivo")
                        c2 = gr.Textbox(label="Coeficiente social")

            refresh_button.click(lambda: gr.update(choices=list_config_files()), outputs=config_dropdown)
            load_button.click(load_config, inputs=config_dropdown, outputs=[name, gens, pop_size, obj,eq, ranges, restrictions, mutP, mutF, recombC, w, c1, c2, accordion, tabs])
        with gr.Tab("Run File"):
            
            with gr.Row():
                alg_type = gr.Dropdown(["Genetic Algorithm","Differential Evolution","Particle Swarm"], type="index", scale=11, label="Algorithm Type")
                n_processors = gr.Textbox(interactive=True, label="Number of Processors",scale=4)
                n_repeats = gr.Textbox(interactive=True, label="Number of Epochs",scale=4)
                run_button = gr.Button(value="▶️", scale = 3)
            loading = gr.Markdown("⏳ Running, please wait...", visible=False)
            with gr.Group(visible=False) as res_group:
                alg_type_label = gr.Markdown()
                with gr.Group() as seq_res:
                    gr.Markdown("<center>Sequential run</center>")
                    seq_best_ind = gr.Textbox(label="Best found result")
                    seq_best_pos = gr.Textbox(label="Values for best found result")
                    seq_time = gr.Textbox(label="Time taken")
                with gr.Group() as par_res:
                    gr.Markdown("<center>Parallel run</center>")
                    early_stop = gr.Textbox(label="Stopped early at epoch:",visible=False)

                    par_best_ind = gr.Textbox(label="Best found result")
                    par_best_pos = gr.Textbox(label = "Values for best found result")
                    par_time = gr.Textbox(label="Time taken")
                    with gr.Accordion("Per Epoch Results", open=False):
                        results = gr.Textbox()
                    speed_up = gr.Textbox(label="Speed up")
                    efficiency = gr.Textbox(label="Efficiency")
            run_button.click(lambda: gr.update(visible=True), outputs=loading, show_progress=False) \
                    .then(run_alg, 
                            inputs=[alg_type, config_dropdown, n_processors, n_repeats], 
                            outputs=[
                                seq_best_ind, seq_best_pos, seq_time,
                                early_stop,
                                par_best_ind, par_best_pos, par_time,
                                results,
                                speed_up, efficiency,
                                res_group, alg_type_label
                            ]) \
                    .then(lambda: gr.update(visible=False), outputs=loading, show_progress=False)

                




if __name__ == "__main__":
    demo.launch()
