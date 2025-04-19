import gradio as gr
import json
import sympy as sp
import re
import time


# Function to hide the status message after a delay
def hide_status_after_delay():
    time.sleep(3)
    return gr.update(value="",visible=False)
    


def hide_status():
    return gr.update(visible=False)

def handle_dropdown(choice):
    if choice == 1:
        return gr.update(visible=True), gr.update(visible=False)
    elif choice == 2:
        return gr.update(visible=False), gr.update(visible=True)
    else:
        return gr.update(visible=False), gr.update(visible=False)
    
def extract_variables(func):
    expr = sp.sympify(func)
    variables = expr.free_symbols
    def sort_key(symbol):
        match = re.match(r"x(\d+)", str(symbol))
        return int(match.group(1)) if match else float('inf')
    sorted_vars = sorted(variables, key=sort_key)
    return sorted_vars


    
def assemble_json(name, exp_type, eval_func, restrictions, ranges, generations, popsize, safety, algs_config, objective):
    exp_type_label = ["---", "Ecuacion Matematica", "Red Neuronal"][exp_type]
    if safety:
        pop_safety = 1
    else:
        pop_safety = 0

    if objective == "Maximizar":
        obj = "MAX"
    else:
        obj = "MIN"

    if exp_type_label == "Ecuacion Matematica":
        task_config = {
            "eq": eval_func,
            "obj": obj,
            "ranges": {
                f"x{i}": {
                    "min": int(pair.split(",")[0]),"max": int(pair.split(",")[1])
                }
                for i, pair in enumerate(ranges)
            },
            "restrictions": restrictions,
            "pop_gen_safety": pop_safety
        }
    else:
        task_config = {
            "val": "Placeholder"
        }
    result = {
        "name": name,
        "type": ("math" if exp_type_label == "Ecuacion Matematica" else "nn"),
        "generations": int(generations),
        "pop_size": int(popsize),
        "task_config": task_config,
        "alg_config":
        {
        "genetic": {
            "mutation_p": float(algs_config[0][0])
        },
        "diff_ev": {
            "mutation_f": float(algs_config[1][0]),
            "recomb_const": float(algs_config[1][1])
        },
        "particle": {
            "w": float(algs_config[2][0]),
            "c1": float(algs_config[2][1]),
            "c2": float(algs_config[2][2])
        }
        }
    }
    
    jsonObject = json.dumps(result, indent=2)
    with open(f"./saved_exp/{name}.json", "w") as outfile:
        outfile.write(jsonObject)
    


with gr.Blocks() as demo:
    status_msg = gr.Markdown("", visible=False)
    
    exp_name = gr.Textbox(label="Nombre del experimento")
    gen_num = gr.Textbox(label= "Número de generaciones")
    pop_size = gr.Textbox(label="Cantidad de individuos")
    exp_type = gr.Dropdown(
        choices=["---","Ecuación Matemática", "Red Neuronal"],
        value=None,
        label="Tipo de experimento",
        interactive=True,
        type='index'
    )
    
    with gr.Row() as algorithms:
            
            with gr.Group() as Genetico:
               gr.Markdown("<center>Configuración de algoritmo genético</center>")
               mutP = gr.Textbox(label="Probabilidad de mutación", placeholder="Valores entre 0 y 1")

            
            with gr.Group() as Evolutivo:
                gr.Markdown("<center>Configuración de algoritmo evolucion diferencial</center>")
                mutF = gr.Textbox(label="Factor de mutación", placeholder= "Valores entre 0 y 1")
                recombC = gr.Textbox(label="Constante de recombinacion", placeholder= "Valores entre 0 y 1")

            
            with gr.Group() as Particulas:
                gr.Markdown("<center>Configuración de algoritmo enjambre de particulas</center>")
                w = gr.Textbox(label="Inercia", placeholder= "Valores entre 0 y 1")
                c1 = gr.Textbox(label="Coeficiente cognitivo", placeholder= "Valores entre 0 y 1")
                c2 = gr.Textbox(label="Coeficiente social", placeholder= "Valores entre 0 y 1")
    
    algs_config = gr.State([])
    restriction_values = gr.State([])
    range_values = gr.State([])
    with gr.Column(visible=False) as menuMath:
        eval_func = gr.Textbox(label="Ecuación a evaluar", placeholder="Solo usar xn para variables ej. x0 + x1 + x2  ")
        objective = gr.Dropdown(label="Se busca _ la función", choices=["Maximizar","Minimizar"], interactive=True, value=None,)
        restrict_num = gr.Textbox(label = "Número de restricciones", value="0")

        
        

        @gr.render(inputs=[restrict_num, eval_func])
        def set_restrict(restrict_n, func):
            pop_safety = gr.Checkbox(label="Asegurar que toda la poblacion sea valida")
            restrictions = []
            if int(restrict_n) > 0:
                gr.Markdown("**Restricciones**")

            try:
                for i in range(int(restrict_n)):
                    restrictions.append(gr.Textbox(key=f"res{i}",label=f"Restricción {i+1}", placeholder="ej. x0 + x1 >= 10"))
            except (ValueError, TypeError):
                restrictions.append(gr.Markdown("⚠️ Ingrese un número válido."))

            ranges = []
            try:
                variables = extract_variables(func)
            except:
                variables = []
            
            if variables:
                gr.Markdown("**Rangos**")
            
            try:
                for n,var in enumerate(variables):
                    ranges.append(gr.Textbox(key=f"ran{n}",label=f"Rango para variable {var}", placeholder="ej. -5,10"))
            except (ValueError, TypeError):
                ranges.append(gr.Markdown("⚠️ Ingrese una función valida"))

            def collect_values(*args):
                return list(args)
            
            def collect_algs(mutP_val, mutF_val, recombC_val, w_val, c1_val, c2_val):
                genetic = [mutP_val]
                differential = [mutF_val, recombC_val]
                particle = [w_val, c1_val, c2_val]
                return [genetic, differential, particle]
            

            
            save_btn.click(collect_values, inputs=restrictions, outputs=restriction_values) \
                .then(collect_values, inputs=ranges, outputs=range_values) \
                .then(collect_algs, inputs=[mutP, mutF, recombC, w, c1, c2], outputs=algs_config)\
                .then(fn=assemble_json,
                        inputs=[exp_name, exp_type, eval_func, restriction_values, range_values, gen_num, pop_size, pop_safety, algs_config, objective],
                        outputs=None)\
                .then(
                    lambda: (
                        gr.update(value="✅ Guardado con éxito", visible=True),
                        gr.update(value=""),  # Clear exp_name
                        gr.update(value=""),  # Clear gen_num
                        gr.update(value=""),  # Clear pop_size
                        gr.update(value=None),  # Reset exp_type dropdown
                        gr.update(value=""),  # Clear eval_func
                        gr.update(value=None),  # Reset objective dropdown
                        gr.update(value="0"),  # Reset restrict_num
                        gr.update(value=""),  # Clear mutP
                        gr.update(value=""),  # Clear mutF
                        gr.update(value=""),  # Clear recombC
                        gr.update(value=""),  # Clear w
                        gr.update(value=""),  # Clear c1
                        gr.update(value=""),  # Clear c2
                    ),
                    inputs=[],
                    outputs=[
                        status_msg,
                        exp_name,
                        gen_num,
                        pop_size,
                        exp_type,
                        eval_func,
                        objective,
                        restrict_num,
                        mutP,
                        mutF,
                        recombC,
                        w,
                        c1,
                        c2,
                    ]
                )\
                .then(hide_status_after_delay,outputs=status_msg)
                

            return restrictions, ranges
            
    
    
    with gr.Column(visible=False) as menuNN:
        plc2 = gr.Textbox(value="Placeholder",interactive=False,label="Menu for Option 2")


    save_btn = gr.Button(value="Guardar experimento")
    
    exp_type.change(fn=handle_dropdown, 
                    inputs=exp_type, 
                    outputs=[menuMath, menuNN])
    


    

    
if __name__ == "__main__":
    demo.launch()