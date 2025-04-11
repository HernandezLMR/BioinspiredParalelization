def math_extract(config):
    outs = []
    outs.append("GENERAL OPTIMIZATION INFO <br />")
    outs.append(f"Operation type: {config['type']} <br />")
    outs.append(f"Number of epochs: {config['generations']} <br />")
    outs.append(f"Initial population size: {config['pop_size']} <br />")

    outs.append("<br />SINGLE EQUATION INFO<br />")
    outs.append(f"Evaluation function: {config['task_config']['eq']} <br />")
    outs.append(f"Function is {'maximizing' if config['task_config']['obj'] == 'MAX' else 'minimizing'} value<br />")
    outs.append("Restrictions:  <br />")
    
    for key, val in config['task_config']['restrictions'].items():
        outs.append(f"No.{key} : {val}<br />") 
    
    outs.append("Ranges: <br />")
    
    for key, val in config['task_config']['ranges'].items():
        outs.append(f"{key} : {val}<br />") 
    
    outs.append(f"Generating initial population {'with' if config['task_config']['pop_gen_safety'] == 1 else 'without'} valid safety<br />")

    outs.append("<br />OPTIMIZATION ALGORITHM INFO<br />")
    outs.append("**Genetic Algorithm**<br />")
    outs.append(f"Mutation probability: {config['alg_config']['genetic']['mutation_p']}<br />")

    outs.append("**Differential Evolution**<br />")
    outs.append(f"Mutation Factor: {config['alg_config']['diff_ev']['mutation_f']}<br />")
    outs.append(f"Recombination Constant: {config['alg_config']['diff_ev']['recomb_const']}<br />")

    outs.append("**Particle Swarm**<br />")
    outs.append(f"Inertia: {config['alg_config']['particle']['w']}<br />")
    outs.append(f"Cognitive Component: {config['alg_config']['particle']['c1']}<br />")
    outs.append(f"Social Component: {config['alg_config']['particle']['c2']}")

    
    return ''.join(outs)  # Return the formatted string

'''def net_exctract(config):
    outs = []
    outs.append(f"Operation type: {config.type} <br />")
    outs.append(f"Number of epochs: {config.generations} <br />")
    outs.append(f"Initial population size: {config.pop_size} <br />")
    #Placeholder'''