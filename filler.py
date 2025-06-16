import os
import json

gen={'1':5,'2':10,'3':15,'4':5,'5':5,'6':10,'7':10,'8':15,'9':15}
pop={'1':5,'2':10,'3':15,'4':10,'5':15,'6':5,'7':15,'8':5,'9':10}
BASE_DIR = 'exp_results'

doc = open(os.path.join(BASE_DIR, 'Ec.1.2.4.2.1.json'))
key = '.'
result = 'Ec.1.2.4.2.1.json'.split(key)
result = result[1:6]
print(result)




for file_name in os.listdir(BASE_DIR):
    if file_name.endswith('.json'):
        print(f"Processing {file_name}")
        
        filepath = os.path.join(BASE_DIR, file_name)
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        combination = file_name.split(key)
        combination = combination[1:6]

        generations = gen[combination[2]]
        pop_size = pop[combination[2]]

        data['ec'] = combination[0]
        data['gen'] = generations
        data['pop_size'] = pop_size
        
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4)

print("Done.")
