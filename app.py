from flask import Flask, request, jsonify, render_template
import pulp as p

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/solve', methods=['POST'])
def solve():
    try:
        # Recibiendo datos desde el formulario
        num_supply_nodes = int(request.form.get('num_supply_nodes'))
        num_intermediate_nodes = int(request.form.get('num_intermediate_nodes'))
        num_demand_nodes = int(request.form.get('num_demand_nodes'))
        
        supply = [float(request.form.get(f'supply_{i}')) for i in range(1, num_supply_nodes + 1)]
        demand = [float(request.form.get(f'demand_{i}')) for i in range(1, num_demand_nodes + 1)]
        
        # Costos de suministro -> intermedios
        supply_to_intermediate_costs = {}
        for i in range(1, num_supply_nodes + 1):
            for j in range(num_supply_nodes + 1, num_supply_nodes + num_intermediate_nodes + 1):
                cost = float(request.form.get(f'cost_{i}_{j}', 0))
                supply_to_intermediate_costs[(i, j)] = cost
        
        # Costos de intermedios -> demanda
        intermediate_to_demand_costs = {}
        for i in range(num_supply_nodes + 1, num_supply_nodes + num_intermediate_nodes + 1):
            for j in range(num_supply_nodes + num_intermediate_nodes + 1, num_supply_nodes + num_intermediate_nodes + num_demand_nodes + 1):
                cost = float(request.form.get(f'cost_{i}_{j}', 0))
                intermediate_to_demand_costs[(i, j)] = cost
        
        # Crear el modelo de optimización
        model = p.LpProblem("Problema_de_Transbordo", p.LpMinimize)
        
        # Variables de decisión para los flujos
        flows = p.LpVariable.dicts(
            "flow",
            list(supply_to_intermediate_costs.keys()) + list(intermediate_to_demand_costs.keys()),
            lowBound=0,
            cat='Continuous'
        )
        
        # Definir la función objetivo
        model += p.lpSum(
            [flows[key] * supply_to_intermediate_costs[key] for key in supply_to_intermediate_costs] +
            [flows[key] * intermediate_to_demand_costs[key] for key in intermediate_to_demand_costs]
        ), "Costo_Total"
        
        # Restricciones de suministro
        for i in range(1, num_supply_nodes + 1):
            model += p.lpSum([flows[(i, j)] for j in range(num_supply_nodes + 1, num_supply_nodes + num_intermediate_nodes + 1)]) == supply[i - 1], f"Suministro_{i}"
        
        # Restricciones de demanda
        for j in range(num_supply_nodes + num_intermediate_nodes + 1, num_supply_nodes + num_intermediate_nodes + num_demand_nodes + 1):
            model += p.lpSum([flows[(i, j)] for i in range(num_supply_nodes + 1, num_supply_nodes + num_intermediate_nodes + 1)]) == demand[j - num_supply_nodes - num_intermediate_nodes - 1], f"Demanda_{j}"
        
        # Restricciones de balance en nodos intermedios
        for j in range(num_supply_nodes + 1, num_supply_nodes + num_intermediate_nodes + 1):
            model += p.lpSum([flows[(i, j)] for i in range(1, num_supply_nodes + 1)]) == \
                     p.lpSum([flows[(j, k)] for k in range(num_supply_nodes + num_intermediate_nodes + 1, num_supply_nodes + num_intermediate_nodes + num_demand_nodes + 1)]), f"Balance_{j}"
        
        # Resolver el modelo
        model.solve()
        
        # Verificar el estado de la solución
        status = p.LpStatus[model.status]
        
        # Flujos óptimos y costo total
        flows_result = {f"Del nodo {key[0]} al nodo {key[1]}": flows[key].varValue for key in flows}
        total_cost = p.value(model.objective)
        
        return jsonify({
            "status": status,
            "flows": flows_result,
            "total_cost": total_cost
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
