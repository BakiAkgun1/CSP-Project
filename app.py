import numpy as np
import random
import time
import gradio as gr # type: ignore
import matplotlib.pyplot as plt
import concurrent.futures
import copy

class CSPAgent:
    """Base agent class for solving Cutting Stock Problem"""
    def __init__(self, stock_length, piece_lengths, piece_demands):
        self.stock_length = stock_length
        self.piece_lengths = piece_lengths
        self.piece_demands = piece_demands
        self.best_solution = None
        self.best_waste = float('inf')
        self.algorithm_name = "Base"
    
    def solve(self):
        # Implement in subclasses
        pass
    
    def evaluate(self, solution):
        """Calculate waste for a given solution"""
        total_waste = 0
        demands_left = self.piece_demands.copy()
        
        for pattern in solution:
            used_length = sum(pattern[i] * self.piece_lengths[i] for i in range(len(self.piece_lengths)))
            waste = self.stock_length - used_length
            total_waste += waste if waste >= 0 else float('inf')
            
            # Update remaining demands
            for i, count in enumerate(pattern):
                demands_left[i] -= count
        
        # Check if all demands are satisfied
        if any(d > 0 for d in demands_left):
            return float('inf')
            
        return total_waste
    
    def initial_solution(self):
        """Generate an initial solution using first-fit decreasing"""
        demands_left = self.piece_demands.copy()
        solution = []
        
        while any(d > 0 for d in demands_left):
            pattern = [0] * len(self.piece_lengths)
            length_left = self.stock_length
            
            # Try to fit pieces in each pattern
            for i in range(len(self.piece_lengths)):
                while demands_left[i] > 0 and length_left >= self.piece_lengths[i]:
                    pattern[i] += 1
                    demands_left[i] -= 1
                    length_left -= self.piece_lengths[i]
            
            solution.append(pattern)
        
        return solution
    
    def share_solution(self):
        return self.best_solution, self.best_waste
    
    def update_from_shared(self, solution, waste):
        if waste < self.best_waste:
            self.best_solution = copy.deepcopy(solution)
            self.best_waste = waste
            return True
        return False

class SimulatedAnnealingAgent(CSPAgent):
    def __init__(self, stock_length, piece_lengths, piece_demands, initial_temp=1000, cooling_rate=0.95, iterations=1000):
        super().__init__(stock_length, piece_lengths, piece_demands)
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.iterations = iterations
        self.algorithm_name = "Simulated Annealing"
    
    def solve(self):
        # Generate initial solution
        current_solution = self.initial_solution()
        current_waste = self.evaluate(current_solution)
        
        self.best_solution = copy.deepcopy(current_solution)
        self.best_waste = current_waste
        
        temperature = self.initial_temp
        
        # Simulated annealing main loop
        for _ in range(self.iterations):
            # Generate neighbor
            neighbor = self.get_neighbor(current_solution)
            neighbor_waste = self.evaluate(neighbor)
            
            # Accept or reject neighbor
            if neighbor_waste < current_waste or random.random() < np.exp(-(neighbor_waste - current_waste) / temperature):
                current_solution = copy.deepcopy(neighbor)
                current_waste = neighbor_waste
                
                # Update best solution
                if current_waste < self.best_waste:
                    self.best_solution = copy.deepcopy(current_solution)
                    self.best_waste = current_waste
            
            # Cooling
            temperature *= self.cooling_rate
        
        return self.best_solution, self.best_waste
    
    def get_neighbor(self, solution):
        """Generate a neighbor solution by modifying the current one"""
        neighbor = copy.deepcopy(solution)
        
        # Choose a random operation
        if len(neighbor) >= 2 and random.random() < 0.5:
            # Move a piece between patterns
            pattern1, pattern2 = random.sample(range(len(neighbor)), 2)
            piece_type = random.randrange(len(self.piece_lengths))
            
            if neighbor[pattern1][piece_type] > 0:
                # Check if there's room in pattern2
                used_length = sum(neighbor[pattern2][i] * self.piece_lengths[i] for i in range(len(self.piece_lengths)))
                if used_length + self.piece_lengths[piece_type] <= self.stock_length:
                    neighbor[pattern1][piece_type] -= 1
                    neighbor[pattern2][piece_type] += 1
        else:
            # Redistribute pieces within a pattern
            pattern_idx = random.randrange(len(neighbor))
            pattern = neighbor[pattern_idx]
            
            piece1, piece2 = random.sample(range(len(self.piece_lengths)), 2)
            
            if pattern[piece1] > 0:
                pattern[piece1] -= 1
                freed_space = self.piece_lengths[piece1]
                
                # Add as many of piece2 as possible
                while freed_space >= self.piece_lengths[piece2]:
                    pattern[piece2] += 1
                    freed_space -= self.piece_lengths[piece2]
        
        return neighbor

class HillClimbingAgent(CSPAgent):
    def __init__(self, stock_length, piece_lengths, piece_demands, iterations=1000):
        super().__init__(stock_length, piece_lengths, piece_demands)
        self.iterations = iterations
        self.algorithm_name = "Hill Climbing"
    
    def solve(self):
        # Generate initial solution
        current_solution = self.initial_solution()
        current_waste = self.evaluate(current_solution)
        
        self.best_solution = copy.deepcopy(current_solution)
        self.best_waste = current_waste
        
        # Hill climbing main loop
        for i in range(self.iterations):
            # Generate multiple neighbors and select the best
            best_neighbor = None
            best_neighbor_waste = float('inf')
            
            for _ in range(5):  # Try 5 neighbors
                neighbor = self.get_neighbor(current_solution)
                neighbor_waste = self.evaluate(neighbor)
                
                if neighbor_waste < best_neighbor_waste:
                    best_neighbor = neighbor
                    best_neighbor_waste = neighbor_waste
            
            # If no improvement, try random restart
            if best_neighbor_waste >= current_waste:
                if i < self.iterations - 10:
                    current_solution = self.initial_solution()
                    current_waste = self.evaluate(current_solution)
                else:
                    break
            else:
                # Move to the best neighbor
                current_solution = best_neighbor
                current_waste = best_neighbor_waste
                
                # Update best solution
                if current_waste < self.best_waste:
                    self.best_solution = copy.deepcopy(current_solution)
                    self.best_waste = current_waste
        
        return self.best_solution, self.best_waste
    
    def get_neighbor(self, solution):
        """Similar to SA's neighbor generation"""
        neighbor = copy.deepcopy(solution)
        
        # Choose a random operation (same as SA for simplicity)
        if len(neighbor) >= 2 and random.random() < 0.5:
            # Move a piece between patterns
            pattern1, pattern2 = random.sample(range(len(neighbor)), 2)
            piece_type = random.randrange(len(self.piece_lengths))
            
            if neighbor[pattern1][piece_type] > 0:
                used_length = sum(neighbor[pattern2][i] * self.piece_lengths[i] for i in range(len(self.piece_lengths)))
                if used_length + self.piece_lengths[piece_type] <= self.stock_length:
                    neighbor[pattern1][piece_type] -= 1
                    neighbor[pattern2][piece_type] += 1
        else:
            # Redistribute pieces within a pattern
            pattern_idx = random.randrange(len(neighbor))
            pattern = neighbor[pattern_idx]
            
            piece1, piece2 = random.sample(range(len(self.piece_lengths)), 2)
            
            if pattern[piece1] > 0:
                pattern[piece1] -= 1
                freed_space = self.piece_lengths[piece1]
                
                while freed_space >= self.piece_lengths[piece2]:
                    pattern[piece2] += 1
                    freed_space -= self.piece_lengths[piece2]
        
        return neighbor

class GeneticAlgorithmAgent(CSPAgent):
    def __init__(self, stock_length, piece_lengths, piece_demands, pop_size=30, generations=50, mutation_rate=0.2):
        super().__init__(stock_length, piece_lengths, piece_demands)
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.algorithm_name = "Genetic Algorithm"
    
    def solve(self):
        # Generate initial population
        population = [self.initial_solution() for _ in range(self.pop_size)]
        
        for _ in range(self.generations):
            # Evaluate fitness
            fitnesses = [1/(1 + self.evaluate(sol)) for sol in population]
            
            # Check for new best solution
            best_idx = fitnesses.index(max(fitnesses))
            current_waste = self.evaluate(population[best_idx])
            
            if current_waste < self.best_waste:
                self.best_solution = copy.deepcopy(population[best_idx])
                self.best_waste = current_waste
            
            # Selection
            parents = self.selection(population, fitnesses)
            
            # Create new population
            new_population = []
            
            while len(new_population) < self.pop_size:
                # Select two parents
                parent1, parent2 = random.sample(parents, 2)
                
                # Crossover
                child = self.crossover(parent1, parent2)
                
                # Mutation
                if random.random() < self.mutation_rate:
                    child = self.mutate(child)
                
                new_population.append(child)
            
            population = new_population
        
        return self.best_solution, self.best_waste
    
    def selection(self, population, fitnesses):
        """Tournament selection"""
        parents = []
        
        for _ in range(self.pop_size):
            indices = random.sample(range(len(population)), min(3, len(population)))
            tournament_fitnesses = [fitnesses[i] for i in indices]
            winner_idx = indices[tournament_fitnesses.index(max(tournament_fitnesses))]
            parents.append(population[winner_idx])
        
        return parents
    
    def crossover(self, parent1, parent2):
        """Single point crossover"""
        if not parent1 or not parent2:
            return self.initial_solution()
        
        # Take patterns from both parents
        crossover_point = random.randint(0, min(len(parent1), len(parent2)))
        child = parent1[:crossover_point] + parent2[crossover_point:]
        
        # Ensure all demands are met
        demands_left = self.piece_demands.copy()
        for pattern in child:
            for i, count in enumerate(pattern):
                demands_left[i] -= count
        
        # Add more patterns if needed
        while any(d > 0 for d in demands_left):
            pattern = [0] * len(self.piece_lengths)
            length_left = self.stock_length
            
            for i in range(len(self.piece_lengths)):
                while demands_left[i] > 0 and length_left >= self.piece_lengths[i]:
                    pattern[i] += 1
                    demands_left[i] -= 1
                    length_left -= self.piece_lengths[i]
            
            child.append(pattern)
        
        return child
    
    def mutate(self, solution):
        """Simple mutation"""
        mutated = copy.deepcopy(solution)
        
        if len(mutated) >= 2:
            # Move a piece between patterns
            pattern1, pattern2 = random.sample(range(len(mutated)), 2)
            piece_type = random.randrange(len(self.piece_lengths))
            
            if mutated[pattern1][piece_type] > 0:
                used_length = sum(mutated[pattern2][i] * self.piece_lengths[i] for i in range(len(self.piece_lengths)))
                if used_length + self.piece_lengths[piece_type] <= self.stock_length:
                    mutated[pattern1][piece_type] -= 1
                    mutated[pattern2][piece_type] += 1
        
        return mutated

class MultiAgentSystem:
    """Multi-agent system for solving the Cutting Stock Problem"""
    def __init__(self, stock_length, piece_lengths, piece_demands, num_agents=3, strategy="collaborative", algorithms=["sa", "hc", "ga"]):
        self.stock_length = stock_length
        self.piece_lengths = piece_lengths
        self.piece_demands = piece_demands
        self.num_agents = num_agents
        self.strategy = strategy  # "collaborative" or "hyper"
        self.algorithms = algorithms[:num_agents] if strategy == "hyper" else [algorithms[0]] * num_agents
        self.agents = []
        self.best_solution = None
        self.best_waste = float('inf')
        self.results = {}
        
        # Create agents
        self.create_agents()
    
    def create_agents(self):
        """Create agents based on strategy and algorithms"""
        self.agents = []
        
        for i, algo in enumerate(self.algorithms):
            if algo == "sa":
                # Different parameters for collaborative agents
                temp = 1000/(i+1) if self.strategy == "collaborative" else 1000
                cooling = 0.95-0.05*i/self.num_agents if self.strategy == "collaborative" else 0.95
                self.agents.append(SimulatedAnnealingAgent(
                    self.stock_length, self.piece_lengths, self.piece_demands,
                    initial_temp=temp, cooling_rate=cooling
                ))
            elif algo == "hc":
                iterations = 1000 + 200*i if self.strategy == "collaborative" else 1000
                self.agents.append(HillClimbingAgent(
                    self.stock_length, self.piece_lengths, self.piece_demands,
                    iterations=iterations
                ))
            elif algo == "ga":
                pop_size = 30 + 5*i if self.strategy == "collaborative" else 30
                generations = 50 + 10*i if self.strategy == "collaborative" else 50
                self.agents.append(GeneticAlgorithmAgent(
                    self.stock_length, self.piece_lengths, self.piece_demands,
                    pop_size=pop_size, generations=generations
                ))
    
    def solve(self):
        """Solve using the multi-agent strategy"""
        start_time = time.time()
        
        # Initialize results tracking
        self.results = {
            "strategy": self.strategy,
            "algorithms": [agent.algorithm_name for agent in self.agents],
            "agent_wastes": [],
            "agent_times": []
        }
        
        # Run agents concurrently
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_agent = {executor.submit(agent.solve): i for i, agent in enumerate(self.agents)}
            
            for future in concurrent.futures.as_completed(future_to_agent):
                agent_idx = future_to_agent[future]
                solution, waste = future.result()
                
                agent_time = time.time() - start_time
                self.results["agent_wastes"].append(waste)
                self.results["agent_times"].append(agent_time)
                
                # Update best solution
                if waste < self.best_waste:
                    self.best_solution = solution
                    self.best_waste = waste
        
        # If collaborative, share solutions among agents
        if self.strategy == "collaborative":
            for i, agent_i in enumerate(self.agents):
                solution_i, waste_i = agent_i.share_solution()
                
                for j, agent_j in enumerate(self.agents):
                    if i != j:
                        agent_j.update_from_shared(solution_i, waste_i)
            
            # Update best solution after collaboration
            for agent in self.agents:
                solution, waste = agent.share_solution()
                if waste < self.best_waste:
                    self.best_solution = solution
                    self.best_waste = waste
        
        # Calculate total time
        self.results["total_time"] = time.time() - start_time
        self.results["best_waste"] = self.best_waste
        
        # Determine best algorithm
        best_agent_idx = self.results["agent_wastes"].index(min(self.results["agent_wastes"]))
        self.results["best_algorithm"] = self.agents[best_agent_idx].algorithm_name
        
        return self.best_solution, self.best_waste, self.results
    
    def visualize_solution(self):
        """Visualize the cutting patterns"""
        if not self.best_solution:
            return plt.figure(figsize=(10, 6))
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Plot waste by agent
        ax1.bar(range(len(self.agents)), self.results["agent_wastes"])
        ax1.set_xticks(range(len(self.agents)))
        ax1.set_xticklabels([f"Agent {i+1}: {agent.algorithm_name}" for i, agent in enumerate(self.agents)])
        ax1.set_ylabel("Waste")
        ax1.set_title(f"Multi-Agent CSP Solution ({self.strategy} strategy)")
        plt.setp(ax1.get_xticklabels(), rotation=45, ha="right")
        
        # Plot cutting patterns
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.piece_lengths)))
        
        for i, pattern in enumerate(self.best_solution):
            y_pos = i
            x_pos = 0
            
            for piece_idx, count in enumerate(pattern):
                for _ in range(count):
                    length = self.piece_lengths[piece_idx]
                    ax2.add_patch(plt.Rectangle((x_pos, y_pos-0.4), length, 0.8, 
                                             color=colors[piece_idx], alpha=0.7))
                    if length > 5:
                        ax2.text(x_pos + length/2, y_pos, f"{length}", 
                               ha='center', va='center', fontsize=8)
                    x_pos += length
            
            # Add waste
            waste = self.stock_length - x_pos
            if waste > 0:
                ax2.add_patch(plt.Rectangle((x_pos, y_pos-0.4), waste, 0.8, 
                                         color='white', alpha=0.3, hatch='//'))
        
        ax2.set_xlim(0, self.stock_length)
        ax2.set_ylim(-0.5, len(self.best_solution) - 0.5)
        ax2.set_yticks(range(len(self.best_solution)))
        ax2.set_yticklabels([f"Pattern {i+1}" for i in range(len(self.best_solution))])
        ax2.set_title("Best Cutting Pattern")
        
        # Add legend
        legend_elements = [plt.Rectangle((0, 0), 1, 1, color=colors[i], alpha=0.7, 
                                       label=f"Piece {i+1}: {length}") 
                          for i, length in enumerate(self.piece_lengths)]
        legend_elements.append(plt.Rectangle((0, 0), 1, 1, color='white', alpha=0.3, 
                                           hatch='//', label='Waste'))
        ax2.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        return fig

# Gradio Interface
def solve_csp(stock_length, piece_lengths, piece_demands, num_agents, strategy, algorithms):
    """Main function for the Gradio interface"""
    # Parse inputs
    stock_length = int(stock_length)
    piece_lengths = [int(x.strip()) for x in piece_lengths.split(',')]
    piece_demands = [int(x.strip()) for x in piece_demands.split(',')]
    num_agents = int(num_agents)
    
    # Validate inputs
    if len(piece_lengths) != len(piece_demands):
        return None, "Error: Number of piece lengths must match number of demands"
        
    # Create and run multi-agent system
    mas = MultiAgentSystem(
        stock_length=stock_length,
        piece_lengths=piece_lengths,
        piece_demands=piece_demands,
        num_agents=num_agents,
        strategy=strategy,
        algorithms=algorithms
    )
    
    _, waste, results = mas.solve()
    
    # Generate visualization
    fig = mas.visualize_solution()
    
    # Generate report
    report = f"""
    ## Cutting Stock Problem Solution Report
    
    ### Problem Settings
    - Stock Length: {stock_length}
    - Piece Lengths: {piece_lengths}
    - Piece Demands: {piece_demands}
    
    ### Multi-Agent Configuration
    - Strategy: {strategy}
    - Number of Agents: {num_agents}
    - Algorithms: {[agent.algorithm_name for agent in mas.agents]}
    
    ### Results
    - Best Waste: {waste}
    - Best Algorithm: {results['best_algorithm']}
    - Total Time: {results['total_time']:.2f} seconds
    
    ### Agent Performance
    """
    
    for i, agent in enumerate(mas.agents):
        report += f"- Agent {i+1} ({agent.algorithm_name}): Waste = {results['agent_wastes'][i]}, Time = {results['agent_times'][i]:.2f}s\n"
    
    return fig, report

# Create the Gradio interface
def create_interface():
    with gr.Blocks(title="Cutting Stock Problem Solver") as interface:
        gr.Markdown("# Cutting Stock Problem Multi-Agent Solver")
        
        with gr.Row():
            with gr.Column():
                stock_length = gr.Number(label="Stock Length", value=100)
                piece_lengths = gr.Textbox(label="Piece Lengths (comma-separated)", value="10, 15, 20")
                piece_demands = gr.Textbox(label="Piece Demands (comma-separated)", value="5, 3, 2")
                
                num_agents = gr.Slider(minimum=1, maximum=5, step=1, label="Number of Agents", value=3)
                strategy = gr.Radio(["collaborative", "hyper"], label="Strategy", value="collaborative")
                algorithms = gr.CheckboxGroup(
                    ["sa", "hc", "ga"], 
                    label="Algorithms (select at least one)",
                    value=["sa", "hc", "ga"]
                )
                
                submit_btn = gr.Button("Solve")
                
            with gr.Column():
                output_plot = gr.Plot(label="Cutting Pattern Visualization")
                output_report = gr.Markdown(label="Results Report")
        
        submit_btn.click(
            solve_csp,
            inputs=[stock_length, piece_lengths, piece_demands, num_agents, strategy, algorithms],
            outputs=[output_plot, output_report]
        )
    
    return interface

# paste.py dosyasının sonuna ekleyin
def run_test_scenarios():
    """Run test scenarios and display results"""
    test_scenarios = [
        # Scenario 1: Simple case
        {
            "name": "Simple Test",
            "stock_length": 100,
            "piece_lengths": [10, 20, 30],
            "piece_demands": [5, 3, 2],
            "num_agents": 3,
            "strategy": "collaborative",
            "algorithms": ["sa", "hc", "ga"]
        },
        # Scenario 2: Hyper strategy
        {
            "name": "Hyper Strategy",
            "stock_length": 200,
            "piece_lengths": [50, 60, 70, 80],
            "piece_demands": [3, 2, 4, 1],
            "num_agents": 3,
            "strategy": "hyper",
            "algorithms": ["sa", "hc", "ga"]
        }
    ]
    
    results = []
    
    for scenario in test_scenarios:
        print(f"\nRunning: {scenario['name']}")
        
        # Create and run MultiAgentSystem
        mas = MultiAgentSystem(
            stock_length=scenario['stock_length'],
            piece_lengths=scenario['piece_lengths'],
            piece_demands=scenario['piece_demands'],
            num_agents=scenario['num_agents'],
            strategy=scenario['strategy'],
            algorithms=scenario['algorithms']
        )
        
        # Find solution
        solution, waste, scenario_results = mas.solve()
        
        # Print results
        print(f"Best Waste: {waste}")
        print(f"Best Algorithm: {scenario_results['best_algorithm']}")
        print(f"Total Time: {scenario_results['total_time']:.2f} seconds")
        
        # Save visualization
        fig = mas.visualize_solution()
        fig.savefig(f"result_{scenario['name'].replace(' ', '_')}.png")
        
        results.append({
            "scenario": scenario['name'],
            "waste": waste,
            "best_algorithm": scenario_results['best_algorithm'],
            "time": scenario_results['total_time']
        })
    
    # Summary table
    print("\n\nAll Test Results:")
    print("=" * 60)
    print(f"{'Scenario':<15} {'Waste':<8} {'Best Algorithm':<20} {'Time (s)':<8}")
    print("-" * 60)
    for result in results:
        print(f"{result['scenario']:<15} {result['waste']:<8} {result['best_algorithm']:<20} {result['time']:<8.2f}")
    
    return results

# Main application logic
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        # Run test scenarios
        run_test_scenarios()
    else:
        # Launch Gradio interface
        app = create_interface()
        app.launch()
        
# # Launch the app when run directly
# if __name__ == "__main__":
#     app = create_interface()
#     app.launch()