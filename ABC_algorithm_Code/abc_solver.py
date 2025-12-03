import numpy as np
import random
from typing import List, Tuple
import pickle
import json
from datetime import datetime
from solution import MDMSMCVRPSolution
import time


class MDMSMCVRP_ABC:
    def __init__(self, problem, num_vehicles_per_satellite=None, swarm_size=50, max_iterations=200,
                 employed_ratio=0.5, onlooker_ratio=0.3, limit=15, verbose=False):
        """
        Enhanced ABC solver for MDMS-MCVRP with multiple vehicles per satellite support

        Args:
            problem: Problem instance
            num_vehicles_per_satellite: Dict specifying vehicles per satellite, e.g., {'S1': 3, 'S2': 2}
            swarm_size: Population size
            max_iterations: Number of iterations
            employed_ratio: Ratio of employed bees
            onlooker_ratio: Ratio of onlooker bees
            limit: Trial limit for scout phase
            verbose: Print detailed information
        """
        self.problem = problem
        self.num_vehicles_per_satellite = num_vehicles_per_satellite or {k: 1 for k in problem.VS}
        self.swarm_size = swarm_size
        self.max_iterations = max_iterations
        self.employed_ratio = employed_ratio
        self.onlooker_ratio = onlooker_ratio
        self.limit = limit
        self.verbose = verbose

        # Core components
        self.swarm: List[MDMSMCVRPSolution] = []
        self.fitness_values: List[float] = []
        self.trials: List[int] = []
        self.best_solution: MDMSMCVRPSolution = None
        self.best_fitness = float('inf')
        self.convergence_curve: List[float] = []

        # Enhanced tracking
        self.improvements_count = 0
        self.improvement_iterations: List[int] = []
        self.best_solutions_history = []

    def validate_mandatory_satellite_visits(self, solution):
        """Validate that each primary vehicle visits all satellites"""
        for depot in self.problem.VD:
            if 'E1' not in solution.routes or depot not in solution.routes['E1']:
                return False

            route = solution.routes['E1'][depot]
            visited_satellites = set([node for node in route if node in self.problem.VS])

            if visited_satellites != set(self.problem.VS):
                return False
        return True

    def enforce_satellite_visits(self, solution):
        """Ensure each primary vehicle visits all satellites"""
        for depot in self.problem.VD:
            route = [depot]
            satellites_to_visit = list(self.problem.VS)
            random.shuffle(satellites_to_visit)
            route.extend(satellites_to_visit)
            route.append(depot)

            if 'E1' not in solution.routes:
                solution.routes['E1'] = {}
            solution.routes['E1'][depot] = route

        return solution

    def calculate_fitness_with_constraint(self, solution):
        """Calculate fitness with heavy penalty for constraint violations"""
        if not self.validate_mandatory_satellite_visits(solution):
            return float('inf')

        base_fitness = solution.calculate_fitness()

        penalty = 0
        for depot in self.problem.VD:
            route = solution.routes['E1'][depot]
            visited_satellites = set([node for node in route if node in self.problem.VS])
            missing_satellites = set(self.problem.VS) - visited_satellites

            if missing_satellites:
                penalty += 1000000 * len(missing_satellites)

        return base_fitness + penalty

    def initialize_swarm(self):
        """Initialize swarm with multiple vehicles per satellite"""
        total_vehicles = sum(self.num_vehicles_per_satellite.values())
        print(f"Initializing swarm with {self.swarm_size} solutions...")
        print(f"✅ Vehicles per satellite: {self.num_vehicles_per_satellite} (Total: {total_vehicles} vehicles)")
        print("✅ Enforcing mandatory satellite visits constraint...")

        self.swarm = []
        feasible_solutions = 0

        for i in range(self.swarm_size):
            solution = MDMSMCVRPSolution(self.problem, self.num_vehicles_per_satellite)
            solution.initialize_random()

            solution = self.enforce_satellite_visits(solution)
            solution.fitness = self.calculate_fitness_with_constraint(solution)

            self.swarm.append(solution)

            if solution.fitness != float('inf'):
                feasible_solutions += 1

        self.fitness_values = [sol.fitness for sol in self.swarm]
        self.trials = [0] * self.swarm_size

        valid_solutions = [sol for sol in self.swarm if sol.fitness != float('inf')]
        if valid_solutions:
            self.best_solution = min(valid_solutions, key=lambda x: x.fitness)
            self.best_fitness = self.best_solution.fitness
            print(f"✅ Generated {feasible_solutions}/{self.swarm_size} feasible solutions")
        else:
            print("⚠️ No feasible solutions found, creating manual solution...")
            manual_solution = self.create_manual_feasible_solution()
            self.swarm[0] = manual_solution
            self.fitness_values[0] = manual_solution.fitness
            self.best_solution = manual_solution
            self.best_fitness = manual_solution.fitness

        print(f"Initial best fitness: {self.best_fitness:.2f}")

        if self.verbose:
            print("🎯 Initial Best Routes:")
            self.print_solution_routes(self.best_solution)

    def create_manual_feasible_solution(self):
        """Create manually constructed feasible solution with multiple vehicles"""
        solution = MDMSMCVRPSolution(self.problem, self.num_vehicles_per_satellite)

        solution.routes = {'E1': {}, 'E2': {}}
        solution.flows = {'Y': {}, 'Z': {}}

        # Primary routes
        for depot in self.problem.VD:
            route = [depot] + list(self.problem.VS) + [depot]
            solution.routes['E1'][depot] = route

        # Secondary routes with multiple vehicles
        customers = list(self.problem.VC)
        random.shuffle(customers)

        all_vehicles = []
        for satellite in self.problem.VS:
            num_vehicles = self.num_vehicles_per_satellite[satellite]
            for v_idx in range(num_vehicles):
                vehicle_id = f"{satellite}_V{v_idx + 1}"
                all_vehicles.append((vehicle_id, satellite))
                solution.routes['E2'][vehicle_id] = [satellite, satellite]

        # Distribute customers across all vehicles
        customers_per_vehicle = len(customers) // len(all_vehicles)
        customer_idx = 0

        for vehicle_id, satellite in all_vehicles:
            end_idx = min(customer_idx + customers_per_vehicle, len(customers))
            route = [satellite] + customers[customer_idx:end_idx] + [satellite]
            solution.routes['E2'][vehicle_id] = route
            customer_idx = end_idx

        # Assign remaining customers
        if customer_idx < len(customers):
            last_vehicle = all_vehicles[-1][0]
            route = solution.routes['E2'][last_vehicle]
            route[-1:-1] = customers[customer_idx:]

        solution._calculate_flows()
        solution.fitness = self.calculate_fitness_with_constraint(solution)

        return solution

    def optimize(self) -> Tuple[MDMSMCVRPSolution, List[float]]:
        """Run ABC optimization with multiple vehicles support"""
        self.initialize_swarm()

        print(f"Starting optimization with {self.max_iterations} iterations...")
        print("✅ Constraint: Each primary vehicle must visit ALL satellites")
        start_time = time.time()

        for iteration in range(self.max_iterations):
            if iteration % 20 == 0:
                print(f"Iteration {iteration}: Best fitness = {self.best_fitness:.2f}")

            prev_best = self.best_fitness

            self._employed_bee_phase()
            self._onlooker_bee_phase()
            self._scout_bee_phase()

            self.convergence_curve.append(self.best_fitness)

            if self.best_fitness < prev_best:
                self.improvements_count += 1
                self.improvement_iterations.append(iteration)
                print(f"  ✅ NEW BEST at iteration {iteration}: {self.best_fitness:.2f}")

                self.best_solutions_history.append({
                    'iteration': iteration,
                    'fitness': self.best_fitness,
                    'solution': self.best_solution.copy(),
                    'timestamp': datetime.now().isoformat()
                })

                if self.verbose:
                    self.print_solution_routes(self.best_solution)

        total_time = time.time() - start_time
        print(f"\nOptimization completed in {total_time:.2f}s")
        print(f"Final best fitness: {self.best_fitness:.2f}")
        print(f"Total improvements: {self.improvements_count}")

        print(f"\n🏆 FINAL BEST SOLUTION ROUTES:")
        self.print_solution_routes(self.best_solution)

        if self.validate_mandatory_satellite_visits(self.best_solution):
            print("✅ Final solution satisfies mandatory satellite visits constraint")
        else:
            print("❌ WARNING: Final solution violates mandatory satellite visits constraint")

        return self.best_solution, self.convergence_curve

    def print_solution_routes(self, solution):
        """Print detailed routes with multiple vehicles support"""
        if solution is None:
            print("    No solution to display!")
            return

        print(f"    🚛 Primary Vehicle Routes (Depots → Satellites):")
        for depot, route in solution.routes['E1'].items():
            if len(route) > 2:
                satellites_count = len([s for s in route if s in self.problem.VS])
                route_str = ' → '.join(route)
                print(f"      Vehicle {depot}: {route_str} ({satellites_count} satellites)")

                visited_sats = set([s for s in route if s in self.problem.VS])
                if visited_sats != set(self.problem.VS):
                    missing = set(self.problem.VS) - visited_sats
                    print(f"        ❌ MISSING satellites: {missing}")
                else:
                    print(f"        ✅ All satellites visited")

        print(f"    🚐 Secondary Vehicle Routes (Satellites → Customers) - Multiple Vehicles:")
        for vehicle_id in sorted(solution.routes['E2'].keys()):
            route = solution.routes['E2'][vehicle_id]
            satellite = vehicle_id.split('_V')[0]
            if len(route) > 2:
                customers_count = len([c for c in route if c in self.problem.VC])
                route_str = ' → '.join(route)
                print(f"      {vehicle_id} (from {satellite}): {route_str} ({customers_count} customers)")
        print()

    def _employed_bee_phase(self):
        """Employed bees phase"""
        employed_count = int(self.employed_ratio * self.swarm_size)

        for i in range(employed_count):
            new_solution = self._modify_solution_with_constraint(self.swarm[i])
            new_fitness = self.calculate_fitness_with_constraint(new_solution)

            if new_fitness < self.fitness_values[i]:
                self.swarm[i] = new_solution
                self.fitness_values[i] = new_fitness
                self.trials[i] = 0

                if new_fitness < self.best_fitness:
                    self.best_solution = new_solution.copy()
                    self.best_fitness = new_fitness
            else:
                self.trials[i] += 1

    def _onlooker_bee_phase(self):
        """Onlooker bees phase"""
        employed_count = int(self.employed_ratio * self.swarm_size)
        onlooker_count = int(self.onlooker_ratio * self.swarm_size)

        valid_fitnesses = [f for f in self.fitness_values[:employed_count] if f != float('inf')]
        if not valid_fitnesses:
            return

        max_fit = max(valid_fitnesses)
        fitnesses = [max_fit - fit + 1 if fit != float('inf') else 0.01
                     for fit in self.fitness_values[:employed_count]]
        total_fit = sum(fitnesses)

        if total_fit > 0:
            probabilities = [f / total_fit for f in fitnesses]

            for _ in range(onlooker_count):
                idx = np.random.choice(range(employed_count), p=probabilities)

                new_solution = self._modify_solution_with_constraint(self.swarm[idx])
                new_fitness = self.calculate_fitness_with_constraint(new_solution)

                if new_fitness < self.fitness_values[idx]:
                    self.swarm[idx] = new_solution
                    self.fitness_values[idx] = new_fitness
                    self.trials[idx] = 0

                    if new_fitness < self.best_fitness:
                        self.best_solution = new_solution.copy()
                        self.best_fitness = new_fitness
                else:
                    self.trials[idx] += 1

    def _scout_bee_phase(self):
        """Scout bees phase"""
        employed_count = int(self.employed_ratio * self.swarm_size)

        for i in range(employed_count):
            if self.trials[i] >= self.limit:
                new_solution = MDMSMCVRPSolution(self.problem, self.num_vehicles_per_satellite)
                new_solution.initialize_random()

                new_solution = self.enforce_satellite_visits(new_solution)
                new_fitness = self.calculate_fitness_with_constraint(new_solution)

                self.swarm[i] = new_solution
                self.fitness_values[i] = new_fitness
                self.trials[i] = 0

                if new_fitness < self.best_fitness:
                    self.best_solution = new_solution.copy()
                    self.best_fitness = new_fitness

    def _modify_solution_with_constraint(self, solution: MDMSMCVRPSolution) -> MDMSMCVRPSolution:
        """Apply modification operators with multiple vehicles support"""
        new_solution = solution.copy()

        modification = random.choice([
            'reorder_satellites',
            'swap_customers',
            'change_customer_order',
            'swap_customers_between_vehicles'  # NEW: For multiple vehicles
        ])

        if modification == 'reorder_satellites':
            depot = random.choice(self.problem.VD)
            route = new_solution.routes['E1'][depot]

            satellite_positions = []
            for i, node in enumerate(route):
                if node in self.problem.VS:
                    satellite_positions.append((i, node))

            if len(satellite_positions) >= 2:
                satellites = [node for _, node in satellite_positions]
                random.shuffle(satellites)

                for (pos, _), new_sat in zip(satellite_positions, satellites):
                    route[pos] = new_sat

        elif modification == 'swap_customers' and len(self.problem.VS) > 1:
            # Get all vehicle IDs
            vehicle_ids = list(new_solution.routes['E2'].keys())
            if len(vehicle_ids) >= 2:
                v1, v2 = random.sample(vehicle_ids, 2)
                route1, route2 = new_solution.routes['E2'][v1], new_solution.routes['E2'][v2]

                if len(route1) > 2 and len(route2) > 2:
                    customers1 = [i for i, node in enumerate(route1) if node in self.problem.VC]
                    customers2 = [i for i, node in enumerate(route2) if node in self.problem.VC]

                    if customers1 and customers2:
                        i = random.choice(customers1)
                        j = random.choice(customers2)
                        route1[i], route2[j] = route2[j], route1[i]

        elif modification == 'change_customer_order':
            vehicle_ids = list(new_solution.routes['E2'].keys())
            if vehicle_ids:
                vehicle_id = random.choice(vehicle_ids)
                route = new_solution.routes['E2'][vehicle_id]

                if len(route) > 4:
                    customers = [i for i, node in enumerate(route) if node in self.problem.VC]
                    if len(customers) >= 2:
                        i, j = random.sample(customers, 2)
                        route[i], route[j] = route[j], route[i]

        elif modification == 'swap_customers_between_vehicles':
            # NEW: Swap customers between vehicles of same satellite
            satellite = random.choice(self.problem.VS)
            satellite_vehicles = [v for v in new_solution.routes['E2'].keys() if v.startswith(satellite)]

            if len(satellite_vehicles) >= 2:
                v1, v2 = random.sample(satellite_vehicles, 2)
                route1, route2 = new_solution.routes['E2'][v1], new_solution.routes['E2'][v2]

                customers1 = [i for i, node in enumerate(route1) if node in self.problem.VC]
                customers2 = [i for i, node in enumerate(route2) if node in self.problem.VC]

                if customers1 and customers2:
                    i = random.choice(customers1)
                    j = random.choice(customers2)
                    route1[i], route2[j] = route2[j], route1[i]

        if not self.validate_mandatory_satellite_visits(new_solution):
            new_solution = self.enforce_satellite_visits(new_solution)

        new_solution._calculate_flows()
        return new_solution

    def plot_convergence(self, save_plot=True, show_plot=True):
        """Enhanced convergence plot"""
        if not self.convergence_curve:
            print("No convergence data available!")
            return

        try:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(12, 6))
            plt.plot(self.convergence_curve, 'b-', linewidth=2, marker='o', markersize=3)

            total_vehicles = sum(self.num_vehicles_per_satellite.values())
            title = f'ABC Algorithm Convergence for MDMS-MCVRP\n({total_vehicles} vehicles: {self.num_vehicles_per_satellite})'
            plt.title(title, fontsize=14, fontweight='bold')
            plt.xlabel('Iteration', fontsize=12)
            plt.ylabel('Best Fitness', fontsize=12)
            plt.grid(True, alpha=0.3)

            if self.improvement_iterations:
                improvement_fitness = [self.convergence_curve[i] for i in self.improvement_iterations]
                plt.scatter(self.improvement_iterations, improvement_fitness,
                            color='red', s=60, marker='*', label='Improvements', zorder=5)
                plt.legend()

            best_idx = self.convergence_curve.index(min(self.convergence_curve))
            best_fitness = min(self.convergence_curve)
            plt.annotate(f'Best: {best_fitness:.2f}\nIteration: {best_idx}',
                         xy=(best_idx, best_fitness),
                         xytext=(best_idx + len(self.convergence_curve) * 0.1,
                                 best_fitness + (max(self.convergence_curve) - min(self.convergence_curve)) * 0.1),
                         arrowprops=dict(arrowstyle='->', color='red'),
                         fontsize=10, color='red')

            plt.tight_layout()

            if save_plot:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f'abc_convergence_multi_vehicle_{timestamp}.png'
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                print(f"✅ Convergence plot saved as '{filename}'")

            if show_plot:
                plt.show()

        except ImportError:
            print("❌ Matplotlib not available. Install with: pip install matplotlib")

    def print_best_solution(self):
        """Print comprehensive details of best solution"""
        if self.best_solution is None:
            print("No solution found!")
            return

        total_vehicles = sum(self.num_vehicles_per_satellite.values())
        print(f"\n=== COMPREHENSIVE BEST SOLUTION ANALYSIS ===")
        print(f"🏆 Best Fitness: {self.best_fitness:.2f}")
        print(f"🚐 Total Vehicles: {total_vehicles} ({self.num_vehicles_per_satellite})")
        print(f"📈 Total Improvements: {self.improvements_count}")
        print(f"🔄 Improvement Iterations: {self.improvement_iterations}")

        if self.validate_mandatory_satellite_visits(self.best_solution):
            print(f"✅ Constraint Satisfied: All primary vehicles visit all satellites")
        else:
            print(f"❌ Constraint Violated: Some satellites missed by primary vehicles")

        self.print_solution_routes(self.best_solution)

        try:
            stats = self.best_solution.get_route_statistics()
            print(f"📊 Solution Statistics:")
            print(f"   Customers Served: {stats['customers_served']}/{stats['total_customers']}")
            print(f"   Service Rate: {stats['service_rate']:.1%}")
            print(f"   Vehicles Used: {len(self.best_solution.routes['E2'])}")
        except AttributeError:
            print("📊 Basic solution metrics only available")

    def save_best_solution(self, filename_base='best_solution_multi_vehicle'):
        """Save best solution with multiple vehicles info"""
        if self.best_solution is None:
            print("No solution to save!")
            return

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        pickle_filename = f"{filename_base}_{timestamp}.pkl"
        with open(pickle_filename, 'wb') as f:
            pickle.dump(self.best_solution, f)
        print(f"✅ Solution saved as pickle: {pickle_filename}")

        json_filename = f"{filename_base}_routes_{timestamp}.json"
        route_data = {
            'timestamp': datetime.now().isoformat(),
            'fitness': float(self.best_fitness),
            'improvements_count': self.improvements_count,
            'num_vehicles_per_satellite': self.num_vehicles_per_satellite,
            'total_vehicles': sum(self.num_vehicles_per_satellite.values()),
            'constraint_satisfied': self.validate_mandatory_satellite_visits(self.best_solution),
            'constraint_type': 'mandatory_satellite_visits',
            'primary_routes': {},
            'secondary_routes': {},
            'optimization_summary': {
                'total_iterations': len(self.convergence_curve),
                'improvement_iterations': self.improvement_iterations,
                'final_fitness': float(self.best_fitness)
            }
        }

        for depot, route in self.best_solution.routes['E1'].items():
            satellites_count = len([s for s in route if s in self.problem.VS])
            visited_satellites = set([s for s in route if s in self.problem.VS])
            missing_satellites = list(set(self.problem.VS) - visited_satellites)

            route_data['primary_routes'][depot] = {
                'route': route,
                'satellites_count': satellites_count,
                'visits_all_satellites': len(missing_satellites) == 0,
                'missing_satellites': missing_satellites
            }

        for vehicle_id, route in self.best_solution.routes['E2'].items():
            satellite = vehicle_id.split('_V')[0]
            customers_count = len([c for c in route if c in self.problem.VC])
            route_data['secondary_routes'][vehicle_id] = {
                'route': route,
                'satellite': satellite,
                'customers_count': customers_count
            }

        with open(json_filename, 'w') as f:
            json.dump(route_data, f, indent=2)
        print(f"✅ Routes saved as JSON: {json_filename}")

        history_filename = f"{filename_base}_history_{timestamp}.json"
        with open(history_filename, 'w') as f:
            history_data = []
            for entry in self.best_solutions_history:
                history_entry = {
                    'iteration': entry['iteration'],
                    'fitness': entry['fitness'],
                    'timestamp': entry['timestamp'],
                    'constraint_satisfied': self.validate_mandatory_satellite_visits(entry['solution']),
                    'routes': {
                        'E1': entry['solution'].routes['E1'],
                        'E2': entry['solution'].routes['E2']
                    }
                }
                history_data.append(history_entry)

            json.dump({
                'optimization_history': history_data,
                'convergence_curve': [float(x) for x in self.convergence_curve],
                'num_vehicles_per_satellite': self.num_vehicles_per_satellite,
                'constraint_info': {
                    'type': 'mandatory_satellite_visits',
                    'description': 'Each primary vehicle must visit all satellites'
                }
            }, f, indent=2)
        print(f"✅ Optimization history saved: {history_filename}")

    @staticmethod
    def load_solution(filename):
        """Load previously saved solution"""
        try:
            with open(filename, 'rb') as f:
                solution = pickle.load(f)
            print(f"✅ Solution loaded from {filename}")
            return solution
        except Exception as e:
            print(f"❌ Error loading solution: {e}")
            return None
