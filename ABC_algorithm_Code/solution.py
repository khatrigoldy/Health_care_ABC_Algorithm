from collections import defaultdict
import random
import json
import copy
from typing import Dict, List, Optional


def create_nested_defaultdict():
    """Create nested defaultdict for flows - pickle-friendly"""
    return defaultdict(dict)


class MDMSMCVRPSolution:
    def __init__(self, problem, num_vehicles_per_satellite=None):
        """
        Initialize MDMS-MCVRP Solution with configurable vehicles per satellite

        Args:
            problem: Problem instance
            num_vehicles_per_satellite: Dict specifying number of vehicles for each satellite
                                       e.g., {'S1': 3, 'S2': 2} means 3 vehicles from S1, 2 from S2
                                       If None, defaults to 1 vehicle per satellite
        """
        self.problem = problem

        # NEW: Store number of vehicles per satellite
        if num_vehicles_per_satellite is None:
            self.num_vehicles_per_satellite = {k: 1 for k in problem.VS}
        else:
            self.num_vehicles_per_satellite = num_vehicles_per_satellite

        # NEW: Initialize multiple routes per satellite
        self.routes = {
            'E1': {l: [l, l] for l in problem.VD},
            'E2': {}  # Will be populated with multiple vehicles per satellite
        }

        # Initialize E2 routes based on number of vehicles
        for k in problem.VS:
            num_vehicles = self.num_vehicles_per_satellite[k]
            for v_idx in range(num_vehicles):
                vehicle_id = f"{k}_V{v_idx + 1}"  # e.g., S1_V1, S1_V2, S1_V3
                self.routes['E2'][vehicle_id] = [k, k]  # Start and end at satellite

        self.flows = {
            'Y': defaultdict(dict),
            'Z': defaultdict(create_nested_defaultdict)
        }
        self.fitness = float('inf')
        self.is_feasible = False
        self.constraint_violations = {}

    def initialize_random(self):
        """Initialize a capacity-aware feasible solution with multiple vehicles"""
        try:
            # Reset routes
            self.routes = {
                'E1': {l: [l, l] for l in self.problem.VD},
                'E2': {}
            }

            # Initialize multiple vehicles per satellite
            for k in self.problem.VS:
                num_vehicles = self.num_vehicles_per_satellite[k]
                for v_idx in range(num_vehicles):
                    vehicle_id = f"{k}_V{v_idx + 1}"
                    self.routes['E2'][vehicle_id] = [k, k]

            self.flows = {
                'Y': defaultdict(dict),
                'Z': defaultdict(create_nested_defaultdict)
            }

            # Assign satellites to primary vehicles
            for l in self.problem.VD:
                route = self.routes['E1'][l]
                available_satellites = self.problem.VS.copy()
                random.shuffle(available_satellites)

                num_satellites = random.choice([1, 2]) if len(available_satellites) > 1 else 1
                selected_satellites = available_satellites[:num_satellites]

                for s in selected_satellites:
                    route.insert(-1, s)

            # NEW: Assign customers to multiple vehicles with capacity constraints
            self._assign_customers_to_multiple_vehicles()

            self.validate_and_repair_solution()
            self._calculate_flows()
            self.calculate_fitness()

        except Exception as e:
            print(f"Error in initialize_random: {e}")
            self.fitness = float('inf')

    def _assign_customers_to_multiple_vehicles(self):
        """Assign customers to multiple satellite vehicles respecting capacity"""
        customers = self.problem.VC.copy()
        random.shuffle(customers)

        # Track vehicle loads for each vehicle
        vehicle_loads = {vehicle_id: 0 for vehicle_id in self.routes['E2'].keys()}

        # Get secondary vehicle capacity (assuming uniform capacity per satellite)
        vehicle_capacity = {}
        for vehicle_id in self.routes['E2'].keys():
            satellite = vehicle_id.split('_V')[0]  # Extract satellite ID
            # Use Q capacity for this satellite
            vehicle_capacity[vehicle_id] = sum(self.problem.Q[satellite].values())

        unassigned_customers = []

        for customer in customers:
            # Calculate customer's total demand
            customer_demand = sum(self.problem.demands.get(customer, {}).get(l, 0)
                                  for l in self.problem.VD)

            # Find vehicle with sufficient capacity (prefer same satellite region)
            assigned = False

            # Try to assign to any vehicle with capacity
            available_vehicles = [v for v in vehicle_loads.keys()
                                  if vehicle_loads[v] + customer_demand <= vehicle_capacity[v]]

            if available_vehicles:
                # Choose vehicle with minimum load (load balancing)
                best_vehicle = min(available_vehicles, key=lambda v: vehicle_loads[v])
                route = self.routes['E2'][best_vehicle]
                insert_pos = random.randint(1, len(route) - 1)
                route.insert(insert_pos, customer)
                vehicle_loads[best_vehicle] += customer_demand
                assigned = True

            if not assigned:
                # Assign to least loaded vehicle even if over capacity
                least_loaded = min(vehicle_loads.keys(), key=lambda v: vehicle_loads[v])
                route = self.routes['E2'][least_loaded]
                route.insert(-1, customer)
                vehicle_loads[least_loaded] += customer_demand
                unassigned_customers.append(customer)

        if unassigned_customers:
            print(f"Warning: {len(unassigned_customers)} customers may exceed vehicle capacities")

    def _calculate_flows(self):
        """Enhanced flow calculation with capacity constraints for multiple vehicles"""
        try:
            self.flows = {
                'Y': defaultdict(dict),
                'Z': defaultdict(create_nested_defaultdict)
            }

            # Calculate satellite demands (aggregate from all vehicles of that satellite)
            satellite_demands = {}
            for j in self.problem.VS:
                satellite_demands[j] = {}
                for l in self.problem.VD:
                    total_demand = 0
                    # Sum demands from all vehicles belonging to this satellite
                    for vehicle_id in self.routes['E2'].keys():
                        if vehicle_id.startswith(j):  # Vehicle belongs to this satellite
                            customers_in_vehicle = [c for c in self.routes['E2'][vehicle_id]
                                                    if c in self.problem.VC]
                            total_demand += sum(self.problem.demands.get(c, {}).get(l, 0)
                                                for c in customers_in_vehicle)

                    max_satellite_demand = self.problem.W[j] / len(self.problem.VD)
                    satellite_demands[j][l] = min(total_demand, max_satellite_demand)

            # Calculate primary flows (depot to satellite)
            for l in self.problem.VD:
                if l not in self.routes['E1']:
                    continue

                route = self.routes['E1'][l]
                remaining_capacity = self.problem.P[l]

                for i in range(1, len(route)):
                    if i >= len(route):
                        break

                    prev_node = route[i - 1]
                    current_node = route[i]

                    if current_node in self.problem.VS:
                        required_flow = satellite_demands.get(current_node, {}).get(l, 0)
                        actual_flow = min(required_flow, remaining_capacity)

                        if actual_flow > 0:
                            self.flows['Y'][(prev_node, current_node)][l] = actual_flow
                            remaining_capacity = max(0, remaining_capacity - actual_flow)

            # Calculate secondary flows (satellite vehicles to customers)
            for vehicle_id, route in self.routes['E2'].items():
                satellite = vehicle_id.split('_V')[0]  # Extract satellite ID
                compartment_loads = {l: 0 for l in self.problem.VD}

                for i in range(1, len(route)):
                    if i >= len(route):
                        break

                    prev_node = route[i - 1]
                    current_node = route[i]

                    if current_node in self.problem.VC:
                        for l in self.problem.VD:
                            demand = self.problem.demands.get(current_node, {}).get(l, 0)
                            available_capacity = self.problem.Q[satellite][l] - compartment_loads[l]

                            actual_flow = min(demand, available_capacity)
                            if actual_flow > 0:
                                self.flows['Z'][(prev_node, current_node)][vehicle_id][l] = actual_flow
                                compartment_loads[l] += actual_flow

        except Exception as e:
            print(f"Error in _calculate_flows: {e}")

    def copy(self):
        """Safe copy method for solution with multiple vehicles"""
        try:
            new_solution = MDMSMCVRPSolution(self.problem, self.num_vehicles_per_satellite)

            new_solution.routes = {
                'E1': {k: list(v) for k, v in self.routes['E1'].items()},
                'E2': {k: list(v) for k, v in self.routes['E2'].items()}
            }

            new_solution.flows = {
                'Y': {k: dict(v) for k, v in self.flows['Y'].items()},
                'Z': {k: {inner_k: dict(inner_v) for inner_k, inner_v in v.items()}
                      for k, v in self.flows['Z'].items()}
            }

            new_solution.fitness = self.fitness
            new_solution.is_feasible = self.is_feasible
            new_solution.constraint_violations = copy.copy(self.constraint_violations)

            return new_solution

        except Exception as e:
            print(f"Error in copy: {e}")
            new_solution = MDMSMCVRPSolution(self.problem, self.num_vehicles_per_satellite)
            return new_solution

    def validate_and_repair_solution(self):
        """Validate solution with multiple vehicles and attempt repairs"""
        print("=== SOLUTION VALIDATION (Multiple Vehicles) ===")
        repairs_made = 0

        # Ensure all customers are assigned exactly once
        assigned_customers = set()
        duplicate_assignments = []

        for vehicle_id in self.routes['E2'].keys():
            route = self.routes['E2'][vehicle_id]
            satellite = vehicle_id.split('_V')[0]
            unique_customers = []

            for node in route:
                if node in self.problem.VC:
                    if node in assigned_customers:
                        duplicate_assignments.append(node)
                    else:
                        assigned_customers.add(node)
                        unique_customers.append(node)
                elif node == satellite:
                    unique_customers.append(node)

            if len(unique_customers) != len([n for n in route if n == satellite or n in self.problem.VC]):
                self.routes['E2'][vehicle_id] = [satellite] + [c for c in unique_customers if c != satellite] + [
                    satellite]
                repairs_made += 1

        # Assign unassigned customers
        unassigned = set(self.problem.VC) - assigned_customers
        if unassigned:
            print(f"WARNING: Repairing {len(unassigned)} unassigned customers")
            for customer in unassigned:
                # Find vehicle with least customers
                best_vehicle = min(self.routes['E2'].keys(),
                                   key=lambda v: len([c for c in self.routes['E2'][v] if c in self.problem.VC]))
                self.routes['E2'][best_vehicle].insert(-1, customer)
                repairs_made += 1

        # Ensure primary routes start and end correctly
        for l in self.problem.VD:
            route = self.routes['E1'][l]
            if len(route) < 2 or route[0] != l or route[-1] != l:
                satellites = [node for node in route if node in self.problem.VS]
                self.routes['E1'][l] = [l] + satellites + [l]
                repairs_made += 1

        # Ensure secondary routes start and end at correct satellite
        for vehicle_id in self.routes['E2'].keys():
            satellite = vehicle_id.split('_V')[0]
            route = self.routes['E2'][vehicle_id]
            if len(route) < 2 or route[0] != satellite or route[-1] != satellite:
                customers = [node for node in route if node in self.problem.VC]
                self.routes['E2'][vehicle_id] = [satellite] + customers + [satellite]
                repairs_made += 1

        if repairs_made > 0:
            print(f"Made {repairs_made} repairs to solution")
            self._calculate_flows()

        print(
            f"Validation complete. All {len(self.problem.VC)} customers assigned across {len(self.routes['E2'])} vehicles.")

    def get_route_statistics(self):
        """Get detailed statistics about routes with multiple vehicles"""
        stats = {
            'primary_routes': {},
            'secondary_routes': {},
            'vehicles_per_satellite': {},
            'total_distance': 0,
            'total_customers': len(self.problem.VC),
            'customers_served': 0
        }

        # Primary route statistics
        for l, route in self.routes['E1'].items():
            route_distance = 0
            for i in range(len(route) - 1):
                if not self.problem.is_route_forbidden(route[i], route[i + 1]):
                    route_distance += self.problem.distances[route[i]][route[i + 1]]

            stats['primary_routes'][l] = {
                'route': route,
                'satellites_visited': len([s for s in route if s in self.problem.VS]),
                'route_length': len(route),
                'distance': route_distance
            }
            stats['total_distance'] += route_distance

        # Secondary route statistics (multiple vehicles)
        customers_served = set()
        for satellite in self.problem.VS:
            stats['vehicles_per_satellite'][satellite] = self.num_vehicles_per_satellite[satellite]

        for vehicle_id, route in self.routes['E2'].items():
            satellite = vehicle_id.split('_V')[0]
            customers_in_route = [c for c in route if c in self.problem.VC]
            customers_served.update(customers_in_route)

            route_distance = 0
            for i in range(len(route) - 1):
                if not self.problem.is_route_forbidden(route[i], route[i + 1]):
                    route_distance += self.problem.distances[route[i]][route[i + 1]]

            stats['secondary_routes'][vehicle_id] = {
                'route': route,
                'satellite': satellite,
                'customers_served': len(customers_in_route),
                'route_length': len(route),
                'customers': customers_in_route,
                'distance': route_distance
            }
            stats['total_distance'] += route_distance

        stats['customers_served'] = len(customers_served)
        stats['service_rate'] = len(customers_served) / len(self.problem.VC) if self.problem.VC else 0

        return stats

    def validate_solution(self):
        """Validate solution with multiple vehicles"""
        violations = []

        served_customers = set()
        for vehicle_id, route in self.routes['E2'].items():
            served_customers.update(c for c in route if c in self.problem.VC)

        unserved = set(self.problem.VC) - served_customers
        if unserved:
            violations.append(f"Unserved customers: {unserved}")

        for l, route in self.routes['E1'].items():
            if len(route) < 2 or route[0] != l or route[-1] != l:
                violations.append(f"Primary route {l} doesn't start/end at depot")

        for vehicle_id, route in self.routes['E2'].items():
            satellite = vehicle_id.split('_V')[0]
            if len(route) < 2 or route[0] != satellite or route[-1] != satellite:
                violations.append(f"Vehicle {vehicle_id} doesn't start/end at satellite {satellite}")

        forbidden_count = 0
        for l, route in self.routes['E1'].items():
            for i in range(len(route) - 1):
                if self.problem.is_route_forbidden(route[i], route[i + 1]):
                    forbidden_count += 1

        for vehicle_id, route in self.routes['E2'].items():
            for i in range(len(route) - 1):
                if self.problem.is_route_forbidden(route[i], route[i + 1]):
                    forbidden_count += 1

        if forbidden_count > 0:
            violations.append(f"Solution uses {forbidden_count} forbidden routes")

        return {
            'is_valid': len(violations) == 0,
            'violations': violations,
            'statistics': self.get_route_statistics()
        }

    def debug_constraints(self):
        """Enhanced constraint debugging for multiple vehicles"""
        print("=== CONSTRAINT ANALYSIS (Multiple Vehicles) ===")
        total_violations = 0
        penalty_weight = getattr(self.problem, 'penalty_weight', 100)

        print("\nPrimary Vehicle Capacity Analysis:")
        for l in self.problem.VD:
            if l not in self.routes['E1']:
                continue

            route = self.routes['E1'][l]
            total_load = 0
            for i in range(1, len(route)):
                if i >= len(route):
                    break
                prev, curr = route[i - 1], route[i]
                if curr in self.problem.VS:
                    load = self.flows['Y'].get((prev, curr), {}).get(l, 0)
                    total_load += load

            capacity = self.problem.P[l]
            violation = max(0, total_load - capacity)
            penalty = penalty_weight * violation if violation > 0 else 0
            total_violations += penalty
            status = "✅" if violation == 0 else "❌"
            print(
                f"  {status} Vehicle {l}: Load={total_load:.1f}, Capacity={capacity}, Violation={violation:.1f}, Penalty={penalty:.1f}")

        print("\nSatellite Capacity Analysis (Aggregated across vehicles):")
        for j in self.problem.VS:
            total_demand = 0
            num_vehicles = self.num_vehicles_per_satellite[j]

            for vehicle_id in self.routes['E2'].keys():
                if vehicle_id.startswith(j):
                    customers = [c for c in self.routes['E2'][vehicle_id] if c in self.problem.VC]
                    for customer in customers:
                        customer_demand = sum(self.problem.demands.get(customer, {}).values())
                        total_demand += customer_demand

            capacity = self.problem.W[j]
            violation = max(0, total_demand - capacity)
            penalty = penalty_weight * violation if violation > 0 else 0
            total_violations += penalty
            status = "✅" if violation == 0 else "❌"
            print(
                f"  {status} Satellite {j}: Demand={total_demand:.1f}, Capacity={capacity}, Vehicles={num_vehicles}, Violation={violation:.1f}, Penalty={penalty:.1f}")

        print("\nForbidden Route Analysis:")
        forbidden_e1 = 0
        forbidden_e2 = 0

        for l, route in self.routes['E1'].items():
            for i in range(len(route) - 1):
                if self.problem.is_route_forbidden(route[i], route[i + 1]):
                    forbidden_e1 += 1

        for vehicle_id, route in self.routes['E2'].items():
            for i in range(len(route) - 1):
                if self.problem.is_route_forbidden(route[i], route[i + 1]):
                    forbidden_e2 += 1

        total_forbidden = forbidden_e1 + forbidden_e2
        forbidden_penalty = total_forbidden * penalty_weight * 100
        total_violations += forbidden_penalty

        status = "✅" if total_forbidden == 0 else "❌"
        print(
            f"  {status} Forbidden Routes: E1={forbidden_e1}, E2={forbidden_e2}, Total={total_forbidden}, Penalty={forbidden_penalty:.1f}")

        route_distance = 0
        for l, route in self.routes['E1'].items():
            for i in range(len(route) - 1):
                if not self.problem.is_route_forbidden(route[i], route[i + 1]):
                    dist = self.problem.distances.get(route[i], {}).get(route[i + 1], 0)
                    route_distance += dist

        for vehicle_id, route in self.routes['E2'].items():
            for i in range(len(route) - 1):
                if not self.problem.is_route_forbidden(route[i], route[i + 1]):
                    dist = self.problem.distances.get(route[i], {}).get(route[i + 1], 0)
                    route_distance += dist

        print(f"\n{'=' * 50}")
        print(f"SUMMARY:")
        print(f"{'=' * 50}")
        print(
            f"  Total Vehicles: {len(self.routes['E2'])} ({', '.join([f'{k}: {v}' for k, v in self.num_vehicles_per_satellite.items()])})")
        print(f"  Total Route Distance: {route_distance:.1f}")
        print(f"  Total Penalty: {total_violations:.1f}")
        print(f"  Solution Fitness: {self.fitness:.1f}")
        print(f"  Expected Fitness: {route_distance + total_violations:.1f}")

        return {
            'route_distance': route_distance,
            'total_penalty': total_violations,
            'expected_fitness': route_distance + total_violations,
            'forbidden_routes': total_forbidden,
            'num_vehicles': len(self.routes['E2'])
        }

    def calculate_fitness(self) -> float:
        """Calculate solution fitness"""
        try:
            evaluation = self.problem.evaluate_solution(self.to_dict())
            self.fitness = evaluation['objective']
            self.constraint_violations = evaluation.get('violations', {})
            self.is_feasible = evaluation.get('penalty', 0) == 0
            return self.fitness
        except Exception as e:
            print(f"Error in calculate_fitness: {e}")
            self.fitness = float('inf')
            return self.fitness

    def to_dict(self) -> Dict:
        """Convert solution to dictionary format"""
        return {
            'E1': self.routes['E1'],
            'E2': self.routes['E2'],
            'Y': dict(self.flows['Y']),
            'Z': {k: dict(v) for k, v in self.flows['Z'].items()}
        }

    def to_dict_serializable(self):
        """Convert solution to JSON-serializable format"""
        return {
            'fitness': float(self.fitness),
            'is_feasible': self.is_feasible,
            'num_vehicles_per_satellite': self.num_vehicles_per_satellite,
            'routes': {
                'E1': {str(k): list(v) for k, v in self.routes['E1'].items()},
                'E2': {str(k): list(v) for k, v in self.routes['E2'].items()}
            },
            'constraint_violations': self.constraint_violations
        }

    def save_to_file(self, filename):
        """Save solution to JSON file"""
        try:
            data = self.to_dict_serializable()
            data['validation'] = self.validate_solution()

            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"Solution saved to {filename}")
        except Exception as e:
            print(f"Error saving solution: {e}")

    @classmethod
    def load_from_file(cls, filename, problem, num_vehicles_per_satellite=None):
        """Load solution from JSON file"""
        try:
            with open(filename, 'r') as f:
                data = json.load(f)

            if num_vehicles_per_satellite is None and 'num_vehicles_per_satellite' in data:
                num_vehicles_per_satellite = data['num_vehicles_per_satellite']

            solution = cls(problem, num_vehicles_per_satellite)
            solution.fitness = data['fitness']
            solution.is_feasible = data.get('is_feasible', False)
            solution.constraint_violations = data.get('constraint_violations', {})

            solution.routes['E1'] = {k: v for k, v in data['routes']['E1'].items()}
            solution.routes['E2'] = {k: v for k, v in data['routes']['E2'].items()}

            solution._calculate_flows()

            print(f"Solution loaded from {filename}")
            return solution

        except Exception as e:
            print(f"Error loading solution: {e}")
            return None

    def print_readable(self):
        """Print solution in human-readable format"""
        print(f"\n{'=' * 50}")
        print(f"SOLUTION SUMMARY (Multiple Vehicles)")
        print(f"{'=' * 50}")
        print(f"Total Fitness: {self.fitness:.2f}")
        print(f"Feasible: {'Yes' if self.is_feasible else 'No'}")
        print(f"Vehicles per Satellite: {self.num_vehicles_per_satellite}")

        if self.constraint_violations:
            print(f"Constraint Violations: {len(self.constraint_violations)} types")

        print(f"\nFirst Echelon Routes (Depots → Satellites):")
        for l, route in self.routes['E1'].items():
            route_str = ' → '.join(map(str, route))
            print(f"  Primary Vehicle {l}: {route_str}")

        print(f"\nSecond Echelon Routes (Satellites → Customers with Multiple Vehicles):")
        for vehicle_id, route in sorted(self.routes['E2'].items()):
            satellite = vehicle_id.split('_V')[0]
            route_str = ' → '.join(map(str, route))
            customer_count = len([c for c in route if c in self.problem.VC])
            print(f"  Vehicle {vehicle_id} (from {satellite}): {route_str} ({customer_count} customers)")

        self.debug_constraints()

    def __str__(self):
        """String representation"""
        total_vehicles = sum(self.num_vehicles_per_satellite.values())
        return f"MDMSMCVRPSolution(fitness={self.fitness:.2f}, feasible={self.is_feasible}, vehicles={total_vehicles})"

    def __repr__(self):
        """Detailed string representation"""
        return self.__str__()
