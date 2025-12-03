from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import json
import numpy as np


class MDMSMCVRPProblem:
    def __init__(self, VD: List[str], VS: List[str], VC: List[str],
                 distances: Dict, demands: Dict,
                 P: Dict, Q: Dict, W: Dict,
                 num_vehicles_per_satellite: Optional[Dict] = None,
                 penalty_weight: float = 100.0):
        """
        Initialize the MDMS-MCVRP problem instance with multiple vehicles per satellite support.

        Args:
            VD: List of depots
            VS: List of satellites
            VC: List of customers
            distances: Distance matrix
            demands: Customer demands for each commodity
            P: Primary vehicle capacities
            Q: Secondary vehicle compartment capacities
            W: Satellite capacities
            num_vehicles_per_satellite: Dict specifying vehicles per satellite, e.g., {'S1': 3, 'S2': 2}
            penalty_weight: Weight for constraint violation penalties
        """
        self.VD = VD
        self.VS = VS
        self.VC = VC
        self.distances = distances
        self.demands = demands
        self.P = P
        self.Q = Q
        self.W = W
        self.num_vehicles_per_satellite = num_vehicles_per_satellite or {k: 1 for k in VS}
        self.penalty_weight = penalty_weight

        # Define forbidden distance threshold
        self.FORBIDDEN_DISTANCE = 9999
        self.MAX_ALLOWED_DISTANCE = 5000

        # Validate problem instance
        self._validate_problem_instance()

        # Calculate problem statistics
        self.problem_stats = self._calculate_problem_statistics()

        # Print initialization summary
        self._print_initialization_summary()

    def _print_initialization_summary(self):
        """Print key problem characteristics with multiple vehicles info"""
        total_vehicles = sum(self.num_vehicles_per_satellite.values())
        print(f"\n{'=' * 50}")
        print("PROBLEM INITIALIZATION SUMMARY")
        print(f"{'=' * 50}")
        print(f"Nodes: {len(self.VD)} depots, {len(self.VS)} satellites, {len(self.VC)} customers")
        print(f"Secondary Vehicles: {total_vehicles} total ({self.num_vehicles_per_satellite})")
        print(f"Total demand: {sum(sum(self.demands[c].values()) for c in self.VC):.1f}")
        print(f"Total satellite capacity: {sum(self.W.values()):.1f}")
        print(f"Penalty weight: {self.penalty_weight}x")

        # Check capacity feasibility
        total_demand = sum(sum(self.demands[c].values()) for c in self.VC)
        total_satellite_capacity = sum(self.W.values())
        capacity_ratio = total_demand / total_satellite_capacity

        if capacity_ratio > 1.0:
            print(
                f"⚠️  WARNING: Total demand ({total_demand:.1f}) exceeds satellite capacity ({total_satellite_capacity:.1f})")
            print(f"   Capacity utilization: {capacity_ratio:.1%}")
        else:
            print(f"✅ Capacity feasible: {capacity_ratio:.1%} utilization")

    def _validate_problem_instance(self):
        """Enhanced validation with multiple vehicles support"""
        errors = []
        warnings = []

        # Check if all required sets are non-empty
        if not self.VD:
            errors.append("Depot set (VD) cannot be empty")
        if not self.VS:
            errors.append("Satellite set (VS) cannot be empty")
        if not self.VC:
            errors.append("Customer set (VC) cannot be empty")

        # Validate num_vehicles_per_satellite
        if self.num_vehicles_per_satellite:
            for satellite in self.VS:
                if satellite not in self.num_vehicles_per_satellite:
                    errors.append(f"Number of vehicles not specified for satellite {satellite}")
                elif self.num_vehicles_per_satellite[satellite] < 1:
                    errors.append(f"Invalid number of vehicles for satellite {satellite}: must be >= 1")

        # Enhanced distance matrix validation
        all_nodes = set(self.VD + self.VS + self.VC)
        forbidden_routes = []

        for node1 in all_nodes:
            if node1 not in self.distances:
                errors.append(f"Distance matrix missing entries for node {node1}")
            else:
                for node2 in all_nodes:
                    if node2 not in self.distances[node1]:
                        errors.append(f"Distance matrix missing entry from {node1} to {node2}")
                    else:
                        distance = self.distances[node1][node2]
                        if distance >= self.FORBIDDEN_DISTANCE:
                            forbidden_routes.append((node1, node2, distance))

        # Report forbidden routes statistics
        if forbidden_routes:
            print(f"Found {len(forbidden_routes)} forbidden routes (distance >= {self.FORBIDDEN_DISTANCE})")

            # Check for critical forbidden routes
            critical_routes = []
            for node1, node2, dist in forbidden_routes:
                if (node1 in self.VD and node2 in self.VS) or (node1 in self.VS and node2 in self.VC):
                    critical_routes.append((node1, node2, dist))

            if critical_routes:
                warnings.append(f"Found {len(critical_routes)} critical forbidden routes that may cause infeasibility")

        # Check demand structure
        for customer in self.VC:
            if customer not in self.demands:
                errors.append(f"Demands missing for customer {customer}")
            else:
                for depot in self.VD:
                    if depot not in self.demands[customer]:
                        errors.append(f"Demand missing for customer {customer}, commodity {depot}")

        # Check capacity structures
        for depot in self.VD:
            if depot not in self.P:
                errors.append(f"Primary vehicle capacity missing for depot {depot}")

        for satellite in self.VS:
            if satellite not in self.Q:
                errors.append(f"Secondary vehicle capacity missing for satellite {satellite}")
            else:
                for depot in self.VD:
                    if depot not in self.Q[satellite]:
                        errors.append(f"Compartment capacity missing for satellite {satellite}, commodity {depot}")

            if satellite not in self.W:
                errors.append(f"Satellite capacity missing for satellite {satellite}")

        if errors:
            raise ValueError("Problem instance validation failed:\n" + "\n".join(errors))

        if warnings:
            print("⚠️ Validation warnings:")
            for warning in warnings:
                print(f"  {warning}")

    def is_route_forbidden(self, from_node: str, to_node: str) -> bool:
        """Check if a route segment uses forbidden connections"""
        if from_node not in self.distances or to_node not in self.distances[from_node]:
            return True

        distance = self.distances[from_node][to_node]
        return distance >= self.FORBIDDEN_DISTANCE

    def get_route_distance(self, from_node: str, to_node: str) -> float:
        """Get distance with forbidden route detection"""
        if self.is_route_forbidden(from_node, to_node):
            return float('inf')

        return self.distances[from_node][to_node]

    def validate_route_feasibility(self, route: List[str]) -> Dict:
        """Validate that a route doesn't use forbidden connections"""
        forbidden_segments = []
        total_distance = 0

        for i in range(len(route) - 1):
            from_node, to_node = route[i], route[i + 1]

            if self.is_route_forbidden(from_node, to_node):
                forbidden_segments.append((from_node, to_node, self.distances[from_node][to_node]))
            else:
                total_distance += self.distances[from_node][to_node]

        return {
            'is_feasible': len(forbidden_segments) == 0,
            'forbidden_segments': forbidden_segments,
            'total_distance': total_distance if len(forbidden_segments) == 0 else float('inf')
        }

    def evaluate_solution(self, solution: Dict) -> Dict:
        """Enhanced solution evaluation with multiple vehicles support"""
        try:
            total_distance = 0
            penalty = 0
            violations = {}
            detailed_costs = {
                'first_echelon_distance': 0,
                'second_echelon_distance': 0,
                'primary_capacity_penalty': 0,
                'secondary_capacity_penalty': 0,
                'satellite_capacity_penalty': 0,
                'forbidden_route_penalty': 0
            }

            # Calculate first echelon distance
            forbidden_routes_e1 = []
            for l, route in solution['E1'].items():
                route_validation = self.validate_route_feasibility(route)

                if not route_validation['is_feasible']:
                    forbidden_routes_e1.extend(route_validation['forbidden_segments'])
                    penalty_amount = len(route_validation['forbidden_segments']) * self.penalty_weight * 100
                    penalty += penalty_amount
                    detailed_costs['forbidden_route_penalty'] += penalty_amount
                else:
                    detailed_costs['first_echelon_distance'] += route_validation['total_distance']
                    total_distance += route_validation['total_distance']

            # Calculate second echelon distance (supports both old format and multi-vehicle format)
            forbidden_routes_e2 = []
            for vehicle_id, route in solution['E2'].items():
                route_validation = self.validate_route_feasibility(route)

                if not route_validation['is_feasible']:
                    forbidden_routes_e2.extend(route_validation['forbidden_segments'])
                    penalty_amount = len(route_validation['forbidden_segments']) * self.penalty_weight * 100
                    penalty += penalty_amount
                    detailed_costs['forbidden_route_penalty'] += penalty_amount
                else:
                    detailed_costs['second_echelon_distance'] += route_validation['total_distance']
                    total_distance += route_validation['total_distance']

            # Record forbidden route violations
            if forbidden_routes_e1 or forbidden_routes_e2:
                violations['forbidden_routes'] = {
                    'first_echelon': forbidden_routes_e1,
                    'second_echelon': forbidden_routes_e2,
                    'total_count': len(forbidden_routes_e1) + len(forbidden_routes_e2)
                }

            # Check constraints
            primary_violations = self._check_primary_capacity_constraints(solution, detailed_costs)
            if primary_violations:
                violations.update(primary_violations)

            secondary_violations = self._check_secondary_capacity_constraints(solution, detailed_costs)
            if secondary_violations:
                violations.update(secondary_violations)

            satellite_violations = self._check_satellite_capacity_constraints(solution, detailed_costs)
            if satellite_violations:
                violations.update(satellite_violations)

            demand_violations = self._check_demand_satisfaction(solution)
            if demand_violations:
                violations.update(demand_violations)

            # Update penalty
            penalty = sum([
                detailed_costs['primary_capacity_penalty'],
                detailed_costs['secondary_capacity_penalty'],
                detailed_costs['satellite_capacity_penalty'],
                detailed_costs['forbidden_route_penalty']
            ])

            return {
                'total_distance': total_distance,
                'penalty': penalty,
                'violations': violations,
                'objective': total_distance + penalty,
                'detailed_costs': detailed_costs,
                'is_feasible': penalty == 0,
                'forbidden_routes_count': len(forbidden_routes_e1) + len(forbidden_routes_e2)
            }

        except Exception as e:
            print(f"Error in evaluate_solution: {e}")
            return {
                'total_distance': float('inf'),
                'penalty': float('inf'),
                'violations': {'evaluation_error': str(e)},
                'objective': float('inf'),
                'detailed_costs': {},
                'is_feasible': False,
                'forbidden_routes_count': 0
            }

    def _check_primary_capacity_constraints(self, solution: Dict, detailed_costs: Dict) -> Dict:
        """Check primary vehicle capacity constraints"""
        violations = {}

        for l in self.VD:
            if l not in solution['E1']:
                continue

            route = solution['E1'][l]
            max_load = 0
            current_load = 0

            for i in range(1, len(route)):
                prev_node = route[i - 1]
                current_node = route[i]

                segment_flow = solution.get('Y', {}).get((prev_node, current_node), {}).get(l, 0)
                current_load += segment_flow
                max_load = max(max_load, current_load)

                if current_node == l:
                    current_load = 0

            if max_load > self.P[l]:
                excess = max_load - self.P[l]
                penalty_amount = self.penalty_weight * excess
                detailed_costs['primary_capacity_penalty'] += penalty_amount

                violations[f'primary_capacity_{l}'] = {
                    'capacity': self.P[l],
                    'used': max_load,
                    'excess': excess,
                    'penalty': penalty_amount
                }

        return violations

    def _check_secondary_capacity_constraints(self, solution: Dict, detailed_costs: Dict) -> Dict:
        """Check secondary vehicle compartment capacity constraints (supports multiple vehicles)"""
        violations = {}

        for vehicle_id, route in solution['E2'].items():
            # Extract satellite ID (handles both 'S1' and 'S1_V1' formats)
            satellite = vehicle_id.split('_V')[0] if '_V' in vehicle_id else vehicle_id

            if satellite not in self.VS:
                continue

            max_compartment_loads = {l: 0 for l in self.VD}
            compartment_loads = {l: 0 for l in self.VD}

            for i in range(1, len(route)):
                prev_node = route[i - 1]
                current_node = route[i]

                for l in self.VD:
                    # Handle both old format (Z[edge][satellite][l]) and new format (Z[edge][vehicle_id][l])
                    segment_flow = solution.get('Z', {}).get((prev_node, current_node), {}).get(vehicle_id, {}).get(l,
                                                                                                                    0)
                    compartment_loads[l] += segment_flow
                    max_compartment_loads[l] = max(max_compartment_loads[l], compartment_loads[l])

                    if current_node in self.VC:
                        delivered = self.demands.get(current_node, {}).get(l, 0)
                        compartment_loads[l] = max(0, compartment_loads[l] - delivered)

            # Check capacity violations
            for l, max_load in max_compartment_loads.items():
                if max_load > self.Q[satellite][l]:
                    excess = max_load - self.Q[satellite][l]
                    penalty_amount = self.penalty_weight * excess
                    detailed_costs['secondary_capacity_penalty'] += penalty_amount

                    violations[f'secondary_capacity_{vehicle_id}_{l}'] = {
                        'vehicle_id': vehicle_id,
                        'satellite': satellite,
                        'capacity': self.Q[satellite][l],
                        'used': max_load,
                        'excess': excess,
                        'penalty': penalty_amount
                    }

        return violations

    def _check_satellite_capacity_constraints(self, solution: Dict, detailed_costs: Dict) -> Dict:
        """Check satellite capacity constraints (aggregated across all vehicles)"""
        violations = {}

        for satellite in self.VS:
            # Aggregate demand from all vehicles serving this satellite
            total_satellite_demand = 0
            total_customers_served = 0

            for vehicle_id, route in solution['E2'].items():
                # Check if this vehicle belongs to this satellite
                vehicle_satellite = vehicle_id.split('_V')[0] if '_V' in vehicle_id else vehicle_id

                if vehicle_satellite == satellite:
                    customers_served = [c for c in route if c in self.VC]
                    total_customers_served += len(customers_served)

                    for customer in customers_served:
                        customer_total_demand = sum(self.demands.get(customer, {}).get(l, 0) for l in self.VD)
                        total_satellite_demand += customer_total_demand

            if total_satellite_demand > self.W[satellite]:
                excess = total_satellite_demand - self.W[satellite]
                penalty_amount = self.penalty_weight * excess
                detailed_costs['satellite_capacity_penalty'] += penalty_amount

                violations[f'satellite_capacity_{satellite}'] = {
                    'capacity': self.W[satellite],
                    'used': total_satellite_demand,
                    'excess': excess,
                    'penalty': penalty_amount,
                    'customers_served': total_customers_served,
                    'num_vehicles': self.num_vehicles_per_satellite.get(satellite, 1)
                }

        return violations

    def _check_demand_satisfaction(self, solution: Dict) -> Dict:
        """Enhanced demand satisfaction checking"""
        violations = {}

        served_customers = set()
        customer_assignments = {}

        for vehicle_id, route in solution['E2'].items():
            for node in route:
                if node in self.VC:
                    if node in served_customers:
                        violations[f'duplicate_assignment_{node}'] = {
                            'previously_assigned_to': customer_assignments[node],
                            'also_assigned_to': vehicle_id
                        }
                    else:
                        served_customers.add(node)
                        customer_assignments[node] = vehicle_id

        unserved_customers = set(self.VC) - served_customers
        if unserved_customers:
            violations['unserved_customers'] = list(unserved_customers)

        return violations

    def debug_solution_constraints(self, solution: Dict) -> Dict:
        """Comprehensive constraint debugging with multiple vehicles support"""
        debug_info = {
            'route_feasibility': {},
            'capacity_analysis': {},
            'flow_analysis': {},
            'demand_analysis': {},
            'vehicle_utilization': {}
        }

        # Analyze route feasibility
        for l, route in solution['E1'].items():
            debug_info['route_feasibility'][f'E1_{l}'] = self.validate_route_feasibility(route)

        for vehicle_id, route in solution['E2'].items():
            debug_info['route_feasibility'][f'E2_{vehicle_id}'] = self.validate_route_feasibility(route)

        # Capacity utilization analysis
        debug_info['capacity_analysis'] = {
            'primary_vehicles': {},
            'satellites': {},
            'secondary_vehicles': {}
        }

        # Satellite utilization (aggregated)
        for satellite in self.VS:
            total_demand = 0
            total_customers = 0
            vehicles_used = 0

            for vehicle_id, route in solution['E2'].items():
                vehicle_satellite = vehicle_id.split('_V')[0] if '_V' in vehicle_id else vehicle_id
                if vehicle_satellite == satellite:
                    vehicles_used += 1
                    customers = [c for c in route if c in self.VC]
                    total_customers += len(customers)
                    total_demand += sum(sum(self.demands.get(c, {}).values()) for c in customers)

            utilization = total_demand / self.W[satellite] if self.W[satellite] > 0 else 0

            debug_info['capacity_analysis']['satellites'][satellite] = {
                'capacity': self.W[satellite],
                'used': total_demand,
                'utilization': utilization,
                'customers_count': total_customers,
                'vehicles_used': vehicles_used,
                'vehicles_available': self.num_vehicles_per_satellite.get(satellite, 1),
                'is_overloaded': total_demand > self.W[satellite]
            }

        # Vehicle utilization
        for vehicle_id, route in solution['E2'].items():
            satellite = vehicle_id.split('_V')[0] if '_V' in vehicle_id else vehicle_id
            customers = [c for c in route if c in self.VC]
            vehicle_demand = sum(sum(self.demands.get(c, {}).values()) for c in customers)
            vehicle_capacity = sum(self.Q[satellite].values())

            debug_info['vehicle_utilization'][vehicle_id] = {
                'satellite': satellite,
                'customers_served': len(customers),
                'total_demand': vehicle_demand,
                'capacity': vehicle_capacity,
                'utilization': vehicle_demand / vehicle_capacity if vehicle_capacity > 0 else 0
            }

        return debug_info

    def get_solution_quality_metrics(self, solution: Dict) -> Dict:
        """Calculate comprehensive solution quality metrics with multiple vehicles"""
        evaluation = self.evaluate_solution(solution)
        total_vehicles = sum(self.num_vehicles_per_satellite.values())

        metrics = {
            'objective_value': evaluation['objective'],
            'total_distance': evaluation['total_distance'],
            'total_penalty': evaluation['penalty'],
            'penalty_percentage': (evaluation['penalty'] / evaluation['objective'] * 100) if evaluation[
                                                                                                 'objective'] > 0 else 0,
            'is_feasible': evaluation['is_feasible'],
            'forbidden_routes_count': evaluation.get('forbidden_routes_count', 0),
            'total_vehicles': total_vehicles,
            'vehicles_per_satellite': self.num_vehicles_per_satellite
        }

        # Calculate efficiency metrics
        total_demand = sum(sum(self.demands[c].values()) for c in self.VC)
        total_satellite_capacity = sum(self.W.values())

        metrics.update({
            'capacity_utilization': total_demand / total_satellite_capacity if total_satellite_capacity > 0 else 0,
            'distance_per_customer': evaluation['total_distance'] / len(self.VC) if len(self.VC) > 0 else 0,
            'distance_per_vehicle': evaluation['total_distance'] / total_vehicles if total_vehicles > 0 else 0,
            'solution_efficiency': total_demand / evaluation['objective'] if evaluation['objective'] > 0 else 0
        })

        return metrics

    def _calculate_problem_statistics(self):
        """Enhanced problem statistics with multiple vehicles info"""
        total_vehicles = sum(self.num_vehicles_per_satellite.values())

        stats = {
            'num_depots': len(self.VD),
            'num_satellites': len(self.VS),
            'num_customers': len(self.VC),
            'num_commodities': len(self.VD),
            'total_nodes': len(self.VD) + len(self.VS) + len(self.VC),
            'total_secondary_vehicles': total_vehicles,
            'vehicles_per_satellite': self.num_vehicles_per_satellite
        }

        # Calculate total demand
        total_demand = 0
        max_customer_demand = 0
        for customer in self.VC:
            customer_demand = sum(self.demands[customer][depot] for depot in self.VD)
            total_demand += customer_demand
            max_customer_demand = max(max_customer_demand, customer_demand)

        stats.update({
            'total_demand': total_demand,
            'average_customer_demand': total_demand / len(self.VC) if self.VC else 0,
            'max_customer_demand': max_customer_demand
        })

        # Calculate capacity statistics
        stats.update({
            'total_primary_capacity': sum(self.P.values()),
            'total_satellite_capacity': sum(self.W.values()),
            'average_primary_capacity': np.mean(list(self.P.values())),
            'average_satellite_capacity': np.mean(list(self.W.values())),
            'capacity_feasibility_ratio': total_demand / sum(self.W.values()) if sum(self.W.values()) > 0 else float(
                'inf')
        })

        # Distance matrix statistics
        valid_distances = []
        forbidden_count = 0

        for from_node, destinations in self.distances.items():
            for to_node, distance in destinations.items():
                if distance >= self.FORBIDDEN_DISTANCE:
                    forbidden_count += 1
                else:
                    valid_distances.append(distance)

        if valid_distances:
            stats.update({
                'min_distance': min(valid_distances),
                'max_distance': max(valid_distances),
                'avg_distance': np.mean(valid_distances),
                'forbidden_routes_count': forbidden_count,
                'valid_routes_count': len(valid_distances)
            })

        return stats

    def calculate_lower_bound(self) -> float:
        """Enhanced lower bound calculation"""
        try:
            min_distance = 0

            for customer in self.VC:
                min_dist_to_customer = float('inf')
                for satellite in self.VS:
                    if not self.is_route_forbidden(satellite, customer):
                        dist = self.distances[satellite][customer]
                        min_dist_to_customer = min(min_dist_to_customer, dist)

                if min_dist_to_customer != float('inf'):
                    min_distance += min_dist_to_customer * 2

            for depot in self.VD:
                min_dist_to_satellite = float('inf')
                for satellite in self.VS:
                    if not self.is_route_forbidden(depot, satellite):
                        dist = self.distances[depot][satellite]
                        min_dist_to_satellite = min(min_dist_to_satellite, dist)

                if min_dist_to_satellite != float('inf'):
                    min_distance += min_dist_to_satellite * 2

            return min_distance

        except Exception as e:
            print(f"Error calculating lower bound: {e}")
            return 0

    def save_problem_instance(self, filename: str):
        """Save problem instance with multiple vehicles info"""
        try:
            problem_data = {
                'VD': self.VD,
                'VS': self.VS,
                'VC': self.VC,
                'distances': self.distances,
                'demands': self.demands,
                'P': self.P,
                'Q': self.Q,
                'W': self.W,
                'num_vehicles_per_satellite': self.num_vehicles_per_satellite,
                'penalty_weight': self.penalty_weight,
                'statistics': self.problem_stats
            }

            with open(filename, 'w') as f:
                json.dump(problem_data, f, indent=2)

            print(f"Problem instance saved to {filename}")

        except Exception as e:
            print(f"Error saving problem instance: {e}")

    @classmethod
    def load_problem_instance(cls, filename: str):
        """Load problem instance from JSON"""
        try:
            with open(filename, 'r') as f:
                data = json.load(f)

            data.pop('statistics', None)

            return cls(**data)

        except Exception as e:
            print(f"Error loading problem instance: {e}")
            return None

    def print_problem_summary(self):
        """Enhanced problem summary with multiple vehicles info"""
        total_vehicles = sum(self.num_vehicles_per_satellite.values())

        print("=" * 60)
        print("MDMS-MCVRP PROBLEM INSTANCE SUMMARY")
        print("=" * 60)

        print(f"Problem Size:")
        print(f"  Depots: {len(self.VD)}")
        print(f"  Satellites: {len(self.VS)}")
        print(f"  Customers: {len(self.VC)}")
        print(f"  Total Nodes: {len(self.VD) + len(self.VS) + len(self.VC)}")
        print(f"  Secondary Vehicles: {total_vehicles} ({self.num_vehicles_per_satellite})")

        print(f"\nCapacity Information:")
        print(f"  Total Primary Vehicle Capacity: {sum(self.P.values())}")
        print(f"  Total Satellite Capacity: {sum(self.W.values())}")
        print(f"  Average Primary Capacity: {np.mean(list(self.P.values())):.2f}")
        print(f"  Average Satellite Capacity: {np.mean(list(self.W.values())):.2f}")

        print(f"\nDemand Information:")
        total_demand = sum(sum(self.demands[c].values()) for c in self.VC)
        print(f"  Total Demand: {total_demand}")
        print(f"  Average Customer Demand: {total_demand / len(self.VC):.2f}")

        # Capacity feasibility
        capacity_ratio = total_demand / sum(self.W.values()) if sum(self.W.values()) > 0 else float('inf')
        print(f"\nFeasibility Analysis:")
        print(f"  Capacity Utilization: {capacity_ratio:.1%}")
        if capacity_ratio > 1.0:
            print(f"  ⚠️ WARNING: Demand exceeds satellite capacity!")
        else:
            print(f"  ✅ Capacity constraints are satisfiable")

        print(f"\nDistance Matrix:")
        if 'forbidden_routes_count' in self.problem_stats:
            print(f"  Valid routes: {self.problem_stats['valid_routes_count']}")
            print(f"  Forbidden routes: {self.problem_stats['forbidden_routes_count']}")
            print(f"  Average valid distance: {self.problem_stats.get('avg_distance', 0):.2f}")

        print(f"\nLower Bound Estimate: {self.calculate_lower_bound():.2f}")
        print(f"Penalty Weight: {self.penalty_weight}x")

        print("=" * 60)

    def __str__(self):
        """String representation"""
        total_vehicles = sum(self.num_vehicles_per_satellite.values())
        return f"MDMS-MCVRP({len(self.VD)}D-{len(self.VS)}S-{len(self.VC)}C-{total_vehicles}V)"

    def __repr__(self):
        """Detailed string representation"""
        return self.__str__()
