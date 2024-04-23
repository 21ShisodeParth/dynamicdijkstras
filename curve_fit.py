import numpy as np
import heapq
from scipy.optimize import curve_fit

# Creates fake polynomial data with noise based on coefficients passed into args
# Returns x values and y values of fake data
def poly_data(coeffs, start, end, num_points, mean, std_dev):
    # Generate evenly spaced data
    X = np.linspace(start, end, num_points)
    Y = sum(np.array([coeffs[i] * X**i for i in range(len(coeffs))]))

    # Generate Gaussian noise to add to data
    Y_noise = np.random.normal(mean, std_dev, num_points)

    return X, Y + Y_noise


def equation(x, args):
    coeffs = [args[i] * np.array(x)**i for i in range(len(args))]
    return sum(coeffs)


# Fits data X and Y to a polynomial curve of specified degree
# Returns polynomial coefficients as a list
def fit_curve(X, Y, deg):
    init = np.ones(deg + 1)
    eq = lambda x, *args: sum([args[i] * x**i for i in range(len(args))])
    return curve_fit(eq, X, Y, init)[0]

def dijkstra(graph_mat, source):
    distances = [np.inf for i in range(len(graph_mat))]
    shortest_paths = [[] for i in range(len(graph_mat))]
    distances[source] = 0
    shortest_paths[source] = [source]


    temp_q = [(0, source)]

    # Heap queue has 3 elements: distance, target node, elapsed time
    # temp_q = [(0, source, 0)]

    while temp_q: # Runs as long as there are still unvisited nodes
                # Pop node with smallest distance from priority queue
            current_distance, current_node = heapq.heappop(temp_q)
            
            # Check if current distance is outdated
            if current_distance > distances[current_node]:
                continue
            
            # Explore neighbors of current node
            for i in range(len(graph_mat[current_node])):
                weight = graph_mat[current_node][i]
                if weight > 0:
                    distance = current_distance + weight
                    
                    # If new distance is shorter, update distances dictionary and priority queue
                    if distance < distances[i]:
                        distances[i] = distance
                        shortest_paths[i] = shortest_paths[current_node] + [i]  # Update shortest path
                        heapq.heappush(temp_q, (distance, i))

    return distances, shortest_paths

def alter_data():
    altered_coeffs = np.array([
    47, 
    9e-03, 
    3e-02,  
    4e-03,
    -1e-05,  
    -3e-05])

    x_points = np.linspace(0, 10, 200)
    altered_y_fit = equation(x_points, altered_coeffs)

    return altered_y_fit




# Extract the parameters
#a_fit, b_fit, c_fit, d_fit, e_fit, f_fit = params

# Generate y values for the fitted curve

# Plot the original data with noise
#x_data, y_data = poly_data([25, -1, 0.2, 0.03, 0.002], 0, 10, 1000, 0, 3)
#params = fit_curve(x_data, y_data, 4)

#print(x_data)

#y_fit = equation(x_data, params)
#print(params)
#print(fit_curve(x_data, y_data, 4))

"""
plt.plot(x_data, y_fit, color='red', label='Fitted Curve')
plt.scatter(x_data, y_data, label ='Data with Noise')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Fitting Data to Unknown Equation')
plt.legend()
plt.grid(True)
plt.show()
"""



# Plot the fitted curve
"""
plt.xlabel('x')
plt.ylabel('y')
plt.title('Fitting Data to Unknown Equation')
plt.legend()
plt.grid(True)
plt.show()
"""

"""
print("Fitted Parameters:")
print("a =", a_fit)
print("b =", b_fit)
print("c =", c_fit)
print("d =", d_fit)
"""
