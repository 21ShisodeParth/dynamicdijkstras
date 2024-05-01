## Intelligent Routing in Dynamic Graphs

### Parth Shisode <br>
### Professor Shyam Parekh <br>
### EE122 Spring 2024
---

### Abstract
In the real world, networks that can be represented as graphs are very dynamic. They often change according to a pattern. Edge weights representing these networks might shift, but it is entirely possible that edges are added and removed at random due to unforeseen circumstances. For a graph, when trying to compute the shortest path from a source to a destination in a setting where edge weights change with respect to time, it's important to not only consider current edge weights, but what future edge weights are predicted to be. For this project specifically, I'll be creating a framework for a data-based single-agent optimal routing technique.

### Motivation

<br><br><br><br><br><br><br><br><br><br><br><br>


### Literature Review

<br><br><br><br><br><br><br><br><br><br><br><br>



### Implementation
#### Edge Rate Prediction
#### Prediction Error Handling
#### Prediction Correction
#### Final Algorithm

```python
# Define our variables
threshold = ...
real-time data = []
predicted data = []
edge models = []


# Train the models for each edge
for each edge

    edge models[current edge] = curve_fit(historical data[current edge])


# Modify Dijkstra's algorithm to include the model's future predictions
def dynamic dijkstra(model)

    times, shortest paths = …
	heap queue = …

    while heap queue

        current time, current node = heap queue.pop()

        if time > times[node]

            continue

        for each neighbor

            time away = model.predict(time, source, neighbor)
            potential time = time + time away

            if potential_time < times[n]
            
                times[n] = potential time
                shortest paths[neighbor] = shortest paths[node] + i
                heap queue.push(potential time, n)

    return shortest paths


# As time goes on, evaluate how well our trained model is doing and change to online model if necessary
as time increases

    if agent arrived at a node

        next edge = dynamic dijkstra(model)[current node][upcoming edge]

        predicted edge weight = model.predict(current time, next edge)
        append(predicted data, predicted edge weight)

        observed edge weight = DATASTREAM.edge_weight(next edge)
        append(real-time data, observed edge weight)

        error = RMSE(predicted data, real-time data)

        if error > threshold
        
            model = ARIMA.fit(real-time data)     
```

### Python Notebook and Educational Direction

<br><br><br><br><br><br><br><br><br><br><br><br>




### Future Work

<br><br><br><br><br><br><br><br><br><br><br><br>

