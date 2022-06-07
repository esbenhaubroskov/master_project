We are currently considering how to implement radio communication. This is a reliminary suggestion for how we might implement it.

# Class diagram of the RF simulation environment

The network `Graph` consists of nodes and edges (connections) between nodes. The nodes in the Networkx package can be any hashable type (strings, numbers, user defined objects). In this case the nodes are `RadioTransceiver` objects. The edges can have user-defined attributes. A standard attribute is `'weight'`, which can be used by the Networkx implementation of Dijkstra's algorithm and others. Here, an `RFLink` is used as an attribute of the edge. Each `RFLink` is associated with the `RadioTransceiver`-nodes that define the edge.

The radio and RF links can exist without the network graph. We can possibly have a link between all radios and only add the viable connections to the network.

```mermaid
classDiagram
Graph "1" o-- "*" Edge
Graph "1" o-- "*" Node

Node --|> RadioTransceiver

Edge "1" --> "1" RFLink

Node "2" <-- "1" Edge
Node "1" --> "*" Edge

RadioTransceiver "2" <-- "1" RFLink
RadioTransceiver "1" --> "*" RFLink
RadioTransceiver "1" --> "1" Antenna

RFLink "*" --> "1" PropagationModel

class Graph{
    +nodes
    +edges
}

class Node

class Edge{
    +attribute : RFLink
}

class RadioTransceiver {
    +tx_power : float
    +frequency : float
    +position : numpy.Array
    +gain(az, el) float
    +is_connected(other_radio) bool
    +get_new_link(other_radio) RFLink
}

class RFLink{
    -channel : PropagationModel
    -radios : List
    -radio_ids : List
    +path_loss() float
}

class PropagationModel{
    +MultiPolygon: obstacle_map
    +path_loss(x1, x2, frequency) float
    -get_line_of_sight(x1, x2) bool
}

class Antenna{
    +gain(az, el) float
}

```

# Incorporation of high(er) level logic and radio communication into RMP
Some RMP nodes depend on knowing the position and in some cases velocities of other dornes. We want to make it so that the RMP nodes only update information about the other drone(s) if there is a direct or indirect connection to the other drone. We suggest implementing a drone class containing a radio, RMP nodes and possibly other high-level controllers and devices that may influence the behavior of the drone.

```mermaid
erDiagram

    Drone ||--|| Radio : has
    Drone ||--|| RMPTree : has
    Radio }|--|| Network : uses
    RMPTree ||--|| Radio : uses
```