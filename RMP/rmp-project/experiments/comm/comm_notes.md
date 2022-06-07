
# Comm stretch

Radio max distance config:
```json
"radio_max_distance":{
    "rssi_critical":-50,
    "rssi_no_signal": -70,
    "d_max": 4,
    "w": 1,
    "alpha": 3,
    "beta": 2,
    "distance_threshold": 2,
    "distance_change_coefficient": 0.005,
    "exponential_falloff": 0.999
},

"radio_model":{
    "name":"SimplifiedStochasticPathLoss",
    "parameters":{
        "K": 1e-6,
        "loss_exponent": 4,
        "d_0": 10,
        "std_dev": 4,
        "nlos_attenuation": 16
    }
},
```

Simulation results
```
Elapsed time: 1.627 s
                                 Robot_0    Robot_1
average velocities              0.050689   0.049644
travelled distances            10.869037  10.294503
times of arrivals                    NaN        NaN
average accelerations           1.013591   1.012822
average smoothness trajectory   1.131621   1.131621
policy summary: 2_rmd_ga_radio_link_stretch
```

The drones keep close to the same distance at all times from each other, though they may move collectively closer to one goal or the other.