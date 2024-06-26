# UAV Trajectory Optimization

The aim of this project is to repeat the experiments given in the following paper and further enhance the algorithms for improved performance.

- H. Bayerlein, P. De Kerret and D. Gesbert, "Trajectory Optimization for 
  Autonomous Flying Base Station via Reinforcement Learning," IEEE 19th 
  International Workshop on Signal Processing Advances in Wireless 
  Communications (SPAWC), 2018, pp. 1-5.

The UAV scenario given the paper is shown below.

<img src="https://github.com/cfoh/UAV-Trajectory-Optimization/assets/51439829/27486986-bcb9-47e0-8e8a-8cc2f810f53a" 
  width="400" height="450">
  
## The Settings

The settings of the scenario are:
- A 15-by-15 grid map.
- The map contains some obstacles (shown in dark grey).
- The UAV makes a trip of 50 steps from the start cell to the final landing cell.
- The start cell is the left-bottom cell (colored in green), 
  and the final landing cell is also the left-bottom cell (marked by `X`).
- The UAV can only move up/down/left/right in each time step from the center of
  one cell to the center of another cell. It cannot move out of the map 
  nor into the obstacle cells.
- There are two stationary users (or called UEs here) on the map. 
  Their locations are marked as blue circles.
- The UAV communicates simultaneously to the two UEs. Due to the obstacles,
  signals at some cells experience non-line-of-sight (NLOS).
  - Clear cells indicate line-of-sight (LOS) communications with both UEs
  - Light grey cells indicate NLOS from a UE
  - Darker grey cells indicate NLOS from both UEs
  - Dark cells indicate obstacles
- The screenshot shows that the UAV (i.e. the red circle) 
  has just completed 10 time steps of its flight time.

## The Objective

The objective of the design is to propose a learning algorithm such that the UAV makes a 50-step trip from the start cell to the final landing cell while providing good communication service to both UEs. 
The paper has proposed two machine learning (ML) algorithms, namely Q-learning and Deep-Q-Network (DQN), to learn the optimal trajectory.
After sufficient training on the algorithms, the authors observe that:
- the UAV is able to discover an optimal region to serve both UEs
- the UAV reaches the optimal region on a short and efficient path
- the UAV avoids flying through the shadowed areas as the communication
  quality in those areas is not excellent
- after reaching the optimal region, the UAV circles around the 
  region to continue to provide good communication service
- the UAV decides when to start returning back to avoid crashing

## The Code

The code is tested with `Python 3.9.19` with the following packages:
```
Package    Version
---------- -------
numpy      1.26.4
pip        24.0
pygame     2.5.2
setuptools 58.1.0
shapely    2.0.4
```

Run the simulation by:
```bash
python uav.py
```

### Scenario Parameters

As the paper does not provide full detail of their scenario setup, we make the following assumptions:
- the map is set to 1590-by-1590 in meters
- the UAV is flying at 40 meters above ground
- For the communication, the thermal noise is -174 dBm per Hz

With the above settings, the rate per UE is in the range between around 35 bps/Hz and 50 bps/Hz which is much higher than that of the paper as illustrated in Fig. 3 in the paper. The paper may have used a much wider area than our considered area, and/or the UAV is flying at a much higher altitude.

### Reward Setting

The paper uses transmission rate as the reward for the learning. Depending on the setup, one can use the rate of downlink or uplink transmissions, or both. For uplink transmission, if the transmissions are not orthogonal (in time or frequency), then transmissions of the UEs will interfere with each other. It is unclear which option is used in the paper, here we use orthogonal transmissions and the transmission can be either uplink or downlink. Without loss of generality, we consider uplink transmissions where in our scenario, there are two IoT devices and a UAV. The mission of the UAV is to collect the data from the IoT devices.

Besides, the paper pointed out that the optimal region to serve both UEs is around the shortest midpoint between the two UEs. However, using sum rate as the reward as indicated in (6) in the paper will not create an optimal region at around the shortest midpoint between the two UEs, instead the optimal regions will be around each UE. To match the optimal trajectory shown in the paper, we use minimum rate of both which creates the optimal region around the shortest midpoint between the two UEs. That is:
```python
# reward = r1 + r2   # sum rate, optimal region at around either UE1 or UE2
reward = min(r1,r2)  # min rate, optimal region at the midpoint of UE1 and UE2
```

Apart from using the rate as the reward, we also add additional rewards so that the UAV will return to the final landing cell at the end of its trip. We apply the following rewards:
- If the UAV returns to the final landing cell before the end of its trip,
  we apply penalty to inform the UAV of its premature returning. The penalty is the last 
  immediate reward times the unused time steps. That is, the earlier the UAV returns, the
  more penalty it will receive, so that it will learn to avoid returning earlier. The paper did not apply this penalty, but we found it useful.
- If the UAV fails to return to the final landing cell at the end of its trip, 
  we apply penalty which is the immediate reward times 10. This way, the UAV will learn to return to the final landing cell at the end of its trip to avoid the penalty. This penalty is also described in the paper, although what penalty to apply is not mentioned.

Note that the paper also applies penalty when the UAV moves outside of the map. However, in our design, we simply do not allow the UAV to move outside of the map.

## The Results

We show the results of the ML performance below. Similar to the observation of the authors, the Q-learning algorithm converged slowly and reached the optimal trajectory at around 800,000 rounds (or episodes). 

We also included the results of SARSA. Interestingly, SARSA fails to discover the optimal path and return to the final landing cell.

<table>
  <tr>
    <th>Learning convergence<br>
        (Q-learning and SARSA)</th>
    <th>Illustration of optimal trajectory with Q-learning<br>
        (reward = 2181.45 after 800,000 rounds)</th>
  </tr>
  <tr>
    <td>
      <img src="https://github.com/cfoh/UAV-Trajectory-Optimization/assets/51439829/82546e65-3633-45df-98d7-d77a930999a8">
    </td>
    <td>
      <img src="https://github.com/cfoh/UAV-Trajectory-Optimization/assets/51439829/ed243570-6838-4b76-b23a-607716c66a8e" width="404">
    </td>
  </tr>
</table>

Q-learning improves very slowly in the last 100,000 rounds:
```
round   reward
======  =======
 ...      ...
690000	2181.02
700000	2181.02
710000	2181.02
720000	2181.17
730000	2181.17
740000	2181.17
750000	2181.45
760000	2181.45
770000	2181.45
780000	2181.45
790000	2181.45
800000	2181.45
 ...      ...
```
