# Isaac VR Pipeline

This document summarizes the runtime equipment and data flow for the Isaac VR
human-robot collaboration project.

## System Flow

```mermaid
flowchart LR
    User["User<br/>VR headset, controllers, hands"]

    subgraph VR["VR Input Layer"]
        SteamVR["SteamVR / OpenXR Runtime"]
        XRCore["Isaac XRCore"]
        HandUDP["Hand Tracking UDP<br/>0.0.0.0:5555"]
    end

    subgraph Isaac["Isaac Sim Runtime"]
        Main["v2/main.py<br/>simulation loop"]
        World["Scene Setup<br/>table, cubes, target area"]
        Panda["Panda Robot"]
        PickPlace["PickPlace Controller"]
        Avatar["VRAvatar<br/>head, hands, arm proxies"]
        Grab["VRGrabManager<br/>cube grabbing"]
        GripCam["GripperCamera<br/>optional viewport / recording"]
        Collision["Collision + ErrP Detection"]
        Logger["EventLogger"]
    end

    subgraph Haptics["Haptics Path"]
        HClient["HapticsUdpClient"]
        Bridge["bhaptics_udp_bridge.py<br/>UDP 5005"]
        Tact["bHaptics TactGlove"]
    end

    subgraph Logs["CSV Logs"]
        Markers["v2/errp_markers.csv<br/>sim_time,event,details"]
        Samples["v2/session_samples.csv<br/>distances + human_robot_collision"]
    end

    User --> SteamVR
    User --> HandUDP

    SteamVR --> XRCore
    XRCore --> Main
    HandUDP --> Main

    Main --> World
    Main --> Panda
    Main --> Avatar
    Main --> Grab
    Main --> GripCam

    World --> PickPlace
    Panda --> PickPlace
    PickPlace --> Panda

    Avatar --> Collision
    Panda --> Collision
    World --> Collision

    Avatar --> Grab
    Grab --> World

    Collision --> Logger
    Main --> Logger

    Logger --> Markers
    Logger --> Samples

    Collision --> HClient
    HClient --> Bridge
    Bridge --> Tact
```

## Per-Frame Runtime Sequence

```mermaid
sequenceDiagram
    participant U as User VR Device
    participant XR as SteamVR OpenXR
    participant HT as Hand Tracking UDP
    participant M as v2/main.py Loop
    participant A as VRAvatar
    participant G as VRGrabManager
    participant R as Panda Robot
    participant C as Collision ErrP Logic
    participant L as EventLogger
    participant H as bHaptics UDP

    U->>XR: headset/controller poses
    XR->>M: XR pose input
    HT->>M: pinch/index/thumb points

    loop every simulation frame
        M->>A: update head and hands
        M->>G: update cube grab state
        M->>R: run pick-place controller
        M->>C: check robot, gripper, cube, human collisions
        C->>L: log ErrP markers if detected
        M->>L: log session sample distances
        C->>H: send haptic pulse on collision
    end
```

## Notes

- `v2/session_samples.csv` stores per-frame or interval samples such as hand
  distances and `human_robot_collision`.
- `v2/errp_markers.csv` stores event markers such as episode starts, ErrP
  candidates, collisions, and episode ends.

## TensorBoard CSV Visualization

The CSV logs can be converted into TensorBoard event files for offline graph
inspection.

```bash
python -m pip install tensorboard tensorboardX
python scripts/csv_to_tensorboard.py
tensorboard --logdir runs/isaac_vr_csv
```

The generated TensorBoard logs include:

- `distance/*`: left, right, and minimum hand-to-gripper distances.
- `collision/human_robot_collision`: sampled robot collision flag.
- `events/*`: impulse markers for each ErrP candidate event type.
- `events_cumulative/*`: cumulative counts per event type.
