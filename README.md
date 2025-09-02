# FFNN Visualizer
A **feedforward network (FFNN)** implementation (MLP) for general D-dimensional tasks. The network supports configurable depth (number of hidden layers) and width (neurons per layer), and nonlinear activations; set the input/output layer sizes for multi-feature or multi-target tasks.
The backend is built on Spring Boot, which serves the web UI and streams live training updates via Server-Sent Events (SSE). The project uses this custom FFNN implementation to learn user-defined functions $f(x)$ from sampled points, while the UI shows the learning process in real time: live predictions over $x$ and training/validation loss values emitted every `interval` epochs.

---

## Getting started
- Java 17+ (21 recommended)
- Maven 3.6.3+

### Ubuntu setup
#### 0. Check what's installed (skip install if already satisfied)
```bash
java -version
mvn -version
```

#### 1. Install Java
```bash
sudo apt install openjdk-21-jdk
```
#### 2. Install Maven
```bash
sudo apt install maven
```

---

## Usage
### 1. Running the project
```bash
mvn spring-boot:run
```
Now go to <http://localhost:8080/>!

> **Note:** Customize the network in [`fa.core.Trainer#initialize`](src/main/java/fa/core/Trainer.java).
> Adjust layers/activations, optimizer and loss.

### 2. API
- #### POST `/validate`
    Validates a math expression and samples `(x, y)` points.
  
    **Request body**
    ```json
    {
        "expression": "sin(x) + 0.1*x",
        "xmin": -10,
        "xmax": 10,
        "npoints": 200,
        "epochs": 1000,
        "interval": 10
    }
    ```
    **Response (success)**
    ```json
    {
        "success": true,
        "x": [-50.0, -49.5, -49.0],
        "y": [2500.0, 2450.25, 2401.0]
    }
    ```
    **Response (invalid)**
    ```json
    {
        "success": false,
        "x": [],
        "y": []
    }
    ```

- #### GET `/stream-sse`
    Streams JSON snapshots every `interval` epochs.
  
    **Payload (example)**
    ```json
    {
        "x": [-50.0, -49.5, -49.0],
        "y": [2500.0, 2450.25, 2401.0],
        "epoch": 120,
        "tl": 0.1234,
        "vl": 0.1502
    }
    ```

---

## Examples

### 1. $f(x) = x^2$

<p align="center">
  <img src="assets/x2.webp" alt="Animated fit of f(x)=x^2" width="80%">
</p>

> **Neural network**: 1 → 32 (ReLU) → 32 (ReLU) → 1 (Linear) • Optimizer: Adam • Loss: MSE
> $x \in [-50, 50]$, 1000 samples, 5000 epochs, log every 50 epochs

<details>
    <summary><strong>View neural network</strong></summary>

```java
// layers: 1 → 32 → 32 → 1
Layer[] layers = new Layer[] {
    new Layer(1, 32, new ReLU()),
    new Layer(32, 32, new ReLU()),
    new Layer(32, 1, new Linear())
};

NeuralNetwork nn = new NeuralNetwork(layers);
nn.setup(new Adam(nn), new MSE());
```
</details>

### 2. $f(x) = \sin(x) + \cos^2(x)$

<p align="center">
  <img src="assets/sinx.webp" alt="Animated fit of f(x)=sin(x)+cos^2(x)" width="80%">
</p>

> **Neural network**: 1 → 32 (ReLU) → 32 (ReLU) → 16 (ReLU) → 1 (Linear) • Optimizer: Adam • Loss: MSE
> $x \in [-\pi, \pi]$, 1000 samples, 5000 epochs, log every 50 epochs

<details>
    <summary><strong>View neural network</strong></summary>

```java
// layers: 1 → 32 → 32 → 16 → 1
Layer[] layers = new Layer[] {
    new Layer(1, 32, new ReLU()),
    new Layer(32, 32, new ReLU()),
    new Layer(32, 16, new ReLU()),
    new Layer(16, 1, new Linear())
};

NeuralNetwork nn = new NeuralNetwork(layers);
nn.setup(new Adam(nn), new MSE());
```
</details>

### 3. Standalone example (no web UI)

There is a runnable demo at [`fa.nn.examples.XSquared#main`](src/main/java/fa/nn/examples/XSquared.java), which predicts $f(x) = x^2$ and outputs the following:
```txt
Epoch: 1/1000 - loss: 605435.6850756434 - val_loss: 605435.6850756434
Epoch: 100/1000 - loss: 12955.138135702706 - val_loss: 12955.138135702706
Epoch: 200/1000 - loss: 267.39683602500065 - val_loss: 267.39683602500065
Epoch: 300/1000 - loss: 32.57086878555213 - val_loss: 32.57086878555213
Epoch: 400/1000 - loss: 9.60322911091964 - val_loss: 9.60322911091964
Epoch: 500/1000 - loss: 8.451498810647601 - val_loss: 8.451498810647601
Epoch: 600/1000 - loss: 7.503802103470668 - val_loss: 7.503802103470668
Epoch: 700/1000 - loss: 6.4493578456226475 - val_loss: 6.4493578456226475
Epoch: 800/1000 - loss: 6.72697777758014 - val_loss: 6.72697777758014
Epoch: 900/1000 - loss: 9.10403740510551 - val_loss: 9.10403740510551
Epoch: 1000/1000 - loss: 4.043075966285006 - val_loss: 4.043075966285006
x: [-30.0, -20.0, -4.0, 4.0, 12.0, 30.0]
y: [900.7798470390085, 397.55756550042827, 15.670785133891101, 16.188383915713302, 146.98579432908795, 904.1184049617249]
```

---

## Testing

Unit tests use **JUnit** and live under [`src/test/java/fa`](src/test/java/fa/).

**Run all tests**
```bash
mvn test
```
