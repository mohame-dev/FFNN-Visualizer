package fa.nn;

import java.util.Arrays;
import java.util.Objects;
import java.util.Random;

import fa.nn.activation.ActivationFunction;
import fa.nn.initialize.Initializer;
import fa.nn.initialize.XavierInitializer;
import fa.nn.loss.LossFunction;
import fa.nn.util.Preconditions;

public class Layer {
    private int inputSize;
    private int outputSize;
    private double[][] weights;
    private double[] biases;
    private double[] z;
    private double[] a;
    private ActivationFunction activationFunction;

    /* Construct a layer with explicit initializer and activation */
    public Layer(int inputSize, int outputSize, ActivationFunction activationFunction, Initializer initializer) {
        if (inputSize < 1 || outputSize < 1) {
            throw new IllegalArgumentException("inputSize and outputSize must be > 0");
        }

        if (activationFunction == null || initializer == null) {
            throw new NullPointerException("activationFunction and initializer must not be null");
        }

        this.inputSize = inputSize;
        this.outputSize = outputSize;

        this.weights = new double[outputSize][inputSize];
        this.biases = new double[outputSize];

        this.z = new double[outputSize];
        this.a = new double[outputSize];

        this.activationFunction = activationFunction;
        initializer.initialize(this);
    }

    /* Construct a layer using Xavier initialization by default. */
    public Layer(int inputSize, int outputSize, ActivationFunction activationFunction) {
        this(inputSize, outputSize, activationFunction, new XavierInitializer(new Random()));
    }

    /* Forward pass */
    public double[] forward(double[] input) {
        Objects.requireNonNull(input, "input");
        Preconditions.requireVector(input, this.inputSize, "input");

        for (int j = 0; j < this.outputSize; j++) {
            this.z[j] = this.biases[j];

            for (int i = 0; i < this.inputSize; i++) {
                this.z[j] += this.weights[j][i] * input[i];
            }

            this.a[j] = this.activationFunction.activate(this.z[j]);
        }

        return Arrays.copyOf(this.a, this.outputSize);
    }

    /* Backward (output layer) */
    public double[] backward(double[] y, LossFunction loss) {
        Objects.requireNonNull(y, "y");
        Objects.requireNonNull(loss, "loss");
        Preconditions.requireVector(y, this.outputSize, "y");

        double[] delta = new double[this.outputSize];

        for (int j = 0; j < this.outputSize; j++) {
            double dc_da = loss.derivative(this.a[j], y[j]);
            double da_dz = this.activationFunction.derivative(this.z[j]);
            delta[j] = dc_da * da_dz;
        }

        return delta;
    }

    /* Backward (hidden layer) */
    public double[] backward(double[] nextDelta, Layer nextLayer) {
        Objects.requireNonNull(nextDelta, "nextDelta");
        Objects.requireNonNull(nextLayer, "nextLayer");
        Preconditions.requireVector(nextDelta, nextLayer.getOutputSize(), "nextDelta");

        if (this.getOutputSize() != nextLayer.getInputSize()) {
            throw new IllegalArgumentException("Layer size mismatch: this.outputSize=" + this.getOutputSize()
                    + ", next.inputSize=" + nextLayer.getOutputSize());
        }

        double[][] nextWeights = nextLayer.getWeights();
        double[] delta = new double[this.outputSize];

        for (int j = 0; j < this.outputSize; j++) {
            double dc_da = 0;
            for (int k = 0; k < nextDelta.length; k++) {
                dc_da += nextDelta[k] * nextWeights[k][j];
            }
            double da_dz = this.activationFunction.derivative(this.z[j]);
            delta[j] = dc_da * da_dz;
        }

        return delta;
    }

    /* Update weight and biases using deltas. */
    public void update(double learningRate, double[] delta, double[] aPrevious) {
        Objects.requireNonNull(delta, "delta");
        Objects.requireNonNull(aPrevious, "aPrevious");
        Preconditions.requireVector(delta, this.outputSize, "delta");
        Preconditions.requireVector(aPrevious, this.inputSize, "aPrevious");

        for (int j = 0; j < this.outputSize; j++) {
            for (int i = 0; i < this.inputSize; i++) {
                double dc_dw = delta[j] * aPrevious[i];
                this.weights[j][i] -= learningRate * dc_dw;
            }

            double dc_db = delta[j];
            this.biases[j] -= learningRate * dc_db;
        }
    }

    /* Update weight and biases using gradients. */
    public void update(double learningRate, double[][] gradientWeights, double[] gradientBiases) {
        Objects.requireNonNull(gradientWeights, "gradientWeights");
        Objects.requireNonNull(gradientBiases, "gradientBiases");
        Preconditions.requireVector(gradientBiases, this.outputSize, "gradientBiases");

        for (int j = 0; j < this.outputSize; j++) {
            for (int i = 0; i < this.inputSize; i++) {
                double dc_dw = gradientWeights[j][i];
                this.weights[j][i] -= learningRate * dc_dw;
            }

            double dc_db = gradientBiases[j];
            this.biases[j] -= learningRate * dc_db;
        }
    }

    /* Replace weights and biases. */
    public void set(double[][] weights, double[] biases) {
        Objects.requireNonNull(weights, "weights");
        Objects.requireNonNull(biases, "biases");
        Preconditions.requireMatrix(weights, this.outputSize, this.inputSize, "weights");
        Preconditions.requireVector(biases, this.outputSize, "biases");

        this.weights = weights;
        this.biases = biases;
    }

    /* Get input size. */
    public int getInputSize() {
        return this.inputSize;
    }

    /* Get output size. */
    public int getOutputSize() {
        return this.outputSize;
    }

    /* Get weight matrix. */
    public double[][] getWeights() {
        return this.weights;
    }

    /* Get bias vector. */
    public double[] getBiases() {
        return this.biases;
    }

    /* Get last activation vector a. */
    public double[] getState() {
        return this.a;
    }
}
