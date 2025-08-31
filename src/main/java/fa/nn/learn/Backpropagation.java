package fa.nn.learn;

import java.util.Objects;

import fa.nn.Layer;
import fa.nn.NeuralNetwork;
import fa.nn.loss.LossFunction;
import fa.nn.util.Preconditions;

public class Backpropagation {
    private NeuralNetwork neuralNetwork;
    private LossFunction loss;

    /* Initialize backpropagation for a neural network. */
    public Backpropagation(NeuralNetwork neuralNetwork, LossFunction loss) {
        Objects.requireNonNull(neuralNetwork, "neuralNetwork");
        Objects.requireNonNull(loss, "lossFunction");

        this.neuralNetwork = neuralNetwork;
        this.loss = loss;
    }

    /*
     * Accumulate per-layer gradients for one (input, expected) using stored
     * activations; requires a prior forward pass on input.
     */
    public void compute(double[] input, double[] expected, double[][][] gradientWeights, double[][] gradientBiases) {
        Objects.requireNonNull(input, "input");
        Objects.requireNonNull(expected, "expected");
        Objects.requireNonNull(gradientWeights, "gradientWeights");
        Objects.requireNonNull(gradientBiases, "gradientBiases");

        Layer[] layers = this.neuralNetwork.getLayers();

        double[][] deltas = this.computeDeltas(expected);
        double[] aPrev = input;

        for (int l = 0; l < layers.length; l++) {
            Layer layer = layers[l];
            int out = layer.getOutputSize();
            int in = layer.getInputSize();

            for (int j = 0; j < out; j++) {
                for (int i = 0; i < in; i++) {
                    gradientWeights[l][j][i] += deltas[l][j] * aPrev[i];
                }

                gradientBiases[l][j] += deltas[l][j];
            }

            aPrev = layer.getState();
        }
    }

    /*
     * Compute deltas for all layers (output→input) using loss and activations.
     */
    public double[][] computeDeltas(double[] expected) {
        Objects.requireNonNull(expected, "expected");

        Layer[] layers = this.neuralNetwork.getLayers();
        double[][] deltas = new double[layers.length][];
        Layer last = layers[layers.length - 1];

        Preconditions.requireVector(expected, last.getOutputSize(), "expected");

        deltas[layers.length - 1] = last.backward(expected, loss);

        for (int l = layers.length - 1; l >= 0; l--) {
            Layer layer = layers[l];

            if (layer == last) {
                continue;
            }

            deltas[l] = layer.backward(deltas[l + 1], layers[l + 1]);
        }

        return deltas;
    }
}
