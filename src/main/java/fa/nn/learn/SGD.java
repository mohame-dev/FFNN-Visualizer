package fa.nn.learn;

import fa.nn.Layer;
import fa.nn.NeuralNetwork;
import fa.nn.loss.LossFunction;

public class SGD implements Trainable {
    private NeuralNetwork neuralNetwork;
    private double learningRate;
    protected int count;
    protected Backpropagation backprop;
    protected double[][][] gradientWeights;
    protected double[][] gradientBiases;

    public SGD(NeuralNetwork neuralNetwork, double learningRate) {
        this.neuralNetwork = neuralNetwork;
        this.count = 0;
        this.learningRate = learningRate;
        this.reset();
    }

    @Override
    public void setLoss(LossFunction loss) {
        this.backprop = new Backpropagation(this.neuralNetwork, loss);
    }

    @Override
    public void learn(double[] input, double exptectedOutput[]) {
        this.backprop.compute(input, exptectedOutput, this.gradientWeights, this.gradientBiases);
        this.count++;
    }

    @Override
    public void step() {
        Layer[] layers = this.neuralNetwork.getLayers();
        int numLayers = this.neuralNetwork.getNumLayers();

        double scale = 1.0 / this.count;

        for (int l = 0; l < numLayers; l++) {
            Layer layer = layers[l];
            layer.update(this.learningRate, mult(this.gradientWeights[l], scale), mult(this.gradientBiases[l], scale));
        }

        this.reset();
    }

    @Override
    public void reset() {
        Layer[] layers = this.neuralNetwork.getLayers();
        int numLayers = this.neuralNetwork.getNumLayers();
        this.gradientWeights = new double[numLayers][][];
        this.gradientBiases = new double[numLayers][];

        for (int l = 0; l < numLayers; l++) {
            Layer layer = layers[l];
            int out = layer.getOutputSize();
            int in = layer.getInputSize();

            this.gradientWeights[l] = new double[out][in];
            this.gradientBiases[l] = new double[out];
        }

        this.count = 0;
    }

    private double[] mult(double[] a, double scale) {
        double[] result = new double[a.length];
        for (int i = 0; i < a.length; i++) {
            result[i] = scale * a[i];
        }
        return result;
    }

    private double[][] mult(double[][] a, double scale) {
        double[][] result = new double[a.length][a[0].length];
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a[0].length; j++) {
                result[i][j] = scale * a[i][j];
            }
        }
        return result;
    }
}
