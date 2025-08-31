package fa.nn.initialize;

import java.util.Random;

import fa.nn.Layer;

public class HeInitializer implements Initializer {
    private Random rand;

    public HeInitializer(Random rand) {
        this.rand = rand;
    }

    @Override
    public void initialize(Layer layer) {
        int outputSize = layer.getOutputSize();
        int inputSize = layer.getInputSize();

        double[][] weights = new double[outputSize][inputSize];
        double[] biases = new double[outputSize];

        for (int j = 0; j < outputSize; j++) {
            for (int i = 0; i < inputSize; i++) {
                weights[j][i] = this.rand.nextGaussian() * Math.sqrt(2.0 / inputSize);
            }

            biases[j] = 0.0;
        }

        layer.set(weights, biases);
    }
}
