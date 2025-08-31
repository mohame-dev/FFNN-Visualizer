package fa.nn.initialize;

import java.util.Random;

import fa.nn.Layer;

public class XavierInitializer implements Initializer {
    private Random rand;

    public XavierInitializer(Random rand) {
        this.rand = rand;
    }

    @Override
    public void initialize(Layer layer) {
        int outputSize = layer.getOutputSize();
        int inputSize = layer.getInputSize();

        double[][] weights = new double[outputSize][inputSize];
        double[] biases = new double[outputSize];

        double std = Math.sqrt(2.0 / (inputSize + outputSize));

        for (int j = 0; j < outputSize; j++) {
            for (int i = 0; i < inputSize; i++) {
                weights[j][i] = rand.nextGaussian() * std;
            }
            biases[j] = 0.0;
        }

        layer.set(weights, biases);
    }
}
