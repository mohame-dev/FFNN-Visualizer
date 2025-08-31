package fa.nn.learn;

import fa.nn.Layer;
import fa.nn.NeuralNetwork;
import fa.nn.loss.LossFunction;

public class Adam implements Trainable {

    private final NeuralNetwork neuralNetwork;
    private final double learningRate;
    private final double beta1;
    private final double beta2;
    private final double epsilon;

    private Backpropagation backprop;

    private double[][][] gW;
    private double[][] gB;
    private int count;

    private double[][][] mW;
    private double[][][] vW;
    private double[][] mB;
    private double[][] vB;

    private int t;

    public Adam(NeuralNetwork neuralNetwork) {
        this(neuralNetwork, 1e-3, 0.9, 0.999, 1e-7);
    }

    public Adam(NeuralNetwork neuralNetwork, double learningRate, double beta1, double beta2, double epsilon) {
        this.neuralNetwork = neuralNetwork;
        this.learningRate = learningRate;
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.epsilon = epsilon;

        allocateBuffers();
        this.t = 0;
    }

    @Override
    public void setLoss(LossFunction loss) {
        this.backprop = new Backpropagation(this.neuralNetwork, loss);
    }

    @Override
    public void learn(double[] input, double[] expectedOutput) {
        backprop.compute(input, expectedOutput, gW, gB);
        count++;
    }

    @Override
    public void step() {
        if (count == 0) {
            return;
        }

        t++;

        Layer[] layers = neuralNetwork.getLayers();
        double biasCorr1 = 1.0 - Math.pow(beta1, t);
        double biasCorr2 = 1.0 - Math.pow(beta2, t);

        for (int l = 0; l < layers.length; l++) {
            int out = layers[l].getOutputSize();
            int in = layers[l].getInputSize();

            double[][] stepW = new double[out][in];
            double[] stepB = new double[out];

            for (int j = 0; j < out; j++) {
                for (int i = 0; i < in; i++) {
                    double g = gW[l][j][i] / count;
                    mW[l][j][i] = beta1 * mW[l][j][i] + (1.0 - beta1) * g;
                    vW[l][j][i] = beta2 * vW[l][j][i] + (1.0 - beta2) * (g * g);

                    double mHat = mW[l][j][i] / biasCorr1;
                    double vHat = vW[l][j][i] / biasCorr2;

                    stepW[j][i] = learningRate * mHat / (Math.sqrt(vHat) + epsilon);
                }
            }

            for (int j = 0; j < out; j++) {
                double g = gB[l][j] / count;
                mB[l][j] = beta1 * mB[l][j] + (1.0 - beta1) * g;
                vB[l][j] = beta2 * vB[l][j] + (1.0 - beta2) * (g * g);

                double mHat = mB[l][j] / biasCorr1;
                double vHat = vB[l][j] / biasCorr2;

                stepB[j] = learningRate * mHat / (Math.sqrt(vHat) + epsilon);
            }

            layers[l].update(1.0, stepW, stepB);
        }

        reset();
    }

    @Override
    public void reset() {
        if (gW == null || gB == null) {
            return;
        }

        for (int l = 0; l < gW.length; l++) {
            for (int j = 0; j < gW[l].length; j++) {
                java.util.Arrays.fill(gW[l][j], 0.0);
            }

            java.util.Arrays.fill(gB[l], 0.0);
        }

        count = 0;
    }

    private void allocateBuffers() {
        Layer[] layers = neuralNetwork.getLayers();
        int L = layers.length;

        gW = new double[L][][];
        gB = new double[L][];
        mW = new double[L][][];
        vW = new double[L][][];
        mB = new double[L][];
        vB = new double[L][];

        for (int l = 0; l < L; l++) {
            int out = layers[l].getOutputSize();
            int in = layers[l].getInputSize();

            gW[l] = new double[out][in];
            gB[l] = new double[out];

            mW[l] = new double[out][in];
            vW[l] = new double[out][in];
            mB[l] = new double[out];
            vB[l] = new double[out];
        }
    }
}
