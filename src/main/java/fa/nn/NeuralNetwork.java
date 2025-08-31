package fa.nn;

import java.util.Arrays;
import java.util.Objects;
import java.util.Random;

import fa.nn.learn.Dataset;
import fa.nn.learn.Trainable;
import fa.nn.loss.LossFunction;
import fa.nn.util.Preconditions;

/**
 * A Feedforward Neural Network implementation composed of independent layers,
 * where backpropagation is performed by the network.
 *
 * @author Mohamed el Majouti
 * @version 1.0
 */
public class NeuralNetwork {
    private Trainable trainer;
    private LossFunction loss;
    private Layer[] layers;

    /*
     * Construct a feed-forward network from ordered layers; adjacent layers'
     * input/output sizes must match.
     */
    public NeuralNetwork(Layer[] layers) {
        Objects.requireNonNull(layers, "layers");

        if (layers.length == 0) {
            throw new IllegalArgumentException("At least one trainable layer is required.");
        }

        verifyLayers(layers); // Check if all layers are correctly connected

        this.layers = layers;
    }

    /*
     * Configure the trainer and loss; must be called after construction.
     */
    public void setup(Trainable trainer, LossFunction loss) {
        Objects.requireNonNull(trainer, "trainer");
        Objects.requireNonNull(loss, "loss");

        this.trainer = trainer;
        this.loss = loss;
        trainer.setLoss(loss);
    }

    /*
     * Run a forward pass through all layers and return the final output.
     */
    public double[] predict(double[] input) {
        Objects.requireNonNull(input, "input");
        Preconditions.requireVector(input, this.layers[0].getInputSize(), "input");

        double output[] = input;

        for (Layer layer : this.layers) {
            output = layer.forward(output);
        }

        return output;
    }

    /*
     * Convenience fit for scalar features/targets.
     */
    public void fit(double[] x, double[] y, double split, int epochs, int batchSize, boolean verbose, Random rand) {
        Objects.requireNonNull(x, "x");
        Objects.requireNonNull(y, "y");

        double[][] mx = Arrays.stream(x)
                .mapToObj(v -> new double[] { v })
                .toArray(double[][]::new);

        double[][] my = Arrays.stream(y)
                .mapToObj(v -> new double[] { v })
                .toArray(double[][]::new);

        fit(mx, my, split, epochs, batchSize, verbose, rand);
    }

    /*
     * Train with mini-batch gradient descent: split train/val, shuffle each
     * epoch, accumulate per-sample grads, step per batch.
     */
    public void fit(double[][] x, double[][] y, double split, int epochs, int batchSize, boolean verbose, Random rand) {
        Objects.requireNonNull(x, "x");
        Objects.requireNonNull(y, "y");
        Preconditions.requireProbability(split, "split");
        Preconditions.requirePositive(epochs, "epochs");
        Preconditions.requirePositive(batchSize, "batchSize");

        if (x.length == 0 || y.length == 0) {
            throw new IllegalArgumentException("x and y must not be empty");
        }

        if (x.length != y.length) {
            throw new IllegalArgumentException("x and y must have same length");
        }

        if (this.trainer == null || this.loss == null) {
            throw new IllegalStateException("Trainer and LossFunction must be set before calling fit()");
        }

        if (rand == null) {
            rand = new Random();
        }

        Dataset d = new Dataset(x, y, split, rand); // Split data into train and validation data

        for (int epoch = 1; epoch <= epochs; epoch++) {
            this.fitNext(d, batchSize);

            if (verbose && (epoch == 1 || epoch % 100 == 0)) {
                System.out.print("Epoch: " + epoch + "/" + epochs + " ");

                // double[][] yhatTrain = Arrays.stream(d.trainX())
                // .map(this::predict)
                // .toArray(double[][]::new);
                // double trainLoss = this.loss.loss(flatten(yhatTrain), flatten(d.trainY()));
                double trainLoss = this.calculateLoss(d.trainX(), d.trainY());
                System.out.print("- loss: " + trainLoss + " ");

                // double[][] yhatVal = Arrays.stream(d.valX())
                // .map(this::predict)
                // .toArray(double[][]::new);
                // double valLoss = this.loss.loss(flatten(yhatVal), flatten(d.valY()));
                double valLoss = this.calculateLoss(d.trainX(), d.trainY());
                System.out.println("- val_loss: " + valLoss);
            }
        }
    }

    /*
     * Advances training by one epoch on the given dataset using mini-batch SGD with
     * the specified batch size.
     */
    public void fitNext(Dataset d, int batchSize) {
        Objects.requireNonNull(d, "dataset");
        Preconditions.requirePositive(batchSize, "batchSize");

        d.shuffle();
        double[][] xTrain = d.trainX(), yTrain = d.trainY();

        for (int start = 0; start < xTrain.length; start += batchSize) { // Iterate over the mini-batches
            int end = Math.min(start + batchSize, xTrain.length);
            for (int i = start; i < end; i++) {

                this.predict(xTrain[i]); // Forward pass (fills activations needed by learn)

                this.trainer.learn(xTrain[i], yTrain[i]); // Accumulate gradients for this sample
            }

            this.trainer.step(); // Update using averaged gradients
        }
    }

    public double calculateLoss(double[][] x, double[][] y) {
        double[][] yhat = Arrays.stream(x)
                .map(this::predict)
                .toArray(double[][]::new);
        return this.loss.loss(flatten(yhat), flatten(y));
    }

    /* Return the layers in forward order. */
    public Layer[] getLayers() {
        return this.layers;
    }

    /* Return the number of layers. */
    public int getNumLayers() {
        return this.layers.length;
    }

    /* Return the configured trainer, or null if unset. */
    public Trainable getTrainer() {
        return this.trainer;
    }

    /* Return the configured loss function, or null if unset. */
    public LossFunction getLoss() {
        return this.loss;
    }

    /* Save network parameters to a file (not implemented). */
    public void save(String filename) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /* Load network parameters from a file (not implemented). */
    public void load(String filename) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /*
     * Verify adjacent layers are size-compatible; throw if any mismatch is
     * found.
     */
    private void verifyLayers(Layer[] layers) {
        for (int i = 0; i < layers.length - 1; i++) {
            if (layers[i].getOutputSize() != layers[i + 1].getInputSize()) {
                throw new IllegalArgumentException("Layers are not connected correctly.");
            }
        }
    }

    /* Flatten a 2D array to 1D. */
    private double[] flatten(double[][] arr) {
        return Arrays.stream(arr)
                .flatMapToDouble(o -> Arrays.stream(o))
                .toArray();
    }
}
