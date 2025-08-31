package fa.nn.loss;

public interface LossFunction {
    double loss(double[] predicted, double[] expected);

    double derivative(double predicted, double expected);
}