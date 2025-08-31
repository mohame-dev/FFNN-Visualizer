package fa.nn.activation;

public interface ActivationFunction {
    double activate(double z);

    double derivative(double z);
}
