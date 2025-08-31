package fa.nn.learn;

import fa.nn.loss.LossFunction;

public interface Trainable {
    void learn(double[] input, double exptectedOutput[]);

    void step();

    void reset();

    void setLoss(LossFunction loss);
}
