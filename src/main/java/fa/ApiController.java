package fa;

import java.io.IOException;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.server.ResponseStatusException;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;

import net.objecthunter.exp4j.Expression;
import net.objecthunter.exp4j.ExpressionBuilder;

import fa.core.FunctionSampler;
import fa.core.Trainer;
import fa.dto.PredictionResponse;
import fa.dto.ValidationRequest;
import fa.dto.ValidationResponse;

@RestController
public class ApiController {
    private double[] x;
    private double[] y;
    private int epochs;
    private int interval;

    /*
     * Validates a math expression and prepares sampling/training parameters.
     * Returns a ValidationResponse with sampled x,y on success; false if invalid.
     */
    @PostMapping("/validate")
    public ValidationResponse validate(@RequestBody ValidationRequest request) {
        this.reset();

        Expression expression;

        try {
            // Parse and vvalidate the expression
            expression = new ExpressionBuilder(request.getExpression())
                    .variables("x")
                    .build();

            expression.setVariable("x", 0);
            expression.evaluate();
        } catch (Exception e) {
            // Return no points if the expression is invalid
            System.out.println(e);
            return new ValidationResponse(false);
        }

        this.epochs = request.getEpochs();
        this.interval = request.getInterval();

        // Generate (x, y) samples from the validated expression.
        FunctionSampler fs = new FunctionSampler(
                expression,
                request.getXmin(),
                request.getXmax(),
                request.getNpoints(),
                new Random());

        return new ValidationResponse(true, this.x = fs.x(), this.y = fs.y());
    }

    @GetMapping("/stream-sse")
    public SseEmitter stream() {
        if (!this.isValid()) {
            throw new ResponseStatusException(HttpStatus.PRECONDITION_FAILED, "Data invalid.");
        }

        SseEmitter emitter = new SseEmitter(0L);
        ExecutorService sseMvcExecutor = Executors.newSingleThreadExecutor();

        Trainer t = new Trainer(this.x, this.y, new Random());

        sseMvcExecutor.execute(() -> {
            try {
                for (int epoch = 1; epoch <= epochs; epoch++) {
                    // Process the data for one epoch
                    t.next();

                    if (epoch % this.interval == 0) {
                        // Predict points from (partially) trained model
                        double[] pred = t.predict(x);
                        double tl = t.trainLoss();
                        double vl = t.valLoss();

                        // Return data
                        PredictionResponse data = new PredictionResponse(x, pred, epoch, tl, vl);
                        emitter.send(SseEmitter.event()
                                .name("epoch")
                                .data(data, MediaType.APPLICATION_JSON));
                    }
                }

                // Send a final event so the client knows to close its EventSource
                emitter.send(SseEmitter.event().name("done").data("ok"));
                emitter.complete();
            } catch (IOException io) {
                emitter.complete(); // client likely disconnected
            } catch (Exception e) {
                emitter.completeWithError(e);
            } finally {
                sseMvcExecutor.shutdown();
            }
        });

        emitter.onTimeout(emitter::complete);
        // emitter.onCompletion(() -> {});
        return emitter;
    }

    private void reset() {
        this.x = null;
        this.y = null;
        this.epochs = 0;
        this.interval = 0;
    }

    private boolean isValid() {
        return this.x != null && this.y != null &&
                this.epochs > 0 && this.interval > 0;
    }
}
