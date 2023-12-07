import pytau

__all__ = ["TauTensorFlowCallbacks"]

try:
    import tensorflow.keras as keras
    class TauTensorFlowCallbacks(keras.callbacks.Callback):

        def __init__(self):
            pytau.setNode(0)
            self.train_stack = []
            self.epoch_stack = []
            self.test_stack = []
            self.predict_stack = []
            self.train_batch_stack = []
            self.test_batch_stack = []
            self.predict_batch_stack = []

        def on_train_begin(self, logs=None):
            x = pytau.phase("Training")
            pytau.start(x)
            self.train_stack.append(x)

        def on_train_end(self, logs=None):
            pytau.stop(self.train_stack.pop())        

        def on_epoch_begin(self, epoch, logs=None):
            x = pytau.phase("Epoch {}".format(epoch))
            pytau.start(x)
            self.epoch_stack.append(x)

        def on_epoch_end(self, epoch, logs=None):
            pytau.stop(self.epoch_stack.pop())

        def on_test_begin(self, logs=None):
            x = pytau.phase("Test")
            pytau.start(x)
            self.test_stack.append(x)

        def on_test_end(self, logs=None):
            pytau.stop(self.test_stack.pop())

        def on_predict_begin(self, logs=None):
            x = pytau.phase("Predict")
            pytau.start(x)
            self.predict_stack.append(x)

        def on_predict_end(self, logs=None):
            pytau.stop(self.predict_stack.pop())

        def on_train_batch_begin(self, batch, logs=None):
            x = pytau.phase("Training Batch {}".format(batch))
            pytau.start(x)
            self.train_batch_stack.append(x)

        def on_train_batch_end(self, batch, logs=None):
            pytau.stop(self.train_batch_stack.pop())

        def on_test_batch_begin(self, batch, logs=None):
            x = pytau.phase("Testing Batch {}".format(batch))
            pytau.start(x)
            self.test_batch_stack.append(x)

        def on_test_batch_end(self, batch, logs=None):
            pytau.stop(self.test_batch_stack.pop())

        def on_predict_batch_begin(self, batch, logs=None):
            x = pytau.phase("Predict Batch {}".format(batch))
            pytau.start(x)
            self.predict_batch_stack.append(x)

        def on_predict_batch_end(self, batch, logs=None):
            pytau.stop(self.predict_batch_stack.pop())
except Exception:
    # Do nothing if TensorFlow isn't available
    pass
