# TensorFlow 2.x in TFX

[TensorFlow 2.0 was released in 2019](https://blog.tensorflow.org/2019/09/tensorflow-20-is-now-available.html),
with
[tight integration of Keras](https://www.tensorflow.org/guide/keras/overview),
[eager execution](https://www.tensorflow.org/guide/eager) by default, and
[Pythonic function execution](https://www.tensorflow.org/guide/function), among
other
[new features and improvements](https://www.tensorflow.org/guide/effective_tf2#a_brief_summary_of_major_changes).

This guide covers what works, what doesn't work yet, and how to work effectively
with TensorFlow 2.x in TFX.

## Which version to use?

TFX is compatible with TensorFlow 2.x., and the high-level APIs that existed in
TensorFlow 1.x (particularly Estimators) continue to work.

### Start new projects in TensorFlow 2.x

Since TensorFlow 2.x retains the high-level capabilities of TensorFlow 1.x,
there is no advantage to using the older version on new projects, even if you
don't plan to use the new features.

Therefore, if you are starting a new TFX project, we recommend that you use
TensorFlow 2.x. You may want to update your code later as full support for Keras
and other new features become available, and the scope of changes will be much
more limited if you start with TensorFlow 2.x, rather than trying to upgrade
from TensorFlow 1.x in the future.

### Consider converting existing projects to TensorFlow 2.x

Code written for TensorFlow 1.x is largely compatible with TensorFlow 2.x. and
will continue to work in TFX.

However, in order to take advantage of improvements and new features as they
become available, consider moving existing projects to TensorFlow 2.x.

For more details, see
[this guide for migrating to TensorFlow 2.0](https://www.tensorflow.org/guide/migrate).

## Keras and Estimator: Which API to use?

### Estimator

The Estimator API has been retained in TensorFlow 2.x, but is not the focus of
new features and development. Code written in TensorFlow 1.x or 2.x using
Estimators will continue to work as expected in TFX.

Here is an e2e example using pure Estimator:
[Taxi example (Estimator)](https://github.com/tensorflow/tfx/blob/r0.21/tfx/examples/chicago_taxi_pipeline/taxi_utils.py)

### Keras

The Keras API is the recommended way of building new models in TensorFlow 2.x.
Currently there are two different ways to move forward with Keras in TFX:

*   Keras with model_to_estimator.
*   Native Keras.

Details will be discussed in the following sections.

Note: Full support for all features is in progress, in most cases, keras in TFX
will work as expected. It may not work with Feature Columns and Sparse Features.

## Keras with model_to_estimator

Keras models can be wrapped with the `tf.keras.estimator.model_to_estimator`
function, which allows them to work as if they were Estimators. To use this:

1.  Build a Keras model.
2.  Pass the compiled model into `model_to_estimator`.
3.  Use the result of `model_to_estimator` in Trainer, the way you would
    typically use an Estimator.

```py
# Build a Keras model.
def _keras_model_builder():
  """Creates a Keras model."""
  ...

  model = tf.keras.Model(inputs=inputs, outputs=output)
  model.compile()

  return model


# Write a typical trainer function
def trainer_fn(trainer_fn_args, schema):
  """Build the estimator, using model_to_estimator."""
  ...

  # Model to estimator
  estimator = tf.keras.estimator.model_to_estimator(
      keras_model=_keras_model_builder(), config=run_config)

  return {
      'estimator': estimator,
      ...
  }
```

Other than the user module file of Trainer, rest part of the pipeline remains
unchanged. Here is an e2e example using Keras with model_to_estimator:
[Iris example (model_to_estimator)](https://github.com/tensorflow/tfx/blob/r0.21/tfx/examples/iris/iris_utils.py)

## Native Keras

### Transform

Transform currently has experimental support of Keras models.

The Transform component itself can be used for native Keras without change, the
`preprocessing_fn` definition remains the same, using
[TensorFlow](https://www.tensorflow.org/api_docs/python/tf) and
[tf.Transform](https://www.tensorflow.org/tfx/transform/api_docs/python/tft)
ops.

The serving function and eval function are changed for native Keras, details
will be discussed in the following Trainer and Evaluator sections.

Note: Transformations within the `preprocessing_fn` cannot be applied to the
label feature for training or eval.

### Trainer

To configure native Keras, `GenericExecutor` needs to be set for Trainer
component to replace the default estimator based executor. For details, please
check [here](trainer.md#configuring-trainer-component-with-genericexecutor).

#### Keras Module file with Transform

Training module file must contains a `run_fn` which will be called by
`GenericExecutor`, a typical Keras `run_fn` would look like below:

```python
def run_fn(fn_args: TrainerFnArgs):
  """Train the model based on given args.

  Args:
    fn_args: Holds args used to train the model as name/value pairs.
  """
  tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)

  # Train and eval files contains transformed examples.
  # _input_fn read dataset based on transformed feature_spec from tft.
  train_dataset = _input_fn(fn_args.train_files, tf_transform_output, 40)
  eval_dataset = _input_fn(fn_args.eval_files, tf_transform_output, 40)

  model = _build_keras_model()

  model.fit(
      train_dataset,
      steps_per_epoch=fn_args.train_steps,
      validation_data=eval_dataset,
      validation_steps=fn_args.eval_steps)

  signatures = {
      'serving_default':
          _get_serve_tf_examples_fn(model,
                                    tf_transform_output).get_concrete_function(
                                        tf.TensorSpec(
                                            shape=[None],
                                            dtype=tf.string,
                                            name='examples')),
  }
  model.save(fn_args.serving_model_dir, save_format='tf', signatures=signatures)
```

In above `run_fn`, a serving signature is needed when exporting the trained
model so that model can take raw examples for prediction, a typical serving
function would look like below:

```python
def _get_serve_tf_examples_fn(model, tf_transform_output):
  """Returns a function that parses a serialized tf.Example."""

  # the layer is added as an attribute to the model in order to make sure that
  # the model assets are handled correctly when exporting.
  model.tft_layer = tf_transform_output.transform_features_layer()

  @tf.function
  def serve_tf_examples_fn(serialized_tf_examples):
    """Returns the output to be used in the serving signature."""
    feature_spec = tf_transform_output.raw_feature_spec()
    feature_spec.pop(_LABEL_KEY)
    parsed_features = tf.io.parse_example(serialized_tf_examples, feature_spec)

    transformed_features = model.tft_layer(parsed_features)
    transformed_features.pop(_transformed_name(_LABEL_KEY))

    return model(transformed_features)

  return serve_tf_examples_fn
```

In above serving function, tf.Transform transformations need to be applied to
the raw data for inference, using the `tft.TransformFeaturesLayer` layer.
Previous `_serving_input_receiver_fn` will no longer be needed with Keras.

#### Keras Module file without Transform

Similar to above module file but without the transformations:

```python
def _get_serve_tf_examples_fn(model, schema):

  @tf.function
  def serve_tf_examples_fn(serialized_tf_examples):
    feature_spec = _get_raw_feature_spec(schema)
    feature_spec.pop(_LABEL_KEY)
    parsed_features = tf.io.parse_example(serialized_tf_examples, feature_spec)
    return model(parsed_features)

  return serve_tf_examples_fn


def run_fn(fn_args: TrainerFnArgs):
  schema = io_utils.parse_pbtxt_file(fn_args.schema_file, schema_pb2.Schema())

  # Train and eval files contains raw examples.
  # _input_fn read dataset based on raw feature_spec from schema.
  train_dataset = _input_fn(fn_args.train_files, schema, 40)
  eval_dataset = _input_fn(fn_args.eval_files, schema, 40)

  model = _build_keras_model()

  model.fit(
      train_dataset,
      steps_per_epoch=fn_args.train_steps,
      validation_data=eval_dataset,
      validation_steps=fn_args.eval_steps)

  signatures = {
      'serving_default':
          _get_serve_tf_examples_fn(model, schema).get_concrete_function(
              tf.TensorSpec(shape=[None], dtype=tf.string, name='examples')),
  }
  model.save(fn_args.serving_model_dir, save_format='tf', signatures=signatures)
```

#### [tf.distribute.Strategy](https://www.tensorflow.org/api_docs/python/tf/distribute/Strategy)

Single worker distribution strategy is supported in TFX currently. For
multi-worker strategy, it requires the implementation of TFX Trainer in certain
execution environment has the ability to bring up the cluster of worker
machines, currently the default TFX Trainer doesn't have ability to spawn
multi-worker cluster, Cloud AI Platform support is WIP.

To use distribution strategy, create an appropriate tf.distribute.Strategy and
move the creation and compiling of Keras model inside strategy scope.

For example, replace above `model = _build_keras_model()` with:

```python
  mirrored_strategy = tf.distribute.MirroredStrategy()
  with mirrored_strategy.scope():
    model = _build_keras_model()

  # Rest of the code can be unchanged.
  model.fit(...)
```

[MirroredStrategy](https://www.tensorflow.org/api_docs/python/tf/distribute/MirroredStrategy)
will use CPU if no GPUs are found. To verify it actually uses GPU, enable info
level tensorflow logging:

```python
import logging
logging.getLogger("tensorflow").setLevel(logging.INFO)
```

and you should be able to see `Using MirroredStrategy with devices (...GPUs...)`
in the log.

Note: environment variable `TF_FORCE_GPU_ALLOW_GROWTH=true` might be needed for
GPU out of memory issue.

### Evaluator

ModelValidator and Evaluator are merged with TFMA v0.2x as the new Evaluator
([details](https://github.com/tensorflow/community/blob/master/rfcs/20200117-tfx-combining-model-validator-with-evaluator.md)).
New evaluator can do both single model evaluation or validating it with previous
models. With this change, Pusher now consumes blessing result from Evaluator
instead of ModelValidator.

New Evaluator support Keras as well as Estimator. Previous
`_eval_input_receiver_fn` and eval saved model will no longer be needed with
Keras as Evaluator is based on the same saved model that is used for serving.

For details, please check [here](evaluator.md).

### Examples and Colab

Here are several examples with native Keras:

*   [Iris](https://github.com/tensorflow/tfx/blob/master/tfx/examples/iris/iris_pipeline_native_keras.py)
    ([module file](https://github.com/tensorflow/tfx/blob/master/tfx/examples/iris/iris_utils_native_keras.py)):
    'Hello world' example.
*   [Mnist](https://github.com/tensorflow/tfx/blob/master/tfx/examples/mnist/mnist_pipeline_native_keras.py)
    ([module file](https://github.com/tensorflow/tfx/blob/master/tfx/examples/mnist/mnist_utils_native_keras.py)):
    Image example.
*   [Taxi](https://github.com/tensorflow/tfx/blob/master/tfx/examples/chicago_taxi_pipeline/taxi_pipeline_native_keras.py)
    ([module file](https://github.com/tensorflow/tfx/blob/master/tfx/examples/chicago_taxi_pipeline/taxi_utils_native_keras.py)):
    example with advanced Transform usage.

We also have a per-component
[Keras Colab](https://www.tensorflow.org/tfx/tutorials/tfx/keras) in addition to
[Estimator Colab](https://www.tensorflow.org/tfx/tutorials/tfx/components).
